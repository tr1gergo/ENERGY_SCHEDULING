"""Official London Smart Meter and PVGIS data pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from tqdm.auto import tqdm


LONDON_ARCHIVE = "https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip"
LONDON_TARIFF = "https://data.london.gov.uk/download/vqm0d/14855047-44c2-4856-8a48-e5649200e6ce/Tariffs.xlsx"
PVGIS_API = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
RATES = {"Low": 0.0399, "Normal": 0.1176, "High": 0.6720}


def _download(url: str, path: Path, expected_bytes: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == expected_bytes:
        return path
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        resumed = offset and response.status_code == 206
        mode, initial = ("ab", offset) if resumed else ("wb", 0)
        total = initial + int(response.headers.get("content-length", 0))
        with partial.open(mode) as target, tqdm(total=total, initial=initial, unit="B", unit_scale=True, desc=path.name) as bar:
            for chunk in response.iter_content(2**20):
                if chunk:
                    target.write(chunk)
                    bar.update(len(chunk))
    if partial.stat().st_size != expected_bytes:
        raise IOError(f"{path.name} has an unexpected size")
    partial.replace(path)
    return path


def ensure_inputs(raw_dir: str | Path) -> dict[str, Path]:
    raw = Path(raw_dir)
    london = _download(LONDON_ARCHIVE, raw / "london_smart_meter.zip", 795_722_689)
    tariff = _download(LONDON_TARIFF, raw / "london_tariffs.xlsx", 245_384)
    pvgis = raw / "pvgis_london_2013.json"
    if not pvgis.exists():
        response = requests.get(PVGIS_API, params={
            "lat": 51.5074, "lon": -0.1278, "startyear": 2013, "endyear": 2013,
            "pvcalculation": 1, "peakpower": 1, "loss": 14, "angle": 35,
            "aspect": 0, "outputformat": "json", "browser": 0,
        }, timeout=(30, 120))
        response.raise_for_status()
        payload = response.json()
        if "hourly" not in payload.get("outputs", {}):
            raise ValueError("unexpected PVGIS response")
        pvgis.write_text(json.dumps(payload), encoding="utf-8")
    return {"london": london, "tariff": tariff, "pvgis": pvgis}


def load_tariff(path: str | Path) -> pd.Series:
    frame = pd.read_excel(path, sheet_name="Sheet1")
    timestamp = pd.to_datetime(frame["TariffDateTime"], errors="coerce")
    price = frame["Tariff"].map(RATES)
    if timestamp.isna().any() or price.isna().any():
        raise ValueError("unrecognized London tariff data")
    out = pd.Series(price.to_numpy(float), index=timestamp, name="buy_price").sort_index()
    if out.index.has_duplicates:
        raise ValueError("duplicate London tariff timestamps")
    return out


def load_pvgis(path: str | Path) -> pd.Series:
    hourly = pd.DataFrame(json.loads(Path(path).read_text(encoding="utf-8"))["outputs"]["hourly"])
    timestamp = pd.to_datetime(hourly["time"], format="%Y%m%d:%H%M", utc=True).dt.tz_convert("Europe/London").dt.tz_localize(None).dt.floor("h")
    energy = pd.to_numeric(hourly["P"], errors="raise").to_numpy(float) / 2000
    index = np.repeat(timestamp.to_numpy(), 2)
    index[1::2] += np.timedelta64(30, "m")
    return pd.Series(np.repeat(energy, 2), index=pd.DatetimeIndex(index), name="pv_kwh_per_kwp").groupby(level=0).sum().sort_index()


def _columns(columns) -> dict[str, str]:
    lower = {str(x).strip().lower(): str(x) for x in columns}
    def find(test):
        return next(original for name, original in lower.items() if test(name))
    return {
        "id": find(lambda x: "lclid" in x or x == "household_id"),
        "group": find(lambda x: "stdortou" in x or x == "group"),
        "time": find(lambda x: "datetime" in x),
        "kwh": find(lambda x: "kwh" in x),
    }


def load_households(archive_path: str | Path, count: int, group: str, seed: int, max_per_member: int = 2) -> pd.DataFrame:
    """Seeded sample spread across randomly ordered archive partitions."""
    rng, selected, pieces = np.random.default_rng(seed), set(), []
    with ZipFile(archive_path) as archive:
        members = [x for x in archive.namelist() if x.lower().endswith(".csv")]
        members = [members[i] for i in rng.permutation(len(members))]
        for member in tqdm(members, desc="London partitions"):
            if len(selected) >= count:
                break
            with archive.open(member) as stream:
                names = _columns(pd.read_csv(stream, nrows=0).columns)
            ids = set()
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream, usecols=[names["id"], names["group"]], chunksize=250_000, low_memory=False):
                    mask = chunk[names["group"]].astype(str).str.casefold().eq(group.casefold())
                    ids.update(chunk.loc[mask, names["id"]].dropna().astype(str))
            candidates = sorted(ids - selected)
            take = min(count - len(selected), max_per_member, len(candidates))
            chosen = set(rng.choice(candidates, take, replace=False)) if take else set()
            selected.update(chosen)
            if not chosen:
                continue
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream, usecols=list(names.values()), chunksize=250_000, low_memory=False):
                    part = chunk[chunk[names["id"]].astype(str).isin(chosen)].rename(columns={names["id"]: "household_id", names["time"]: "timestamp", names["kwh"]: "demand_kwh"})
                    if part.empty:
                        continue
                    part["timestamp"] = pd.to_datetime(part["timestamp"], errors="coerce")
                    part["demand_kwh"] = pd.to_numeric(part["demand_kwh"], errors="coerce")
                    part = part.dropna(subset=["timestamp", "demand_kwh"])
                    pieces.append(part.loc[part["timestamp"].between("2013-01-01", "2014-01-01", inclusive="left") & part["demand_kwh"].ge(0), ["household_id", "timestamp", "demand_kwh"]])
    if len(selected) < count:
        raise ValueError(f"found {len(selected)} households; requested {count}")
    return pd.concat(pieces, ignore_index=True).drop_duplicates(["household_id", "timestamp"]).sort_values(["household_id", "timestamp"]).reset_index(drop=True)


def balanced_windows(readings: pd.DataFrame, tariff: pd.Series, pv: pd.Series, starts, target: int, seed: int, days: int, qc: dict) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    periods, starts, audit, eligible = days * 48, list(map(pd.Timestamp, starts)), [], {}
    for household, group in readings.groupby("household_id", sort=True):
        demand = group.set_index("timestamp")["demand_kwh"].sort_index()
        annual_load, expected = float(demand.mean() * 48 * 365), float(demand.mean() * periods)
        household_windows, accepted = [], True
        for start in starts:
            index = pd.date_range(start, periods=periods, freq="30min")
            frame = pd.DataFrame({"demand_kwh": demand.reindex(index), "buy_price": tariff.reindex(index), "pv_kwh_per_kwp": pv.reindex(index)}, index=index)
            complete = not frame.isna().any().any()
            total = float(frame.demand_kwh.sum()) if complete else np.nan
            positive = float(frame.demand_kwh.gt(0).mean()) if complete else np.nan
            relative = total / expected if complete and expected else np.nan
            reasons = []
            if not complete:
                reasons.append("incomplete")
            elif positive < qc["minimum_positive_share"]:
                reasons.append("too_many_zero_readings")
            if complete and not qc["minimum_relative_load"] <= relative <= qc["maximum_relative_load"]:
                reasons.append("atypical_load")
            passed = not reasons
            accepted &= passed
            audit.append({"household_id": household, "season": str(start.date()), "complete": complete, "load_kwh": total, "relative_load": relative, "positive_share": positive, "passed": passed, "reason": ";".join(reasons) or "pass"})
            if passed:
                frame.attrs = {"household_id": str(household), "season": str(start.date()), "annual_load_kwh": annual_load, "annual_pv_yield": float(pv.sum())}
                household_windows.append(frame)
        if accepted and len(household_windows) == len(starts):
            eligible[str(household)] = household_windows
    if len(eligible) < target:
        raise ValueError(f"only {len(eligible)} households passed every seasonal check; requested {target}")
    selected = set(np.random.default_rng(seed).choice(sorted(eligible), target, replace=False))
    windows = [window for household in sorted(selected) for window in eligible[household]]
    audit = pd.DataFrame(audit)
    audit["eligible"] = audit.household_id.astype(str).isin(eligible)
    audit["selected"] = audit.household_id.astype(str).isin(selected)
    return windows, audit


def prepare_data(config: dict, quick: bool):
    data = config["data"]["quick" if quick else "full"]
    paths = ensure_inputs(config["paths"]["raw"])
    cache = Path(config["paths"]["processed"]) / f"households_{data['candidates']}_{config['seed']}.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        readings = pd.read_pickle(cache)
    else:
        readings = load_households(paths["london"], data["candidates"], config["data"]["group"], config["seed"])
        readings.to_pickle(cache)
    tariff, pv = load_tariff(paths["tariff"]), load_pvgis(paths["pvgis"])
    windows, audit = balanced_windows(readings, tariff, pv, data["season_starts"], data["households"], config["seed"], config["data"]["window_days"], config["data"]["quality_control"])
    return windows, audit
