# Cost-Minimizing Battery Scheduling for Prosumers

This repository reproduces the computational study for the PEAC model family.
It contains three experiments:

1. a paired runtime comparison of the paper's exact algorithms with Gurobi;
2. a price-taking case study of heterogeneous prosumers in a microgrid;
3. a price-forecast robustness study, both purchase-only and with sales.

## Methods represented

- **PEAC-BS-General:** Theorem 5 reduces sales to a purchase-only instance,
  followed by Algorithm 1 (Buying backwards), with an O(n^2) guarantee.
- **PEAC-BS-Ex-General:** Algorithm 2 maintains its moving range maxima with
  monotone deques and reconstructs an optimal schedule in O(nC) time.
- **Comparators:** Gurobi solves the matching LP and MIP with one thread and a
  zero requested proof gap. Every optimal schedule is checked period by period;
  time-limit observations remain visible as censored benchmark outcomes. Main
  timings include model construction, while solver-only time is also recorded.

Small instances are also tested against an independent state-transition dynamic
program. The test suite covers arbitrary purchase and sales bounds, excess
generation, infeasibility, zero objectives, and cumulative energy rounding.

## Empirical inputs

Demand comes from the [Low Carbon London smart-meter trial](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d),
including its dynamic time-of-use signal. Photovoltaic production comes from the
[European Commission PVGIS API](https://re.jrc.ec.europa.eu/pvg_tools/en/), for
central London in 2013. The program downloads the official files once and
streams household partitions directly from the ZIP archive.

The empirical panel contains the same 14 London households in four seasonal
weeks. Selection occurs after documented completeness and load-quality checks;
PV and battery sizes are fixed per household across seasons. Buying prices are
anchored to the observed tariff and aggregate scarcity. Three pre-specified
export-price scenarios test whether the conclusions depend on the payment rule.
Prices are transparent scenario signals, not a market-clearing mechanism.

The case study compares both exact algorithms with no storage and with a myopic
self-consumption policy that stores available PV before exporting it. Community
participation scenarios combine optimized and myopic schedules. The runtime
benchmark covers load-only, load-dominant, balanced, and PV-dominant profiles
without increasing the number of horizon-capacity-seed combinations.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[gurobi,test]"
.\.venv\Scripts\python.exe -m jupyter lab study.ipynb
```

The notebook starts with `QUICK = True`. Run it once to validate the complete
pipeline, then set `QUICK = False` and run all cells for the paper study. The
runtime benchmark checkpoints after each instance.

To run only the computational benchmark, restart the kernel, set `QUICK = False`,
and execute the notebook through Section 3. That section writes the runtime data,
summary tables, and both runtime figures.

Generated CSV files are placed in `results/`; paper-ready PDF and PNG figures
are placed in `figures/`. Raw downloads, caches, results, and figures are ignored
by Git.

## Interpretation

The case study optimizes each participating prosumer independently under common
prices, consistent with the paper's decentralized price-taking interpretation.
It reports incremental value over myopic self-consumption, the cost of block
purchases, community bills, imports, exports, peak import, and ramping.

The purchase-only forecast experiment is a demand-and-storage scenario for the
paper's multiplicative bound. The sales-enabled experiment uses the empirical
prosumer profiles. Both use common autocorrelated error paths across error levels.
With sales, net cost can be negative, so the study reports additive regret
normalized by gross import cost and makes no multiplicative claim.
