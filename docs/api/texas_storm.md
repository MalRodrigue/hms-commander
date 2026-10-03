# TexasStorm

Texas dimensionless hyetographs: empirical percentile curves, the triangular
model, and the L-gamma model.

## Usage

```python
from hms_commander import TexasStorm

# Empirical: 2nd-quartile, median curve, 24-hr class '12-24', 8 in
hyeto = TexasStorm.generate_hyetograph(
    total_depth_inches=8.0, duration_hours=24, method="empirical",
    quartile=2, percentile=50, duration_class="12-24", time_interval_min=30,
)

# Triangular: the parameter set must be named (the sources publish two)
hyeto = TexasStorm.generate_hyetograph(
    8.0, 12, "triangular", param_set="nws_hourly", time_interval_min=60,
)

# L-gamma
hyeto = TexasStorm.generate_hyetograph(15.0, 24, "lgamma", duration_class="12-24")
```

The result has the same columns as `ScsTypeStorm` and `Atlas14Storm`
(`hour`, `incremental_depth`, `cumulative_depth`, with a t=0 row). The source
and selections are in `hyeto.attrs["provenance"]`.

## Sources

| Method | Source |
|--------|--------|
| empirical | Williams-Sether and others (2004), USGS SIR 2004-5075, Supplements 4 and 5 (SIR Table 4 is the median of Supplement 5, 0-72 hr) |
| triangular | Asquith and others (2005), TxDOT 0-4194-4, Eq. 1-2; HDM 2019 Eq. 4-26, 4-27 |
| lgamma | TxDOT 0-4194-4; HDM 2019 Eq. 4-28, Table 4-15 |

The tabulated empirical values are in `hms_commander/data/texas_empirical_hyetographs.csv`.
`scripts/texas_hyetograph_transcription.py` compares two independent readings
of the report (PDF text layer and OCR of the page images) and writes the file.

## Selections the caller must make

- **Triangular parameter set.** Two sets are published and they differ:
  `usgs_runoff` (0-4194-4 Tables 4-5: a = 0.23 for 0-24 hr, 0.35 for 24-72 hr;
  runoff-producing storms) and `nws_hourly` (0-4194-4 Tables 7-9 and HDM
  Table 4-13: a = 0.02197, 0.28936, 0.38959 for 5-12, 13-24, 25-72 hr).
  No default is applied.
- **Duration class.** Classes are 0-6, 6-12, 12-24, 24-72 hr (empirical);
  0-12, 12-24, 24-72 hr (L-gamma); and the triangular classes above. A
  duration on a class boundary, or between the `nws_hourly` classes, needs an
  explicit `duration_class`. HDM says to use the class giving the more severe
  runoff for exactly 12 or 24 hr; that choice is left to the caller.
- **Quartile** (empirical): 1, 2, 3, 4, or `"all"`.

## Applicability

- Small watersheds, less than about 160 mi2 (SIR 2004-5075; HDM). Pass
  `drainage_area_sqmi` to get a `UserWarning` above the limit.
- Storms of at least 1 inch of rainfall (warning below), durations up to 72 hr.
- Empirical curves are tabulated at the centers of 2.5-percent intervals;
  generation adds (0, 0) and (100, 100) and interpolates linearly.
- A few published columns decrease slightly with time; the running maximum is
  used and a warning is logged.
- The source reports no data for the 10th and 90th percentiles of
  fourth-quartile 0-6 hr storms; requesting them raises `ValueError`.

::: hms_commander.TexasStorm
    options:
      show_source: true
      heading_level: 2
      show_root_heading: true
      show_root_toc_entry: false
      members_order: source
