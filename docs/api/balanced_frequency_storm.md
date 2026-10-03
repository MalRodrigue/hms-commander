# BalancedFrequencyStorm

HEC-HMS **Frequency Storm** (balanced / nested, alternating-block) hyetograph
generation from a depth-duration-frequency table. This is an external generator,
validated only against the documented HEC-HMS 4.13 fixture cases; see
`tests/fixtures/hms_frequency_storm/README.md`.

Unlike [`FrequencyStorm`](frequency_storm.md) (fixed HCFCD temporal pattern scaled
to one total depth), this class uses the depths at every duration, area
reduction, optional partial-to-annual conversion, log-log interpolation and the
alternating block method.

## Basis and validation boundary

The implementation follows the description of a balanced Frequency Storm in the
[HEC-HMS Technical Reference Manual, §5.2.6 and §5.2.6.1](https://www.hec.usace.army.mil/Software/hec-hms/documentation/HEC-HMS_Technical_Reference_Manual-20231106.pdf)
and the Frequency Storm parameter descriptions in the [HEC-HMS 4.13 User's
Manual, *Precipitation*](https://www.hec.usace.army.mil/confluence/cwmsdocs/hmsum/4.13/precipitation-292095083.html).
The fixture cases also cover the 4.13 release behavior for missing depths and
the re-sort option described in the [4.13 release notes](https://www.hec.usace.army.mil/confluence/hmsdocs/hmsum/4.14/release-notes/v-4-13-0-release-notes).

Tests assert at most 0.001 in difference for every interval and total depth in
all 68 checked HEC-HMS 4.13 cases (the observed maximum is approximately
5e-12 in). This is evidence for those cases, not a claim of general numerical
equivalence across HMS versions or settings.

Not validated or implemented: user-specified area-reduction functions; spatial
distribution by subbasin, precipitation-frequency grids, or depth-area analysis;
metric units; and direct `.met` parsing/writing. The public generator accepts
depth mappings or sequences only. It requires an explicit depth at both the
interval and total duration, so 4.13's automatic interpolation for arbitrary
missing intermediate depths is not exposed. The HEC-HMS re-sort option is not a
public switch; the one fixture used monotonic increments, for which it does not
change the result.

```python
from hms_commander import BalancedFrequencyStorm

depths = {5: 1.17, 10: 1.88, 15: 2.32, 30: 3.20, 60: 4.27, 120: 5.77,
          180: 6.82, 360: 8.63, 720: 10.3, 1440: 12.1}   # Kerrville TX, 100-yr PDS
hyeto = BalancedFrequencyStorm.generate_hyetograph(
    depths, total_duration_min=1440, time_interval_min=15,
    peak_position_pct=50, storm_area_sqmi=0)
```

::: hms_commander.BalancedFrequencyStorm
    options:
      show_source: true
      heading_level: 2
      show_root_heading: true
      show_root_toc_entry: false
      members_order: source
