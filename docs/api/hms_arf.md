# HmsArf

Areal reduction factor helpers for design-storm workflows.

::: hms_commander.HmsArf
    options:
      show_source: true
      heading_level: 2
      show_root_heading: true
      show_root_toc_entry: false
      members_order: source


## Texas 1-day ARF (Asquith 1999)

`HmsArfTexas` implements the circular-watershed and cell-based areal-reduction
factors of USGS WRIR 99-4267 (Table 7, Table 8) for Austin, Dallas and Houston.
It is scoped to the published 1-day duration, 2-year-or-greater recurrence
interval and 0-50 mi radius; out-of-range input raises unless
`extrapolate=True`. Per the TxDOT HDM, applicability diminishes with distance
from those cities and as duration departs from 1 day. It does not edit met files.

```python
from hms_commander import HmsArfTexas
arf = HmsArfTexas.circular_arf("dallas", area_mi2=50.3, recurrence_interval_yr=100)  # 0.85
depth = HmsArfTexas.scale_depth(9.55, "dallas", area_mi2=50.3, recurrence_interval_yr=100)
```

::: hms_commander.HmsArfTexas
    options:
      show_source: false
      heading_level: 3
      show_root_heading: true
      show_root_toc_entry: false
      members_order: source
