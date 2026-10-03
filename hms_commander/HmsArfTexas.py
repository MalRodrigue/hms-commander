"""
HmsArfTexas - Texas 1-day areal-reduction factor (Asquith, 1999)

Implements the areal-reduction factor (ARF) equations published in Table 7 of

    Asquith, W.H., 1999, Areal-Reduction Factors for the Precipitation of the
    1-Day Design Storm in Texas: U.S. Geological Survey Water-Resources
    Investigations Report 99-4267, 81 p.

The same equations are reproduced as Table 4-12 of the TxDOT Hydraulic Design
Manual (Ch. 4, Sec. 13, 09/2019).

Scope (everything outside it is rejected unless ``extrapolate=True``):

- 1-day design storm only (the method rests on daily precipitation data).
- Three study areas only: Austin, Dallas, Houston.
- Circular-watershed radius 0 to 50 mi (area 0 to ~7,850 mi2); non-circular
  watersheds use the report's cell method (distance of each cell to the
  watershed centroid, 0 to 50 mi).
- Recurrence interval of 2 years or greater. The published relation is the
  "2-year or greater" depth-distance relation and is not a function of the
  recurrence interval within that range; the report notes that the true ARF
  does vary with recurrence interval, which this relation does not capture.

HDM caution (Sec. 13, p. 4-64): "the applicability of this method diminishes
the farther away from the Austin, Dallas, or Houston areas the study area is
and as the duration of the design storm increasingly differs from that of
1 day." The report (p. 24) says the same. Use for a watershed far from these
cities, for other durations, or for rainfall events shorter than a day is an
engineering judgement that this module does not validate.

All methods are static - no instantiation required.
"""

import math
from typing import Dict, Sequence, Tuple, Union

import numpy as np

from .LoggingConfig import get_logger
from .Decorators import log_call

logger = get_logger(__name__)

# Each row: (r_lo, r_hi, S2_a, S2_b, ARF_a, ARF_b, ARF_c) with
#   S2(r)  = S2_a  - S2_b  * r                      (mi)
#   ARF(r) = ARF_a - ARF_b * r + ARF_c / r**2        (r = circular radius, mi)
# Transcribed from WRIR 99-4267 Table 7 (p. 54). ARF_c is 0 for the first
# segment, where the published ARF is linear.
_Segment = Tuple[float, float, float, float, float, float, float]

_TABLE7: Dict[str, Tuple[_Segment, ...]] = {
    "austin": (
        (0, 1, 1.0000, 0.1400, 1.0000, 0.0933, 0.0),
        (1, 2, 0.9490, 0.0890, 0.9490, 0.0593, 0.0170),
        (2, 3, 0.8410, 0.0350, 0.8410, 0.0233, 0.1610),
        (3, 4.5, 0.8080, 0.0240, 0.8080, 0.0160, 0.2600),
        (4.5, 9, 0.7750, 0.0167, 0.7750, 0.0111, 0.4828),
        (9, 13, 0.7420, 0.0130, 0.7420, 0.0087, 1.3737),
        (13, 19, 0.7203, 0.0113, 0.7203, 0.0076, 2.5943),
        (19, 28, 0.6950, 0.0100, 0.6950, 0.0067, 5.6427),
        (28, 33, 0.6502, 0.0084, 0.6502, 0.0056, 17.3505),
        (33, 41, 0.6040, 0.0070, 0.6040, 0.0047, 34.1211),
        (41, 50, 0.3717, 0.0013, 0.3717, 0.0009, 164.3052),
    ),
    "dallas": (
        (0, 2, 1.0000, 0.0600, 1.0000, 0.0400, 0.0),
        (2, 4, 0.9670, 0.0435, 0.9670, 0.0290, 0.0440),
        (4, 6, 0.8910, 0.0245, 0.8910, 0.0163, 0.4493),
        (6, 8, 0.8760, 0.0220, 0.8760, 0.0147, 0.6293),
        (8, 12, 0.8460, 0.0183, 0.8460, 0.0122, 1.2693),
        (12, 16, 0.8130, 0.0155, 0.8130, 0.0103, 2.8533),
        (16, 18, 0.7650, 0.0125, 0.7650, 0.0083, 6.9493),
        (18, 24, 0.7200, 0.0100, 0.7200, 0.0067, 11.8093),
        # ARF intercept corrected 0.6800 -> 0.6880 (see _DALLAS_24_27_PRINTED)
        (24, 27, 0.6880, 0.0087, 0.6880, 0.0058, 17.9533),
        (27, 31, 0.6228, 0.0063, 0.6228, 0.0042, 33.8091),
        (31, 50, 0.5563, 0.0041, 0.5563, 0.0027, 55.1070),
    ),
    "houston": (
        (0, 1, 1.0000, 0.1200, 1.0000, 0.0800, 0.0),
        (1, 2, 0.9400, 0.0600, 0.9400, 0.0400, 0.0200),
        (2, 4, 0.8800, 0.0300, 0.8800, 0.0200, 0.1000),
        (4, 7, 0.8667, 0.0267, 0.8667, 0.0178, 0.1711),
        (7, 11, 0.8078, 0.0183, 0.8078, 0.0122, 1.1334),
        (11, 15, 0.7363, 0.0118, 0.7363, 0.0078, 4.0173),
        (15, 20, 0.6800, 0.0080, 0.6800, 0.0053, 8.2360),
        (20, 50, 0.6187, 0.0049, 0.6187, 0.0033, 16.4138),
    ),
}

# Table 7 as printed gives the Dallas 24 <= r <= 27 ARF intercept as 0.6800
# while the S2 intercept in the same row is 0.6880. 0.6800 leaves the ARF
# discontinuous at r = 24 and disagrees with the integral of S2; 0.6880
# restores both. The HDM Table 4-12 repeats the printed 0.6800.
_DALLAS_24_27_PRINTED = 0.6800

STUDY_AREAS = tuple(_TABLE7)
MAX_RADIUS_MI = 50.0
MAX_AREA_MI2 = math.pi * MAX_RADIUS_MI ** 2   # ~7,854 mi2
MIN_RECURRENCE_INTERVAL_YR = 2.0
DURATION_HR = 24.0


class HmsArfTexas:
    """
    Texas 1-day areal-reduction factors per Asquith (1999), USGS WRIR 99-4267.

    ARF = (mean depth over the watershed) / (point depth of the design storm).
    The factor depends on watershed size and shape and on the city whose
    precipitation network defined the depth-distance relation. Multiply a
    1-day, 2-year-or-greater point depth (e.g. NOAA Atlas 14) by the factor.

    Caution: derived from Austin, Dallas and Houston daily data only. The HDM
    states the method's applicability diminishes with distance from those
    cities and as the storm duration departs from 1 day. Out-of-scope input
    raises ``ValueError`` unless ``extrapolate=True``, which logs a warning.

    Example:
        >>> from hms_commander import HmsArfTexas
        >>> arf = HmsArfTexas.circular_arf("dallas", area_mi2=50.3,
        ...                                recurrence_interval_yr=100)
        >>> round(arf, 2)
        0.85
        >>> HmsArfTexas.scale_depth(9.55, "dallas", area_mi2=50.3,
        ...                         recurrence_interval_yr=100)  # doctest: +SKIP
    """

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _city(city: str) -> str:
        key = str(city).strip().lower()
        if key not in _TABLE7:
            raise ValueError(
                f"Unknown study area {city!r}; Asquith (1999) published "
                f"equations only for {', '.join(STUDY_AREAS)}."
            )
        return key

    @staticmethod
    def _check_scope(recurrence_interval_yr, duration_hr, extrapolate: bool) -> None:
        problems = []
        if not np.isfinite(recurrence_interval_yr) or recurrence_interval_yr <= 0:
            raise ValueError("recurrence_interval_yr must be a positive number")
        if recurrence_interval_yr < MIN_RECURRENCE_INTERVAL_YR:
            problems.append(
                f"recurrence interval {recurrence_interval_yr} yr is below the "
                f"{MIN_RECURRENCE_INTERVAL_YR:g}-yr minimum of the published relation"
            )
        if not np.isfinite(duration_hr) or duration_hr <= 0:
            raise ValueError("duration_hr must be a positive number")
        if duration_hr != DURATION_HR:
            problems.append(
                f"duration {duration_hr} hr is not the 1-day ({DURATION_HR:g} hr) "
                "duration the method was derived for"
            )
        HmsArfTexas._report(problems, extrapolate)

    @staticmethod
    def _report(problems, extrapolate: bool) -> None:
        if not problems:
            return
        msg = "; ".join(problems)
        if not extrapolate:
            raise ValueError(
                f"Outside the published scope of Asquith (1999): {msg}. "
                "Pass extrapolate=True to proceed anyway (a warning is logged)."
            )
        logger.warning("Extrapolating Asquith (1999) Texas ARF beyond its published scope: %s", msg)

    @staticmethod
    def _segment(city: str, r: float, extrapolate: bool, as_printed: bool) -> _Segment:
        if not np.isfinite(r) or r < 0:
            raise ValueError(f"distance/radius must be a finite number >= 0 mi, got {r}")
        if r > MAX_RADIUS_MI * (1 + 1e-12):  # tolerate area->radius round-off
            HmsArfTexas._report(
                [f"radius {r:.3f} mi exceeds the published maximum of {MAX_RADIUS_MI:g} mi"],
                extrapolate,
            )
        segs = _TABLE7[city]
        # Segments share their endpoints; the lower segment owns r == lo.
        seg = segs[-1]
        for s in segs:
            if r < s[1]:
                seg = s
                break
        if as_printed and city == "dallas" and seg[0] == 24:
            seg = seg[:4] + (_DALLAS_24_27_PRINTED,) + seg[5:]
        return seg

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @staticmethod
    def radius_from_area(area_mi2: float) -> float:
        """Radius (mi) of the circle with the given area (mi2)."""
        if not np.isfinite(area_mi2) or area_mi2 < 0:
            raise ValueError(f"area_mi2 must be a finite number >= 0, got {area_mi2}")
        return math.sqrt(area_mi2 / math.pi)

    @staticmethod
    @log_call
    def depth_distance(city: str, r_mi: float, *, extrapolate: bool = False) -> float:
        """
        Estimated 2-year-or-greater depth-distance relation S2(r), Table 7.

        S2(r) is the expected ratio of concurrent depth at distance r from
        the point of the design storm to the point depth (dimensionless).

        Args:
            city: 'austin', 'dallas' or 'houston'.
            r_mi: Distance in miles, 0 to 50.
            extrapolate: Allow r > 50 mi (last published segment, warning).

        Returns:
            S2(r), dimensionless.
        """
        c = HmsArfTexas._city(city)
        _, _, a, b, *_ = HmsArfTexas._segment(c, float(r_mi), extrapolate, False)
        return a - b * float(r_mi)

    @staticmethod
    @log_call
    def circular_arf(city: str, area_mi2: float = None, *, radius_mi: float = None,
                     recurrence_interval_yr: float, duration_hr: float = DURATION_HR,
                     extrapolate: bool = False, as_printed: bool = False) -> float:
        """
        ARF for a circular watershed, Table 7 column "ARF2(r)".

        Give either ``area_mi2`` (the circle of equal area is used) or
        ``radius_mi``.

        Valid for: 1-day duration, recurrence interval >= 2 yr, radius
        0-50 mi (area up to ~7,854 mi2), and the three study areas. See the
        module docstring for the HDM caution about use away from Austin,
        Dallas and Houston.

        Args:
            city: 'austin', 'dallas' or 'houston'.
            area_mi2: Watershed area, square miles.
            radius_mi: Radius of the circular watershed, miles.
            recurrence_interval_yr: Design-storm recurrence interval, years
                (required; must be >= 2).
            duration_hr: Design-storm duration, hours (must be 24).
            extrapolate: Permit out-of-range input; logs a warning. Beyond
                50 mi the last published segment is continued.
            as_printed: Use the printed Dallas 24-27 mi ARF intercept
                (0.6800) instead of the continuity-consistent 0.6880.

        Returns:
            Areal-reduction factor, dimensionless (0 to 1).

        Raises:
            ValueError: unknown city, bad/ambiguous size input, or input
                outside the published scope with ``extrapolate=False``.
        """
        c = HmsArfTexas._city(city)
        if (area_mi2 is None) == (radius_mi is None):
            raise ValueError("Provide exactly one of area_mi2 or radius_mi")
        r = HmsArfTexas.radius_from_area(area_mi2) if radius_mi is None else float(radius_mi)
        HmsArfTexas._check_scope(recurrence_interval_yr, duration_hr, extrapolate)
        _, _, _, _, a, b, k = HmsArfTexas._segment(c, r, extrapolate, as_printed)
        if r == 0:
            return 1.0
        return a - b * r + k / r ** 2

    @staticmethod
    @log_call
    def noncircular_arf(city: str, distances_mi: Sequence[float],
                        areas_mi2: Sequence[float], *, recurrence_interval_yr: float,
                        duration_hr: float = DURATION_HR,
                        extrapolate: bool = False) -> float:
        """
        ARF for a non-circular watershed by the report's cell method (Table 8).

        The watershed is divided into cells; each cell's distance to the
        centroid is substituted into S2(r) and the area-weighted mean of
        S2 is the ARF (Asquith 1999, p. 25-26; HDM Sec. 13 steps 1-8).

        Args:
            city: 'austin', 'dallas' or 'houston'.
            distances_mi: Distance from each cell to the watershed centroid.
            areas_mi2: Area of each cell, square miles.
            recurrence_interval_yr, duration_hr, extrapolate: as in
                :meth:`circular_arf`.

        Returns:
            Areal-reduction factor, dimensionless.
        """
        c = HmsArfTexas._city(city)
        d = np.asarray(distances_mi, dtype=float)
        a = np.asarray(areas_mi2, dtype=float)
        if d.ndim != 1 or d.shape != a.shape or d.size == 0:
            raise ValueError("distances_mi and areas_mi2 must be equal-length, non-empty 1-D sequences")
        if not np.all(np.isfinite(a)) or np.any(a <= 0):
            raise ValueError("cell areas must be finite and > 0")
        HmsArfTexas._check_scope(recurrence_interval_yr, duration_hr, extrapolate)
        s2 = np.array([HmsArfTexas.depth_distance(c, x, extrapolate=extrapolate) for x in d])
        return float(np.sum(s2 * a) / np.sum(a))

    @staticmethod
    @log_call
    def scale_depth(depth, city: str, area_mi2: float = None, *, radius_mi: float = None,
                    recurrence_interval_yr: float, duration_hr: float = DURATION_HR,
                    extrapolate: bool = False):
        """
        Multiply a 1-day point depth (any depth unit) by the circular ARF.

        Accepts a scalar, array or pandas object; returns the same kind.
        Parameters other than ``depth`` are as in :meth:`circular_arf`.
        """
        arf = HmsArfTexas.circular_arf(
            city, area_mi2, radius_mi=radius_mi,
            recurrence_interval_yr=recurrence_interval_yr,
            duration_hr=duration_hr, extrapolate=extrapolate)
        return depth * arf

    @staticmethod
    @log_call
    def scale_hyetograph(incremental_depths, city: str, area_mi2: float = None, *,
                         radius_mi: float = None, recurrence_interval_yr: float,
                         duration_hr: float = DURATION_HR, extrapolate: bool = False):
        """
        Scale every ordinate of a 24-hour incremental hyetograph by the ARF.

        The published factor is a single value for the whole 1-day storm, so
        the temporal pattern is preserved and the total depth is scaled by
        the ARF. Accepts a list, numpy array or pandas Series/DataFrame.
        """
        if isinstance(incremental_depths, (list, tuple)):
            incremental_depths = np.asarray(incremental_depths, dtype=float)
        return HmsArfTexas.scale_depth(
            incremental_depths, city, area_mi2, radius_mi=radius_mi,
            recurrence_interval_yr=recurrence_interval_yr,
            duration_hr=duration_hr, extrapolate=extrapolate)
