"""
TexasStorm - Texas Dimensionless Hyetographs (Empirical, Triangular, L-gamma)

Generates cumulative-rainfall hyetographs from the dimensionless Texas
hyetograph models published by USGS and TxDOT and adopted in the TxDOT
Hydraulic Design Manual (HDM, Chapter 4, Section 13).

Methods:
    - ``empirical``: tabulated percentile curves by storm quartile and storm
      duration class (Williams-Sether and others, 2004, USGS SIR 2004-5075,
      Supplements 4 and 5; Table 4 of that report is the 50th percentile of
      Supplement 5, 0-72 hr).
    - ``triangular``: two-parameter triangular model (Asquith and others,
      2005, TxDOT 0-4194-4, Eq. 1-2; HDM Eq. 4-26 and 4-27).
    - ``lgamma``: L-gamma model p(F) = F**b * exp(c * (1 - F)) (0-4194-4,
      Eq. 11; HDM Eq. 4-28 and Table 4-15).

Explicit choices (no silent defaults):
    - Two different triangular parameter sets are published (see
      ``TRIANGULAR_PARAMETER_SETS``); ``param_set`` must be given.
    - Duration class boundaries (for example exactly 12 hr) are not resolved
      by the sources in a way this module can apply, so a duration that falls
      on a boundary requires an explicit ``duration_class``.
    - The empirical ``quartile`` must be given.

Applicability limits (from the sources):
    - Empirical curves were developed for small watersheds, less than about
      160 square miles (SIR 2004-5075; HDM). A ``UserWarning`` is issued when
      ``drainage_area_sqmi`` exceeds ``MAX_DRAINAGE_AREA_SQMI``.
    - All three models are for storms of at least 1 inch of rainfall; a
      warning is issued for smaller depths.
    - Storm durations up to 72 hr.

Output follows the sibling storm classes: a DataFrame with ``hour``,
``incremental_depth`` and ``cumulative_depth`` columns and a t=0 zero row.

Example:
    >>> from hms_commander import TexasStorm
    >>> hyeto = TexasStorm.generate_hyetograph(
    ...     total_depth_inches=8.0,
    ...     duration_hours=12,
    ...     method='triangular',
    ...     param_set='nws_hourly',
    ...     time_interval_min=60,
    ... )
    >>> round(float(hyeto['cumulative_depth'].iloc[-1]), 6)
    8.0

References:
    Williams-Sether, T., Asquith, W.H., Thompson, D.B., Cleveland, T.G., and
    Fang, X., 2004, Empirical, dimensionless, cumulative-rainfall hyetographs
    developed from 1959-86 storm data for selected small watersheds in Texas:
    USGS Scientific Investigations Report 2004-5075.

    Asquith, W.H., Roussel, M.C., Thompson, D.B., Cleveland, T.G., and Fang,
    X., 2005, Summary of dimensionless Texas hyetographs and distribution of
    storm depth developed for Texas Department of Transportation Research
    Project 0-4194: TxDOT Report 0-4194-4.

    Texas Department of Transportation, 2019, Hydraulic Design Manual,
    Chapter 4, Section 13 (Equations 4-26 to 4-28, Tables 4-13 to 4-16).
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .LoggingConfig import get_logger
from .Decorators import log_call
from ._hyetograph import (
    build_hyetograph_frame,
    incremental_depths_from_cumulative_values,
)

logger = get_logger(__name__)


class TexasStorm:
    """
    Static class for generating Texas dimensionless hyetographs.

    All methods are static - no instantiation required.

    Attributes:
        METHODS: Valid ``method`` values.
        EMPIRICAL_QUARTILES: Valid quartile identifiers ('all' is the
            first- through fourth-quartile combined data group).
        EMPIRICAL_DURATION_CLASSES: Storm duration classes in hours.
        EMPIRICAL_PERCENTILES: Percentiles tabulated in the supplements.
        MAX_DRAINAGE_AREA_SQMI: Approximate applicability limit (160 mi2).
        MIN_STORM_DEPTH_INCHES: Minimum storm depth in the source data (1 in).
        TRIANGULAR_PARAMETER_SETS: The two published triangular parameter sets.
        LGAMMA_PARAMETERS: L-gamma parameters (b, c) by duration class.
    """

    METHODS = ("empirical", "triangular", "lgamma")

    EMPIRICAL_QUARTILES = ("1", "2", "3", "4", "all")
    # '0-72' is tabulated in the source but only selectable explicitly.
    EMPIRICAL_DURATION_CLASSES = ("0-6", "6-12", "12-24", "24-72", "0-72")
    EMPIRICAL_PERCENTILES = (10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90)

    MAX_DRAINAGE_AREA_SQMI = 160.0
    MIN_STORM_DEPTH_INCHES = 1.0
    MAX_DURATION_HOURS = 72.0

    _DATA_FILE = "data/texas_empirical_hyetographs.csv"
    _empirical_cache: Optional[pd.DataFrame] = None

    # Triangular model: p1 = F**2 / a for 0 <= F <= a;
    # p2 = -F**2/b + (2a/b + 2) F - (a**2/b + a) for a < F <= 1.
    # classes: name -> (min_hours, max_hours, a, b)
    TRIANGULAR_PARAMETER_SETS: Dict[str, dict] = {
        "usgs_runoff": {
            "source": (
                "Asquith and others (2005), TxDOT 0-4194-4, Tables 4 and 5 "
                "(Eq. 5-8): runoff-producing storms of at least 1 inch in the "
                "USGS small-watershed database; mean ordinate 59 percent "
                "(0-24 hr) and 55 percent (24-72 hr)."
            ),
            "classes": {
                "0-24": (0.0, 24.0, 0.23, 0.77),
                "24-72": (24.0, 72.0, 0.35, 0.65),
            },
        },
        "nws_hourly": {
            "source": (
                "Asquith and others (2005), TxDOT 0-4194-4, Tables 7-9 (NWS "
                "hourly database, storms of at least 1 inch); identical to "
                "TxDOT HDM 2019 Table 4-13."
            ),
            "classes": {
                "5-12": (5.0, 12.0, 0.02197, 0.97803),
                "13-24": (13.0, 24.0, 0.28936, 0.71064),
                "25-72": (25.0, 72.0, 0.38959, 0.61041),
            },
        },
    }

    # L-gamma: p(F) = F**b * exp(c * (1 - F)).  classes: name -> (min, max, b, c)
    # Source: 0-4194-4 text on L-gamma hyetographs; HDM 2019 Table 4-15.
    LGAMMA_PARAMETERS: Dict[str, Tuple[float, float, float, float]] = {
        "0-12": (0.0, 12.0, 1.262, 1.227),
        "12-24": (12.0, 24.0, 0.783, 0.4368),
        "24-72": (24.0, 72.0, 0.3388, -0.8152),
    }

    # ------------------------------------------------------------------
    # Model evaluation (dimensionless)
    # ------------------------------------------------------------------
    @staticmethod
    def triangular_cumulative(F: Union[float, np.ndarray], a: float, b: float):
        """
        Cumulative depth fraction of the triangular model (Eq. 1-2 / HDM 4-26, 4-27).

        Args:
            F: Elapsed fraction of storm duration, 0 to 1.
            a: Relative duration before peak intensity.
            b: Relative duration after peak intensity (a + b = 1).

        Returns:
            Cumulative fraction of total storm depth, same shape as ``F``.
        """
        F = np.asarray(F, dtype=float)
        p1 = F ** 2 / a
        p2 = -(F ** 2) / b + (2.0 * a / b + 2.0) * F - (a ** 2 / b + a)
        return np.where(F <= a, p1, p2)

    @staticmethod
    def lgamma_cumulative(F: Union[float, np.ndarray], b: float, c: float):
        """
        Cumulative depth fraction of the L-gamma model, p(F) = F**b * exp(c*(1-F)).

        Args:
            F: Elapsed fraction of storm duration, 0 to 1.
            b: Shape parameter (HDM Table 4-15).
            c: Shape parameter (HDM Table 4-15).
        """
        F = np.asarray(F, dtype=float)
        return F ** b * np.exp(c * (1.0 - F))

    # ------------------------------------------------------------------
    # Empirical tables
    # ------------------------------------------------------------------
    @staticmethod
    def _load_empirical() -> pd.DataFrame:
        if TexasStorm._empirical_cache is None:
            path = Path(__file__).parent / TexasStorm._DATA_FILE
            if not path.exists():
                raise FileNotFoundError(
                    f"Texas empirical hyetograph table not found: {path}\n"
                    "This file should be bundled with hms-commander."
                )
            df = pd.read_csv(
                path,
                comment="#",
                dtype={"quartile": str, "duration_class": str},
            )
            TexasStorm._empirical_cache = df
        return TexasStorm._empirical_cache

    @staticmethod
    def get_empirical_curve(
        quartile: Union[int, str],
        duration_class: str,
        percentile: int = 50,
        trimmed: bool = True,
    ) -> pd.DataFrame:
        """
        Return a published empirical curve exactly as tabulated.

        Args:
            quartile: 1, 2, 3, 4, or 'all' (first- through fourth-quartile
                storms combined).
            duration_class: One of ``EMPIRICAL_DURATION_CLASSES``.
            percentile: One of ``EMPIRICAL_PERCENTILES``.
            trimmed: True for Supplement 5 (trimmed and smoothed, the source
                of SIR Table 4); False for Supplement 4 (untrimmed and
                smoothed).

        Returns:
            DataFrame with columns ``duration_pct`` (center of each 2.5-percent
            interval, 2.5 to 97.5) and ``depth_pct`` (cumulative percent of
            storm depth). The curve is not anchored at 0 or 100 percent.

        Raises:
            ValueError: For unknown selectors or when the source reports no
                data ('--') for the requested curve.
        """
        quartile = str(quartile).lower()
        if quartile not in TexasStorm.EMPIRICAL_QUARTILES:
            raise ValueError(
                f"Invalid quartile: '{quartile}'. "
                f"Valid: {TexasStorm.EMPIRICAL_QUARTILES}"
            )
        if duration_class not in TexasStorm.EMPIRICAL_DURATION_CLASSES:
            raise ValueError(
                f"Invalid duration_class: '{duration_class}'. "
                f"Valid: {TexasStorm.EMPIRICAL_DURATION_CLASSES}"
            )
        if percentile not in TexasStorm.EMPIRICAL_PERCENTILES:
            raise ValueError(
                f"Invalid percentile: {percentile}. "
                f"Valid: {TexasStorm.EMPIRICAL_PERCENTILES}"
            )

        df = TexasStorm._load_empirical()
        sel = df[
            (df["supplement"] == (5 if trimmed else 4))
            & (df["quartile"] == quartile)
            & (df["duration_class"] == duration_class)
        ]
        out = pd.DataFrame(
            {
                "duration_pct": sel["interval_pct"].to_numpy(dtype=float),
                "depth_pct": sel[f"p{percentile}"].to_numpy(dtype=float),
            }
        )
        if out["depth_pct"].isna().any():
            raise ValueError(
                f"SIR 2004-5075 reports no data for quartile={quartile}, "
                f"duration_class={duration_class}, percentile={percentile} "
                f"({'Supplement 5' if trimmed else 'Supplement 4'})."
            )
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Class selection and validation
    # ------------------------------------------------------------------
    @staticmethod
    def _select_class(
        duration_hours: float,
        classes: Dict[str, Tuple[float, ...]],
        duration_class: Optional[str],
        what: str,
    ) -> str:
        """Pick a duration class; refuse to guess on boundaries or gaps."""
        if duration_class is not None:
            if duration_class not in classes:
                raise ValueError(
                    f"Invalid duration_class '{duration_class}' for {what}. "
                    f"Valid: {tuple(classes)}"
                )
            lo, hi = classes[duration_class][0], classes[duration_class][1]
            if not (lo <= duration_hours <= hi):
                raise ValueError(
                    f"duration_hours={duration_hours} is outside class "
                    f"'{duration_class}' ({lo}-{hi} hr) for {what}."
                )
            return duration_class

        matches = [
            name
            for name, v in classes.items()
            if name != "0-72" and v[0] <= duration_hours <= v[1]
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(
                f"duration_hours={duration_hours} is not covered by any class "
                f"for {what}: {tuple(classes)}. Pass duration_class explicitly "
                "if it applies."
            )
        raise ValueError(
            f"duration_hours={duration_hours} falls on a boundary between "
            f"classes {matches} for {what}; the sources do not give a rule "
            "this module can apply. Pass duration_class explicitly."
        )

    @staticmethod
    def _check_applicability(
        total_depth_inches: float, drainage_area_sqmi: Optional[float]
    ) -> None:
        if total_depth_inches < TexasStorm.MIN_STORM_DEPTH_INCHES:
            msg = (
                f"Total depth {total_depth_inches} in is below 1 inch; the "
                "Texas hyetograph data are for storms of at least 1 inch."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)
        if drainage_area_sqmi is None:
            logger.info(
                "Texas hyetographs were developed for watersheds smaller than "
                f"about {TexasStorm.MAX_DRAINAGE_AREA_SQMI:.0f} mi2; "
                "drainage_area_sqmi not provided, limit not checked."
            )
        elif drainage_area_sqmi > TexasStorm.MAX_DRAINAGE_AREA_SQMI:
            msg = (
                f"Drainage area {drainage_area_sqmi} mi2 exceeds the approximate "
                f"{TexasStorm.MAX_DRAINAGE_AREA_SQMI:.0f} mi2 limit of the Texas "
                "empirical/triangular/L-gamma hyetograph data (SIR 2004-5075; "
                "HDM 2019)."
            )
            logger.warning(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # Public generator
    # ------------------------------------------------------------------
    @staticmethod
    @log_call
    def generate_hyetograph(
        total_depth_inches: float,
        duration_hours: float,
        method: str,
        time_interval_min: int = 60,
        quartile: Optional[Union[int, str]] = None,
        percentile: int = 50,
        trimmed: bool = True,
        param_set: Optional[str] = None,
        duration_class: Optional[str] = None,
        drainage_area_sqmi: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate a Texas dimensionless hyetograph scaled to a depth and duration.

        Args:
            total_depth_inches: Total storm depth (inches).
            duration_hours: Storm duration (hours), at most 72.
            method: 'empirical', 'triangular', or 'lgamma'.
            time_interval_min: Output time step in minutes (default 60). The
                duration must be a whole multiple of the interval.
            quartile: Required for 'empirical': 1, 2, 3, 4, or 'all'.
            percentile: Empirical percentile (default 50).
            trimmed: Empirical only. True (default) uses SIR 2004-5075
                Supplement 5 (trimmed, smoothed; SIR Table 4 is its median);
                False uses Supplement 4 (untrimmed, smoothed).
            param_set: Required for 'triangular': 'usgs_runoff' or
                'nws_hourly'. See ``TRIANGULAR_PARAMETER_SETS`` for sources.
            duration_class: Optional explicit class. Required when
                ``duration_hours`` falls on a class boundary (for example 12 hr)
                or, for the 'nws_hourly' set, between classes.
            drainage_area_sqmi: Optional watershed area. A ``UserWarning`` is
                issued above ``MAX_DRAINAGE_AREA_SQMI`` (160 mi2).

        Returns:
            pd.DataFrame with columns:
                - 'hour': time from storm start (float)
                - 'incremental_depth': depth in the interval ending at 'hour' (in)
                - 'cumulative_depth': cumulative depth (in)
            Length = duration / interval + 1 (row 0 is the t=0 zero row).
            ``attrs['provenance']`` records the source and selections.

        Raises:
            ValueError: Invalid or ambiguous selections, durations over 72 hr,
                an interval that does not divide the duration, or an empirical
                curve for which the source gives no data.

        Note:
            For 'empirical', the tabulated curve (values at the centers of
            2.5-percent intervals) is linearly interpolated with (0, 0) and
            (100, 100) added as end points. If a published column decreases
            slightly with time (a smoothing artifact in a few columns), the
            running maximum is used and a warning is logged.

        Example:
            >>> hyeto = TexasStorm.generate_hyetograph(
            ...     10.0, 24, 'empirical', quartile=2, duration_class='12-24',
            ...     time_interval_min=30)
            >>> len(hyeto)
            49
        """
        if method not in TexasStorm.METHODS:
            raise ValueError(
                f"Invalid method: '{method}'. Valid: {TexasStorm.METHODS}"
            )
        if total_depth_inches <= 0:
            raise ValueError(f"Total depth must be positive: {total_depth_inches}")
        if not (0 < duration_hours <= TexasStorm.MAX_DURATION_HOURS):
            raise ValueError(
                f"duration_hours must be in (0, {TexasStorm.MAX_DURATION_HOURS:g}]: "
                f"{duration_hours}"
            )
        if time_interval_min <= 0:
            raise ValueError(f"Time interval must be positive: {time_interval_min}")
        n_float = duration_hours * 60.0 / time_interval_min
        n = int(round(n_float))
        if n < 1 or abs(n_float - n) > 1e-9:
            raise ValueError(
                f"Time interval {time_interval_min} min does not evenly divide "
                f"the {duration_hours}-hour duration."
            )

        TexasStorm._check_applicability(total_depth_inches, drainage_area_sqmi)

        F = np.linspace(0.0, 1.0, n + 1)
        provenance: dict = {"method": method, "duration_hours": duration_hours}

        if method == "triangular":
            if param_set not in TexasStorm.TRIANGULAR_PARAMETER_SETS:
                raise ValueError(
                    "param_set is required for the triangular method: "
                    f"{tuple(TexasStorm.TRIANGULAR_PARAMETER_SETS)}. The two "
                    "published sets differ; see TRIANGULAR_PARAMETER_SETS."
                )
            pset = TexasStorm.TRIANGULAR_PARAMETER_SETS[param_set]
            cls = TexasStorm._select_class(
                duration_hours, pset["classes"], duration_class,
                f"triangular/{param_set}",
            )
            _, _, a, b = pset["classes"][cls]
            frac = TexasStorm.triangular_cumulative(F, a, b)
            provenance.update(
                param_set=param_set, duration_class=cls, a=a, b=b,
                source=pset["source"],
            )

        elif method == "lgamma":
            cls = TexasStorm._select_class(
                duration_hours, TexasStorm.LGAMMA_PARAMETERS, duration_class,
                "lgamma",
            )
            _, _, b, c = TexasStorm.LGAMMA_PARAMETERS[cls]
            frac = TexasStorm.lgamma_cumulative(F, b, c)
            provenance.update(
                duration_class=cls, b=b, c=c,
                source="TxDOT 0-4194-4 (L-gamma); HDM 2019 Eq. 4-28, Table 4-15",
            )

        else:  # empirical
            if quartile is None:
                raise ValueError(
                    "quartile is required for the empirical method: "
                    f"{TexasStorm.EMPIRICAL_QUARTILES}"
                )
            classes = {
                name: tuple(float(x) for x in name.split("-")) + (None, None)
                for name in TexasStorm.EMPIRICAL_DURATION_CLASSES
            }
            cls = TexasStorm._select_class(
                duration_hours, classes, duration_class, "empirical"
            )
            curve = TexasStorm.get_empirical_curve(
                quartile, cls, percentile, trimmed
            )
            x = np.concatenate(([0.0], curve["duration_pct"].to_numpy(), [100.0]))
            y = np.concatenate(([0.0], curve["depth_pct"].to_numpy(), [100.0]))
            y_mono = np.maximum.accumulate(y)
            n_adj = int(np.count_nonzero(y_mono != y))
            if n_adj:
                logger.warning(
                    f"Published empirical curve is not monotone at {n_adj} "
                    f"point(s) (max drop {np.max(y_mono - y):.2f} percent); "
                    "running maximum applied."
                )
            frac = np.interp(F * 100.0, x, y_mono) / 100.0
            provenance.update(
                quartile=str(quartile).lower(), duration_class=cls,
                percentile=percentile, trimmed=trimmed,
                monotone_points_adjusted=n_adj,
                source=(
                    "USGS SIR 2004-5075 Supplement "
                    f"{5 if trimmed else 4}"
                ),
            )

        incremental = incremental_depths_from_cumulative_values(
            frac, total_depth_inches=total_depth_inches
        )
        total_check = float(incremental.sum())
        if abs(total_check - total_depth_inches) > 1e-6:
            logger.warning(
                f"Depth conservation warning: expected {total_depth_inches:.6f}, "
                f"got {total_check:.6f}"
            )
        logger.info(
            f"Generated Texas {method} hyetograph: {n} intervals, "
            f"{total_check:.6f} inches total, peak {incremental.max():.3f} inches"
        )

        hyetograph = build_hyetograph_frame(incremental, time_interval_min)
        hyetograph.attrs["provenance"] = provenance
        return hyetograph
