"""Spatially-resolved unit-based ON/OFF detection via depth *bands*.

:class:`SpatialOffModel` runs :class:`on_off_detection.on_off.OnOffModel` within
several depth **bands** (regions of the probe over which the units' spike trains
are pooled), then merges the detected OFF periods within and across bands into a
single catalogue of depth x time OFF states.

Two ways of defining the bands are supported (``spatial_params["band_definition"]``):

- ``"fixed_tiled"`` (default): tile the structure's depth extent with fixed-size
  bands. ``band_sizes`` is a list of band sizes (um); each size is one *scale* of a
  multi-scale ladder (use ``None`` for a single whole-structure band). Successive
  bands at a scale overlap by ``band_overlap``; tiling starts from ``tile_start``
  (``"superficial"`` = the minimum depth coordinate, ``"deep"`` = the maximum;
  whether min-depth is anatomically superficial depends on the recording's depth
  convention).
- ``"greedy_fr"``: grow each band shallow->deep until its pooled firing rate exceeds
  ``band_min_fr`` *and* its span exceeds ``band_min_size``; successive bands overlap
  by ``band_fr_overlap`` of pooled FR. This equalises pooled rate across bands but
  yields a single scale.

A soft per-band inclusion floor (``band_min_units`` / ``band_min_keep_fr``) *drops*
(does not error on) bands that are too sparse, so low-rate bands are allowed when the
detection engine tolerates them (e.g. ``sticky`` + ``off_rate_max=0.0``).

Detection parameters can be **shared** across bands (default: ``shared_on_off_params``
deep-copied to every band) or made **adaptive** per band by assigning
:attr:`SpatialOffModel.per_band_on_off_params` (e.g. a per-band off-rate cap scaled by
band unit count). Either way each base method still *fits* its own operating point to
each band's pooled population.
"""

import pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import on_off
from .methods.exceptions import ALL_METHOD_EXCEPTIONS
from .utils import subset_sorted_train

SPATIAL_PARAMS = {
    # -- Band definition ----------------------------------------------------
    # "fixed_tiled" (multi-scale ladder of fixed-size bands) or "greedy_fr".
    "band_definition": "fixed_tiled",
    # fixed_tiled: one entry per scale; band size in um (None -> whole structure).
    "band_sizes": [250.0, 500.0, None],
    "band_overlap": 0.5,  # (fraction) depth overlap between successive bands at a scale
    "tile_start": "superficial",  # "superficial" (min depth) | "deep" (max depth)
    # greedy_fr:
    "band_min_size": 150.0,  # (um) smallest allowed band size
    "band_min_fr": 100.0,  # (Hz) smallest allowed within-band pooled rate (greedy target)
    "band_fr_overlap": 0.75,  # (fraction) pooled-FR overlap between successive greedy bands
    # -- Soft per-band inclusion floor (all definitions; drops sparse bands) -
    "band_min_units": 1,  # drop bands with fewer than this many units
    "band_min_keep_fr": 0.0,  # (Hz) drop bands with pooled FR below this
    # -- Merging of OFF states within and across bands ----------------------
    # Morphological duration cleaning: drop short band-OFFs *before* merging, and
    # (optionally) short *merged* OFFs after merging. See run_off_df.
    "min_band_off_duration": 0.03,  # (s) remove band-OFFs shorter than this before merging
    # (s) drop merged OFFs whose UNION duration (full extent) is shorter than this, after
    # merging. None / 0.0 / <= min_band_off_duration all mean "no post-merge filter"
    # (the union is always >= every constituent band-OFF, hence >= the pre-merge floor).
    "min_merged_off_duration": None,
    "nearby_off_max_time_diff": 3,  # (s) coarse temporal pre-filter for merge candidates
    "min_shared_duration_overlap": 0.02,  # (s) min temporal overlap to merge
    "min_depth_overlap": 50,  # (um) min depth overlap to merge
}


def _tile_band_scale(depth_min, depth_max, size, overlap, tile_start):
    """Return [(lo, hi), ...] tiling [depth_min, depth_max] with fixed-size bands.

    ``size`` is the band depth (um); ``None`` or a size spanning the whole structure
    yields a single band. Successive bands step by ``size * (1 - overlap)`` and are
    clipped to the structure extent. ``tile_start`` selects the end to start from.
    """
    full = depth_max - depth_min
    if size is None or size >= full:
        return [(depth_min, depth_max)]
    step = max(size * (1.0 - overlap), 1e-6)
    bands = []
    if tile_start == "superficial":
        lo = depth_min
        while lo < depth_max:
            hi = min(lo + size, depth_max)
            bands.append((lo, hi))
            if hi >= depth_max:
                break
            lo += step
    elif tile_start == "deep":
        hi = depth_max
        while hi > depth_min:
            lo = max(hi - size, depth_min)
            bands.append((lo, hi))
            if lo <= depth_min:
                break
            hi -= step
    else:
        raise ValueError(
            f"Unrecognized tile_start={tile_start!r}; use 'superficial' or 'deep'."
        )
    return bands


def rect_union_area(boxes):
    """Exact area of the union of axis-aligned rectangles (geometry-agnostic).

    ``boxes`` is an (N, 4) array of ``[x0, x1, y0, y1]`` (here x=time s, y=depth um).
    Returns the union area in x*y units (s*um) via coordinate compression -- the true
    footprint of a merged OFF (union of its constituent band boxes), which is <= the
    bounding-box area ``(x1-x0)*(y1-y0)`` whenever the boxes don't tile a full rectangle.
    """
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 2 or boxes.shape[0] == 0:
        return 0.0
    xs = np.unique(np.concatenate([boxes[:, 0], boxes[:, 1]]))
    ys = np.unique(np.concatenate([boxes[:, 2], boxes[:, 3]]))
    area = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        xm = 0.5 * (x0 + x1)
        cov = boxes[(boxes[:, 0] <= xm) & (boxes[:, 1] >= xm)]
        if not len(cov):
            continue
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            ym = 0.5 * (y0 + y1)
            if np.any((cov[:, 2] <= ym) & (cov[:, 3] >= ym)):
                area += (x1 - x0) * (y1 - y0)
    return float(area)


def _run_detection(
    band_row,
    band_trains_list,
    band_cluster_ids,
    bouts_df,
    on_off_method,
    on_off_params,
    verbose=False,
):
    band_id = band_row.name
    if verbose:
        print(f'Run band {band_id}, depth={band_row["band_lo"]}-{band_row["band_hi"]}')

    on_off_model = on_off.OnOffModel(
        band_trains_list,
        bouts_df,
        cluster_ids=band_cluster_ids,
        method=on_off_method,
        params=on_off_params,
        verbose=verbose,
    )
    try:
        band_on_off_df, band_output_info = on_off_model.run()
        band_output_info["raised_exception"] = False
        band_output_info["exception"] = None
    except ALL_METHOD_EXCEPTIONS as e:
        print(
            f"Caught the following exception for band="
            f"{band_row['band_lo']}-{band_row['band_hi']}"
        )
        print(e)
        band_on_off_df = pd.DataFrame()
        band_output_info = {
            "raised_exception": True,
            "exception": None,
            "state": e,
        }

    # Store band information on every detected period.
    band_on_off_df["band_id"] = band_id
    band_on_off_df["lo"] = band_row["band_lo"]
    band_on_off_df["hi"] = band_row["band_hi"]
    band_on_off_df["span"] = band_row["band_span"]
    band_on_off_df["scale"] = band_row["band_scale"]
    band_output_info["band_id"] = band_id
    band_output_info["lo"] = band_row["band_lo"]
    band_output_info["hi"] = band_row["band_hi"]
    band_output_info["span"] = band_row["band_span"]
    band_output_info["scale"] = band_row["band_scale"]

    return band_on_off_df, band_output_info


class SpatialOffModel:
    """Spatially-resolved ON/OFF detection via depth bands (see module docstring).

    Args:
            trains_list (list of array-like): Sorted MUA spike times for each cluster
            cluster_depths (array-like): Depth of each cluster in um
            bouts_df (pd.DataFrame): Bouts of interest (must contain 'start_time',
                    'end_time', 'duration', 'state'). Only spikes within these bouts
                    are used (cut-and-concatenate per cluster).

    Kwargs:
            cluster_ids (array-like): Cluster ids. (default None)
            on_off_method: Method used for ON/OFF detection within each band
                    (default "hmmem"; "sticky" recommended for banded use).
            on_off_params: Dict of parameters passed to the on/off detection
                    algorithm within each band (the *shared* params). (default None)
            spatial_params: Dict of band-definition and merging parameters. See
                    :data:`SPATIAL_PARAMS`. (default None -> all defaults)
            n_jobs (int or None): Parallel jobs. No parallelization if None or 1.
            verbose (bool): Default True
    """

    def __init__(
        self,
        trains_list,
        cluster_depths,
        bouts_df,
        cluster_ids=None,
        on_off_method="hmmem",
        on_off_params=None,
        spatial_params=None,
        n_jobs=1,
        verbose=True,
    ):
        self.trains_list = [
            subset_sorted_train(bouts_df, np.sort(train)) for train in trains_list
        ]
        if cluster_ids is not None:
            assert len(cluster_ids) == len(trains_list)
            self.cluster_ids = np.array(cluster_ids)
        else:
            self.cluster_ids = np.array(["" for i in range(len(trains_list))])
        assert all(
            c in bouts_df.columns
            for c in ["start_time", "end_time", "duration", "state"]
        )
        assert bouts_df.duration.sum(), "Empty bouts"
        self.bouts_df = bouts_df
        self.method = on_off_method
        # Params
        self._per_band_on_off_params = None
        self.shared_on_off_params = on_off_params
        #
        self.verbose = verbose
        self.n_jobs = n_jobs
        #
        self._spatial_params = None
        self.spatial_params = spatial_params if spatial_params is not None else {}
        # Depths
        assert len(cluster_depths) == len(self.cluster_ids)
        self.cluster_depths = np.array(cluster_depths)
        # Spatial pooling info
        self.cluster_firing_rates = self.get_cluster_firing_rates()
        sumFR = float(np.sum(self.cluster_firing_rates))
        if sumFR <= 0:
            raise ValueError("No spikes within bouts; cannot run detection.")
        if sumFR < self.spatial_params["band_min_fr"]:
            # Not fatal: low-FR bands are allowed when the engine tolerates them
            # (e.g. sticky + off_rate_max=0.0). Only the greedy_fr definition uses
            # band_min_fr as a growth target.
            print(
                f"Note: cumulative FR={sumFR:.1f}Hz < band_min_fr="
                f"{self.spatial_params['band_min_fr']}Hz; bands may be sparse."
            )
        self.initialize_bands_df()
        # Output
        self.all_bands_on_off_df = None  # Pre-merging
        self.off_df = None  # Final, post-merging
        self.band_output_infos = None  # {<band_id>: <band_output_info>}

    def get_cluster_firing_rates(self):
        total_duration = self.bouts_df.duration.sum()
        return np.array([len(train) / total_duration for train in self.trains_list])

    @property
    def spatial_params(self):
        if self._spatial_params is None:
            raise ValueError("Spatial params were not assigned")
        return self._spatial_params

    @spatial_params.setter
    def spatial_params(self, params):
        unrecognized_params = set(params.keys()) - set(SPATIAL_PARAMS.keys())
        if len(unrecognized_params):
            raise ValueError(
                f"Unrecognized parameter keys for spatial algorithm: "
                f"{unrecognized_params}.\n\n"
                f"Default (recognized) parameters for spatial algo: {SPATIAL_PARAMS}"
            )
        missing_params = set(SPATIAL_PARAMS.keys()) - set(params.keys())
        if len(missing_params):
            print(f"Setting self.spatial_params: Use default value for: {missing_params}")
        self._spatial_params = {k: v for k, v in SPATIAL_PARAMS.items()}
        self._spatial_params.update(params)
        print(f"Spatial params: {self._spatial_params}")

    @property
    def per_band_on_off_params(self):
        if self._per_band_on_off_params is None:
            return [
                deepcopy(self.shared_on_off_params) for _ in self.bands_df.itertuples()
            ]
        if not len(self._per_band_on_off_params) == len(self.bands_df):
            raise ValueError(
                "Number of parameter dictionaries doesn't match number of bands."
            )
        return self._per_band_on_off_params

    @per_band_on_off_params.setter
    def per_band_on_off_params(self, per_band_on_off_params):
        if not len(per_band_on_off_params) == len(self.bands_df):
            raise ValueError(
                "Number of parameter dictionaries doesn't match number of bands."
            )
        print(f"Setting custom per-band parameters for N={len(per_band_on_off_params)} bands")
        self._per_band_on_off_params = deepcopy(per_band_on_off_params)

    @property
    def bands_df(self):
        return self._bands_df

    @bands_df.setter
    def bands_df(self, bands_df):
        assert all(c in bands_df.columns for c in ["band_lo", "band_hi", "band_span"])
        bands_df = bands_df.copy()
        if "band_scale" not in bands_df.columns:
            bands_df["band_scale"] = bands_df["band_span"]

        bands_df["band_cluster_indices"] = bands_df.apply(
            lambda row: np.where(
                np.logical_and(
                    row.band_lo <= self.cluster_depths,
                    self.cluster_depths <= row.band_hi,
                )
            )[0].astype(int),
            axis=1,
        )
        bands_df["band_cluster_ids"] = bands_df.apply(
            lambda row: self.cluster_ids[row.band_cluster_indices], axis=1
        )
        bands_df["band_sumFR"] = bands_df.apply(
            lambda row: np.sum(self.cluster_firing_rates[row.band_cluster_indices]),
            axis=1,
        )

        self._bands_df = bands_df

    def initialize_bands_df(self):
        band_definition = self.spatial_params["band_definition"]
        if band_definition == "greedy_fr":
            raw = self._build_greedy_fr_bands()
        elif band_definition == "fixed_tiled":
            raw = self._build_fixed_tiled_bands()
        else:
            raise ValueError(
                f"Unrecognized band_definition={band_definition!r}; "
                f"use 'fixed_tiled' or 'greedy_fr'."
            )
        bands_df = pd.DataFrame(
            raw, columns=["band_lo", "band_hi", "band_span", "band_scale"]
        ).reset_index(drop=True)
        self.bands_df = bands_df  # setter assigns clusters/sumFR
        self._apply_band_floor()
        if self.verbose:
            print(f"Defined N={len(self.bands_df)} bands ({band_definition}).")

    def _build_fixed_tiled_bands(self):
        """Multi-scale ladder of fixed-size tiled bands. Returns (lo, hi, span, scale)."""
        depth_min = float(self.cluster_depths.min())
        depth_max = float(self.cluster_depths.max())
        overlap = self.spatial_params["band_overlap"]
        tile_start = self.spatial_params["tile_start"]
        full = depth_max - depth_min
        raw = []
        seen = set()
        for size in self.spatial_params["band_sizes"]:
            scale = "whole" if (size is None or size >= full) else float(size)
            for lo, hi in _tile_band_scale(depth_min, depth_max, size, overlap, tile_start):
                key = (round(lo, 3), round(hi, 3))
                if key in seen:
                    continue
                seen.add(key)
                raw.append((lo, hi, hi - lo, scale))
        return raw

    def _build_greedy_fr_bands(self):
        """Grow bands shallow->deep to a pooled-FR target. Returns (lo, hi, span, scale)."""
        SPATIAL_RES = 20

        band_min_fr = self.spatial_params["band_min_fr"]
        band_min_size = self.spatial_params["band_min_size"]
        band_fr_overlap = self.spatial_params["band_fr_overlap"]

        depths = self.cluster_depths
        bins = np.arange(depths.min(), depths.max() + SPATIAL_RES, SPATIAL_RES)
        depthFR, bins = np.histogram(depths, weights=self.cluster_firing_rates, bins=bins)
        cumFR = np.cumsum(depthFR)

        lo_idx = 0
        hi_idx = 0
        raw = []
        while hi_idx < len(bins) - 1:
            idx_above = np.where(
                ((cumFR - cumFR[lo_idx]) > band_min_fr)
                & ((bins[:-1] - bins[lo_idx]) > band_min_size)
            )[0]
            if not len(idx_above):
                hi_idx = len(bins) - 1
                lo_idx = np.where(
                    ((cumFR - cumFR[-1]) < -band_min_fr)
                    & ((bins[:-1] - bins[hi_idx]) < -band_min_size)
                )[0][-1]
            else:
                hi_idx = idx_above[0]

            lo, hi = float(bins[lo_idx]), float(bins[hi_idx])
            raw.append((lo, hi, hi - lo, "greedy"))

            lo_cumFR = cumFR[lo_idx]
            hi_cumFR = cumFR[min(hi_idx, len(cumFR) - 1)]
            lo_idx = np.where(
                cumFR > (hi_cumFR - lo_cumFR) * (1 - band_fr_overlap) + lo_cumFR
            )[0][0]
        return raw

    def _apply_band_floor(self):
        """Drop bands below the soft inclusion floor (too few units / too low FR)."""
        df = self._bands_df
        min_units = self.spatial_params["band_min_units"]
        min_fr = self.spatial_params["band_min_keep_fr"]
        n_units = df["band_cluster_indices"].apply(len)
        keep = (n_units >= min_units) & (df["band_sumFR"] >= min_fr)
        dropped = int((~keep).sum())
        if dropped and self.verbose:
            print(
                f"Dropping {dropped}/{len(df)} bands below floor "
                f"(band_min_units={min_units}, band_min_keep_fr={min_fr})."
            )
        self._bands_df = df[keep].reset_index(drop=True)
        if not len(self._bands_df):
            raise ValueError(
                "All bands dropped by the per-band floor; relax band_min_units / "
                "band_min_keep_fr or widen the bands."
            )

    def get_band_trains(self, band_row):
        """Return band_trains_list for a row of ``self.bands_df``."""
        assert "band_cluster_indices" in band_row
        return [
            self.trains_list[cluster_idx]
            for cluster_idx in band_row["band_cluster_indices"]
        ]

    def get_band_cluster_ids(self, band_row):
        """Return band_cluster_ids for a row of ``self.bands_df``."""
        assert "band_cluster_indices" in band_row
        ids = band_row["band_cluster_ids"]
        assert np.all(ids == self.cluster_ids[band_row["band_cluster_indices"]])
        return ids

    def get_member_boxes(self, off_row):
        """Constituent band-OFF boxes of a merged OFF (row of ``self.off_df``).

        Returns an (N, 4) array of ``[start_time, end_time, lo, hi]`` (s, s, um, um) --
        the per-band rectangles that were merged into ``off_row``. Their union is the
        merged OFF's true depth x time footprint (vs. the single bounding box stored in
        ``lo``/``hi``/``union_start_time``/``union_end_time``). The boxes are the
        band-OFFs' own detected extents.
        """
        idxs = off_row["merged_band_offs_indices"]
        sub = self.all_bands_on_off_df.loc[
            idxs, ["start_time", "end_time", "lo", "hi"]
        ]
        return sub.to_numpy(dtype=float)

    def _add_union_area(self):
        """Add a ``union_area`` column (s*um) = true footprint of each merged OFF."""
        if not len(self.off_df):
            return
        self.off_df["union_area"] = [
            rect_union_area(self.get_member_boxes(row))
            for _, row in self.off_df.iterrows()
        ]

    def _filter_merged_by_duration(self):
        """Post-merge morphological cleaning: drop merged OFFs whose UNION span is short.

        Tests ``union_duration`` (the merged OFF's full extent, earliest start -> latest
        end) against ``min_merged_off_duration`` (``None`` -> no-op). Distinct from the
        pre-merge ``min_band_off_duration`` floor applied to the raw band-OFFs in
        :meth:`_merge_all_bands_offs`. Because the union is always >= every constituent
        band-OFF (hence >= the pre-merge floor), a post-merge floor only removes anything
        when it *exceeds* ``min_band_off_duration``.
        """
        min_dur = self.spatial_params["min_merged_off_duration"]
        if min_dur is None or not len(self.off_df):
            return
        n0 = len(self.off_df)
        self.off_df = self.off_df[
            self.off_df["union_duration"] >= min_dur
        ].reset_index(drop=True)
        if self.verbose and len(self.off_df) < n0:
            print(
                f"Post-merge duration filter (union_duration >= {min_dur}s): "
                f"kept {len(self.off_df)}/{n0} merged OFFs."
            )

    def dump(self, filepath):
        assert not Path(filepath).exists()
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def run(self):
        self.run_all_bands_on_off_df()
        self.run_off_df()
        return self.off_df

    def run_all_bands_on_off_df(self):
        print(f"Run on-off detection for each band (N={len(self.bands_df)})")

        per_band_on_off_params = self.per_band_on_off_params

        if self.n_jobs == 1:
            on_off_dfs = []
            band_output_infos = {}
            for i, (band_id, band_row) in tqdm(list(enumerate(self.bands_df.iterrows()))):
                band_on_off_df, band_output_info = _run_detection(
                    band_row,
                    self.get_band_trains(band_row),
                    self.get_band_cluster_ids(band_row),
                    self.bouts_df,
                    self.method,
                    per_band_on_off_params[i],
                    self.verbose,
                )
                on_off_dfs.append(band_on_off_df)
                band_output_infos[band_id] = band_output_info
        else:
            from joblib import Parallel, delayed

            on_off_dfs, output_infos = zip(
                *Parallel(n_jobs=self.n_jobs, backend="multiprocessing")(
                    delayed(_run_detection)(
                        band_row,
                        self.get_band_trains(band_row),
                        self.get_band_cluster_ids(band_row),
                        self.bouts_df,
                        self.method,
                        per_band_on_off_params[i],
                        self.verbose,
                    )
                    for i, (band_id, band_row) in enumerate(self.bands_df.iterrows())
                )
            )
            band_output_infos = {
                band_id: output_info
                for (band_id, _), output_info in zip(
                    self.bands_df.iterrows(), output_infos
                )
            }

        dfs_to_concat = [df for df in on_off_dfs if df is not None and len(df)]
        if len(dfs_to_concat):
            self.all_bands_on_off_df = pd.concat(dfs_to_concat).reset_index(drop=True)
        else:
            self.all_bands_on_off_df = pd.DataFrame()
        self.band_output_infos = band_output_infos
        print(
            f"Done. Found N={len(self.all_bands_on_off_df)} ON and OFF periods "
            f"across bands."
        )

        return self.all_bands_on_off_df, self.band_output_infos

    def run_off_df(self):
        all_bands_on_off_df = self.all_bands_on_off_df
        if not len(all_bands_on_off_df) or not len(
            all_bands_on_off_df[all_bands_on_off_df["state"] == "off"]
        ):
            print("No OFF states to merge")
            self.off_df = pd.DataFrame()
        else:
            print("Merge off periods within and across bands.")
            off_df = self._merge_all_bands_offs(
                self.all_bands_on_off_df, self.spatial_params
            )
            print(f"Found N={len(off_df)} off periods after merging")
            self.off_df = off_df
            # Post-merge morphological cleaning (drop short merged OFFs), then compute
            # union_area only on the survivors.
            self._filter_merged_by_duration()
            self._add_union_area()

        return self.off_df

    @classmethod
    def _merge_all_bands_offs(cls, all_bands_on_off_df, spatial_params):
        """Merge detected off states within and across bands.

        - Remove all ON periods (work only on OFFs)
        - Remove OFF periods shorter than ``min_band_off_duration``
        - Sort OFFs by descending duration
        - Greedily absorb, for each seed OFF, the candidates that overlap it in both
          depth (>= ``min_depth_overlap``) and time (>= ``min_shared_duration_overlap``).

        Each OFF period in the final df has:
                - ``intersection_start_time``/``intersection_end_time``/``intersection_duration``:
                    intersection (consensus core) of all merged OFF periods
                - ``union_start_time``/``union_end_time``/``union_duration``:
                    union (earliest/latest) across merged OFFs
                - ``lo``/``hi``/``span``: union of depth extent across merged OFFs
        """
        assert len(all_bands_on_off_df.index.unique()) == len(all_bands_on_off_df)

        # Remove ON periods and short offs
        all_bands_off_df = all_bands_on_off_df[
            all_bands_on_off_df["state"] == "off"
        ].copy()
        if spatial_params["min_band_off_duration"] is not None:
            all_bands_off_df = all_bands_off_df[
                all_bands_off_df["duration"] >= spatial_params["min_band_off_duration"]
            ]

        if not len(all_bands_off_df):
            return pd.DataFrame()

        # Sort by duration and break ties with start time
        off_df = all_bands_off_df.sort_values(
            by=["duration", "start_time"], ascending=False
        )  # Keep same indices as all_bands_off_df

        # Initialize cols for extended off duration etc
        off_df["intersection_start_time"] = off_df["start_time"]
        off_df["intersection_end_time"] = off_df["end_time"]
        off_df["intersection_duration"] = off_df["duration"]
        off_df["union_start_time"] = off_df["start_time"]
        off_df["union_end_time"] = off_df["end_time"]
        off_df["union_duration"] = off_df["duration"]
        off_df["N_merged"] = 1
        off_df["merged_band_offs_indices"] = [[idx] for idx in off_df.index]
        # Keep only meaningful columns
        off_df = off_df.loc[
            :,
            [
                "state", "intersection_start_time", "intersection_end_time", "intersection_duration",
                "union_start_time", "union_end_time", "union_duration",
                "lo", "hi", "span",
                "N_merged", "merged_band_offs_indices",
            ],
        ]

        merged_off_rows_list = []
        initial_off_df = off_df.copy()

        # Greedy merge, vectorised. The per-seed candidate pre-filter keeps OFFs whose
        # intersection start/end lie within +/- nearby_off_max_time_diff of the seed.
        # Because start <= end for every OFF, a candidate's intersection start always
        # falls in [seed_start - dt, seed_end + dt], so the candidates form a CONTIGUOUS
        # slice in start-sorted order -- located with np.searchsorted in O(log N + K)
        # instead of an O(N) full-array boolean scan per seed (which made the loop
        # quadratic in N). Within a seed, the depth/time overlaps and best-candidate pick
        # are done with numpy on the small pool (was row-wise pandas ``.apply`` -- ~85% of
        # merge time). Net: ~linear in N for fixed local OFF density, and bit-identical to
        # the prior implementation (tests/test_spatial_off.py::test_merge_matches_reference).
        nearby_off_max_time_diff = spatial_params["nearby_off_max_time_diff"]
        min_depth_overlap = spatial_params["min_depth_overlap"]
        min_shared_duration_overlap = spatial_params["min_shared_duration_overlap"]

        labels = initial_off_df.index.to_numpy()
        lo_arr = initial_off_df["lo"].to_numpy(dtype=float)
        hi_arr = initial_off_df["hi"].to_numpy(dtype=float)
        istart = initial_off_df["intersection_start_time"].to_numpy(dtype=float)
        iend = initial_off_df["intersection_end_time"].to_numpy(dtype=float)
        keep = np.ones(len(initial_off_df), dtype=bool)
        start_order = np.argsort(istart, kind="stable")
        sorted_starts = istart[start_order]

        for pos in tqdm(range(len(initial_off_df))):
            if not keep[pos]:
                continue
            keep[pos] = False  # Don't merge a seed with itself

            # Seed's fixed nearby pool: contiguous start-sorted slice within +/- dt,
            # filtered to ``iend <= hi_t`` -> exactly ``start>=lo_t & iend<=hi_t & keep``.
            lo_t = istart[pos] - nearby_off_max_time_diff
            hi_t = iend[pos] + nearby_off_max_time_diff
            left = np.searchsorted(sorted_starts, lo_t, side="left")
            right = np.searchsorted(sorted_starts, hi_t, side="right")
            pool = start_order[left:right]
            pool = np.sort(pool[(iend[pool] <= hi_t) & keep[pool]])

            merged_off_row = initial_off_df.iloc[pos].copy()
            m_lo, m_hi = lo_arr[pos], hi_arr[pos]
            m_istart, m_iend = istart[pos], iend[pos]
            pool_keep = np.ones(len(pool), dtype=bool)

            # Greedily absorb the single best candidate until none qualifies.
            while pool_keep.any():
                depth_ov = np.minimum(hi_arr[pool], m_hi) - np.maximum(lo_arr[pool], m_lo)
                time_ov = np.minimum(iend[pool], m_iend) - np.maximum(istart[pool], m_istart)
                cand = (
                    pool_keep
                    & (depth_ov >= min_depth_overlap)
                    & (time_ov >= min_shared_duration_overlap)
                )
                if not cand.any():
                    break
                # Best = max temporal overlap; ties -> lowest label (matches the reference
                # np.intersect1d + stable descending sort on shared overlap).
                ci = np.flatnonzero(cand)
                best = ci[np.lexsort((labels[pool[ci]], -time_ov[ci]))[0]]

                merged_off_row = _merge_off_rows(
                    merged_off_row, initial_off_df.loc[[labels[pool[best]]]]
                )
                m_lo, m_hi = merged_off_row["lo"], merged_off_row["hi"]
                m_istart = merged_off_row["intersection_start_time"]
                m_iend = merged_off_row["intersection_end_time"]
                pool_keep[best] = False
                keep[pool[best]] = False

            merged_off_rows_list.append(merged_off_row)

        return (
            pd.DataFrame(merged_off_rows_list)
            .sort_values(by="intersection_start_time", ascending=True)
            .reset_index(drop=True)
        )


def _merge_off_rows(base_off_row, selected_off_df):
    merged_off_row = base_off_row.copy()

    # New min/max depths
    los, his = selected_off_df["lo"].values, selected_off_df["hi"].values
    new_lo = min(*list(los), merged_off_row["lo"])
    new_hi = max(*list(his), merged_off_row["hi"])

    # New start/end time: intersection (restrictive) and union (extensive)
    intersection_starts = list(selected_off_df["intersection_start_time"].values)
    union_starts = list(selected_off_df["union_start_time"].values)
    intersection_ends = list(selected_off_df["intersection_end_time"].values)
    union_ends = list(selected_off_df["union_end_time"].values)
    new_intersection_start_time = max(*intersection_starts, merged_off_row["intersection_start_time"])
    new_union_start_time = min(*union_starts, merged_off_row["union_start_time"])
    new_intersection_end_time = min(*intersection_ends, merged_off_row["intersection_end_time"])
    new_union_end_time = max(*union_ends, merged_off_row["union_end_time"])
    assert new_intersection_start_time <= new_intersection_end_time
    assert new_intersection_start_time >= merged_off_row["intersection_start_time"]
    assert new_intersection_end_time <= merged_off_row["intersection_end_time"]
    assert new_intersection_start_time >= new_union_start_time
    assert new_intersection_end_time <= new_union_end_time

    # Update fields - times
    merged_off_row["intersection_start_time"] = new_intersection_start_time
    merged_off_row["union_start_time"] = new_union_start_time
    merged_off_row["intersection_end_time"] = new_intersection_end_time
    merged_off_row["union_end_time"] = new_union_end_time
    merged_off_row["intersection_duration"] = new_intersection_end_time - new_intersection_start_time
    merged_off_row["union_duration"] = new_union_end_time - new_union_start_time
    # Depths
    merged_off_row["lo"] = new_lo
    merged_off_row["hi"] = new_hi
    merged_off_row["span"] = new_hi - new_lo
    # Origins
    merged_off_row["N_merged"] += len(selected_off_df)
    merged_off_row["merged_band_offs_indices"] += list(selected_off_df.index)

    return merged_off_row
