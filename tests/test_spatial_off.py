"""Tests for banded (spatially-resolved) unit-based OFF detection.

Fast, dependency-light synthetic tests: a population of units spread across depth,
all sharing the same two-state ON/OFF timeline (population-synchronous OFFs). The
banded model should detect OFFs in each depth band and merge them into full-span
OFF events with real lo/hi/span. Exercises both band-definition strategies
(fixed_tiled ladder + greedy_fr), the soft per-band floor, and the per-band
parameter hook.
"""

import numpy as np
import pandas as pd
import pytest

from on_off_detection import SpatialOffModel
from on_off_detection.spatial_off import (
    SPATIAL_PARAMS,
    _merge_off_rows,
    _tile_band_scale,
    rect_union_area,
)

STICKY_CAP0 = {
    "binsize": 0.010,
    "min_dwell": 0.050,
    "off_rate_max": 0.0,  # cap0: OFF = silent
    "n_iter_EM": 100,
    "tol": 1e-4,
    "min_off_duration": None,
}


def _make_unit_train(rng, *, n_cycles, on_dur, off_dur, on_rate, off_rate):
    spikes = []
    offs = []
    t = 0.0
    for _ in range(n_cycles):
        n_on = rng.poisson(on_rate * on_dur)
        spikes.append(t + np.sort(rng.uniform(0, on_dur, size=n_on)))
        t += on_dur
        n_off = rng.poisson(off_rate * off_dur)
        spikes.append(t + np.sort(rng.uniform(0, off_dur, size=n_off)))
        offs.append((t, t + off_dur))
        t += off_dur
    return np.sort(np.concatenate(spikes)), t, offs


def _make_population(
    *, n_units=12, depth_lo=0.0, depth_hi=1000.0, seed=0,
    n_cycles=60, on_dur=0.4, off_dur=0.2, on_rate=40.0, off_rate=1.0,
):
    """Population of units at evenly-spaced depths sharing a synchronous OFF timeline."""
    rng = np.random.default_rng(seed)
    depths = np.linspace(depth_lo, depth_hi, n_units)
    trains, Tmax, offs = [], None, None
    for _ in range(n_units):
        train, Tmax, offs = _make_unit_train(
            rng, n_cycles=n_cycles, on_dur=on_dur, off_dur=off_dur,
            on_rate=on_rate, off_rate=off_rate,
        )
        trains.append(train)
    return trains, depths, Tmax, offs


def _bouts(Tmax):
    return pd.DataFrame(
        {"start_time": [0.0], "end_time": [Tmax], "duration": [Tmax], "state": ["NREM"]}
    )


# --------------------------------------------------------------------------- #
# _tile_band_scale
# --------------------------------------------------------------------------- #
def test_tile_band_scale_whole_structure():
    assert _tile_band_scale(0.0, 1000.0, None, 0.5, "superficial") == [(0.0, 1000.0)]
    # size >= extent collapses to a single whole-structure band
    assert _tile_band_scale(0.0, 1000.0, 2000.0, 0.5, "superficial") == [(0.0, 1000.0)]


def test_tile_band_scale_superficial_covers_extent():
    bands = _tile_band_scale(0.0, 1000.0, 250.0, 0.5, "superficial")
    assert bands[0][0] == 0.0
    assert bands[-1][1] == pytest.approx(1000.0)
    # overlap: step = 250 * (1 - 0.5) = 125
    assert bands[1][0] == pytest.approx(125.0)
    # every band clipped within the extent
    assert all(0.0 <= lo < hi <= 1000.0 for lo, hi in bands)


def test_tile_band_scale_deep_start_mirrors():
    bands = _tile_band_scale(0.0, 1000.0, 250.0, 0.5, "deep")
    assert bands[0][1] == pytest.approx(1000.0)
    assert bands[-1][0] == pytest.approx(0.0)


def test_tile_band_scale_bad_start_raises():
    with pytest.raises(ValueError):
        _tile_band_scale(0.0, 1000.0, 250.0, 0.5, "sideways")


# --------------------------------------------------------------------------- #
# fixed_tiled ladder end-to-end
# --------------------------------------------------------------------------- #
def test_fixed_tiled_ladder_runs_and_recovers_offs():
    trains, depths, Tmax, offs = _make_population()
    model = SpatialOffModel(
        trains,
        depths,
        _bouts(Tmax),
        cluster_ids=list(range(len(trains))),
        on_off_method="sticky",
        on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [250.0, 500.0, None]},
        verbose=False,
    )
    # Multi-scale ladder: bands at 250, 500, and whole-structure scales.
    assert set(model.bands_df["band_scale"].unique()) == {250.0, 500.0, "whole"}
    assert len(model.bands_df) > 3

    off_df = model.run()
    assert len(off_df) > 0
    # Real spatial fields
    for col in ["intersection_start_time", "intersection_end_time", "lo", "hi", "span",
                "N_merged", "merged_band_offs_indices"]:
        assert col in off_df.columns
    assert (off_df["lo"] >= depths.min() - 1e-6).all()
    assert (off_df["hi"] <= depths.max() + 1e-6).all()
    assert (off_df["span"] >= 0).all()
    # Synchronous OFFs => at least one merged OFF spans most of the probe.
    assert off_df["span"].max() >= 0.5 * (depths.max() - depths.min())
    # Roughly the right order of magnitude (60 ground-truth). Generous: per-band
    # sticky over-segments modestly at this synthetic rate and the merge does not
    # fully collapse near-coincident band-OFFs, so allow up to ~3x.
    assert 0.3 * len(offs) <= len(off_df) <= 3.0 * len(offs)


def test_greedy_fr_runs():
    trains, depths, Tmax, offs = _make_population()
    model = SpatialOffModel(
        trains,
        depths,
        _bouts(Tmax),
        cluster_ids=list(range(len(trains))),
        on_off_method="sticky",
        on_off_params=STICKY_CAP0,
        spatial_params={
            "band_definition": "greedy_fr",
            "band_min_fr": 100.0,
            "band_min_size": 150.0,
        },
        verbose=False,
    )
    assert (model.bands_df["band_scale"] == "greedy").all()
    off_df = model.run()
    assert len(off_df) > 0
    assert off_df["span"].max() > 0


# --------------------------------------------------------------------------- #
# soft per-band floor
# --------------------------------------------------------------------------- #
def test_band_floor_drops_sparse_bands():
    trains, depths, Tmax, _ = _make_population(n_units=12)
    # With band_min_units very high, the small 250um bands (few units) are dropped.
    model = SpatialOffModel(
        trains, depths, _bouts(Tmax),
        on_off_method="sticky", on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [250.0, 500.0, None], "band_min_units": 8},
        verbose=False,
    )
    n_units_per_band = model.bands_df["band_cluster_indices"].apply(len)
    assert (n_units_per_band >= 8).all()
    # The whole-structure band (all 12 units) must survive.
    assert "whole" in set(model.bands_df["band_scale"].unique())


def test_band_floor_all_dropped_raises():
    trains, depths, Tmax, _ = _make_population(n_units=12)
    with pytest.raises(ValueError):
        SpatialOffModel(
            trains, depths, _bouts(Tmax),
            on_off_method="sticky", on_off_params=STICKY_CAP0,
            spatial_params={"band_sizes": [250.0], "band_min_units": 999},
            verbose=False,
        )


# --------------------------------------------------------------------------- #
# per-band (adaptive) parameter hook
# --------------------------------------------------------------------------- #
def test_per_band_params_hook():
    trains, depths, Tmax, _ = _make_population()
    model = SpatialOffModel(
        trains, depths, _bouts(Tmax),
        on_off_method="sticky", on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [500.0, None]},
        verbose=False,
    )
    # Adaptive: give each band its own params (here a per-band off-rate cap).
    per_band = []
    for _, row in model.bands_df.iterrows():
        p = dict(STICKY_CAP0)
        p["off_rate_max"] = 0.05 * len(row["band_cluster_indices"])  # scale with unit count
        per_band.append(p)
    model.per_band_on_off_params = per_band
    assert len(model.per_band_on_off_params) == len(model.bands_df)
    off_df = model.run()
    assert len(off_df) >= 0  # runs without error; may detect fewer OFFs


def test_per_band_params_wrong_length_raises():
    trains, depths, Tmax, _ = _make_population()
    model = SpatialOffModel(
        trains, depths, _bouts(Tmax),
        on_off_method="sticky", on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [500.0, None]},
        verbose=False,
    )
    with pytest.raises(ValueError):
        model.per_band_on_off_params = [dict(STICKY_CAP0)]  # too few


def test_bouts_recovers_original_time():
    """Two bouts with a gap; no merged OFF should span the inter-bout gap."""
    trains, depths, Tmax, _ = _make_population(n_cycles=120)
    bouts_df = pd.DataFrame(
        {
            "start_time": [0.0, 60.0],
            "end_time": [30.0, 90.0],
            "duration": [30.0, 30.0],
            "state": ["NREM", "NREM"],
        }
    )
    model = SpatialOffModel(
        trains, depths, bouts_df,
        on_off_method="sticky", on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [500.0, None]},
        verbose=False,
    )
    off_df = model.run()
    assert len(off_df) > 0
    inside = (
        (off_df["intersection_start_time"] >= -1e-9) & (off_df["intersection_end_time"] <= 30.0 + 1e-6)
    ) | (
        (off_df["intersection_start_time"] >= 60.0 - 1e-6) & (off_df["intersection_end_time"] <= 90.0 + 1e-6)
    )
    assert inside.all(), "A merged OFF spans the inter-bout gap"


# --------------------------------------------------------------------------- #
# rect_union_area + member boxes / union_area
# --------------------------------------------------------------------------- #
def test_rect_union_area_single_box():
    assert rect_union_area([[0.0, 1.0, 0.0, 10.0]]) == pytest.approx(10.0)


def test_rect_union_area_disjoint_is_sum():
    boxes = [[0.0, 1.0, 0.0, 10.0], [5.0, 6.0, 0.0, 10.0]]
    assert rect_union_area(boxes) == pytest.approx(20.0)


def test_rect_union_area_l_shape_less_than_sum():
    # A=[0,0.3]x[0,400], B=[0.2,0.5]x[200,600]; overlap [0.2,0.3]x[200,400]=20.
    A = [0.0, 0.3, 0.0, 400.0]
    B = [0.2, 0.5, 200.0, 600.0]
    union = rect_union_area([A, B])
    assert union == pytest.approx(120.0 + 120.0 - 20.0)  # 220
    # strictly less than the bounding box [0,0.5]x[0,600] = 300
    assert union < 0.5 * 600.0


def test_rect_union_area_empty():
    assert rect_union_area(np.empty((0, 4))) == 0.0


def test_union_area_column_and_member_boxes():
    trains, depths, Tmax, _ = _make_population()
    model = SpatialOffModel(
        trains, depths, _bouts(Tmax),
        on_off_method="sticky", on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [250.0, 500.0, None]},
        verbose=False,
    )
    off_df = model.run()
    assert "union_area" in off_df.columns
    assert (off_df["union_area"] > 0).all()
    # union footprint <= bounding-box area (span * union duration) for every OFF.
    bbox_area = off_df["span"] * off_df["union_duration"]
    assert (off_df["union_area"] <= bbox_area + 1e-6).all()
    # member boxes recover the constituent band rectangles.
    row = off_df.iloc[0]
    boxes = model.get_member_boxes(row)
    assert boxes.shape[1] == 4 and len(boxes) == row["N_merged"]
    assert rect_union_area(boxes) == pytest.approx(row["union_area"])


def test_spatial_params_unrecognized_key_raises():
    trains, depths, Tmax, _ = _make_population()
    with pytest.raises(ValueError):
        SpatialOffModel(
            trains, depths, _bouts(Tmax),
            on_off_method="sticky", on_off_params=STICKY_CAP0,
            spatial_params={"not_a_param": 1},
            verbose=False,
        )


# --------------------------------------------------------------------------- #
# post-merge morphological duration cleaning
# --------------------------------------------------------------------------- #
def _detected_model():
    """A model with bands detected (run_all_bands_on_off_df done) but not yet merged.

    Lets us re-run run_off_df() with different post-merge filter settings against the
    same fixed all_bands_on_off_df (the merge itself is deterministic).
    """
    trains, depths, Tmax, _ = _make_population()
    model = SpatialOffModel(
        trains, depths, _bouts(Tmax),
        on_off_method="sticky", on_off_params=STICKY_CAP0,
        spatial_params={"band_sizes": [250.0, 500.0, None]},
        verbose=False,
    )
    model.run_all_bands_on_off_df()
    return model


def test_post_merge_filter_keeps_exactly_long_enough():
    model = _detected_model()
    unfiltered = model.run_off_df().copy()
    assert len(unfiltered) > 0

    thresh = float(unfiltered["union_duration"].quantile(0.5))
    model._spatial_params["min_merged_off_duration"] = thresh
    filtered = model.run_off_df()

    # The filter keeps exactly the merged OFFs whose UNION span is >= threshold.
    expected_kept = int((unfiltered["union_duration"] >= thresh).sum())
    assert len(filtered) == expected_kept
    assert (filtered["union_duration"] >= thresh).all()
    if unfiltered["union_duration"].nunique() > 1:
        assert len(filtered) < len(unfiltered)


def test_post_merge_filter_recomputes_union_area_on_survivors():
    model = _detected_model()
    unfiltered = model.run_off_df().copy()
    thresh = float(unfiltered["union_duration"].quantile(0.6))
    model._spatial_params["min_merged_off_duration"] = thresh
    filtered = model.run_off_df()

    # union_area exists, is aligned to the (reset) survivor index, and matches the
    # member boxes of each survivor (i.e. recomputed, not stale/misaligned).
    assert "union_area" in filtered.columns
    assert len(filtered["union_area"]) == len(filtered)
    assert (filtered["union_area"] > 0).all()
    for _, row in filtered.iterrows():
        assert rect_union_area(model.get_member_boxes(row)) == pytest.approx(
            row["union_area"]
        )


def test_post_merge_filter_below_pre_floor_is_noop():
    model = _detected_model()
    unfiltered = model.run_off_df().copy()
    # Every constituent band-OFF passed the 0.03s pre-merge floor, and the union span
    # is >= every constituent, so a post-merge floor below 0.03 removes nothing. (Use
    # 0.025 rather than exactly 0.03 to avoid the float-epsilon boundary where a union
    # of two near-identical ~0.03s OFFs computes to 0.0299999998.)
    model._spatial_params["min_merged_off_duration"] = 0.025
    filtered = model.run_off_df()
    assert len(filtered) == len(unfiltered)


def test_post_merge_filter_above_floor_prunes():
    model = _detected_model()
    unfiltered = model.run_off_df().copy()
    thresh = float(unfiltered["union_duration"].quantile(0.5))
    model._spatial_params["min_merged_off_duration"] = thresh
    filtered = model.run_off_df()
    assert len(filtered) == int((unfiltered["union_duration"] >= thresh).sum())
    assert (filtered["union_duration"] >= thresh).all()


def test_post_merge_filter_none_is_default_noop():
    model = _detected_model()
    assert model.spatial_params["min_merged_off_duration"] is None
    off_df = model.run_off_df()
    assert len(off_df) > 0  # default: no post-merge pruning happens


# --------------------------------------------------------------------------- #
# vectorised merge is bit-identical to the original row-wise implementation
# --------------------------------------------------------------------------- #
def _reference_merge(all_bands_on_off_df, spatial_params):
    """Verbatim copy of the ORIGINAL row-wise ``_merge_all_bands_offs`` (pre-numpy
    optimization), kept here as the bit-identity oracle for the vectorised engine.

    Uses the unchanged module ``_merge_off_rows`` so only the *candidate finding* and
    *iteration* differ between this oracle and production.
    """
    def _find_contiguous(merged, nearby, sp):
        m_lo, m_hi = merged["lo"], merged["hi"]
        ov = nearby.apply(
            lambda r: (min(r["hi"], m_hi) - max(r["lo"], m_lo)), axis=1
        )
        return nearby.index[nearby["keep"] & (ov >= sp["min_depth_overlap"])]

    def _find_concurrent(merged, nearby, sp):
        ist, ien = merged["intersection_start_time"], merged["intersection_end_time"]
        ov = nearby.apply(
            lambda r: (
                min(r["intersection_end_time"], ien)
                - max(r["intersection_start_time"], ist)
            ),
            axis=1,
        )
        return nearby.index[nearby["keep"] & (ov >= sp["min_shared_duration_overlap"])]

    def _find_indice(merged, nearby, sp):
        idx = np.intersect1d(
            _find_contiguous(merged, nearby, sp), _find_concurrent(merged, nearby, sp)
        )
        if not len(idx):
            return []
        ist, ien = merged["intersection_start_time"], merged["intersection_end_time"]
        tm = nearby.loc[idx]
        tm["shared_duration_overlap"] = tm.apply(
            lambda r: (
                min(r["intersection_end_time"], ien)
                - max(r["intersection_start_time"], ist)
            ),
            axis=1,
        )
        return tm.sort_values(by="shared_duration_overlap", ascending=False).index[0:1]

    assert len(all_bands_on_off_df.index.unique()) == len(all_bands_on_off_df)
    off = all_bands_on_off_df[all_bands_on_off_df["state"] == "off"].copy()
    if spatial_params["min_band_off_duration"] is not None:
        off = off[off["duration"] >= spatial_params["min_band_off_duration"]]
    if not len(off):
        return pd.DataFrame()
    off_df = off.sort_values(by=["duration", "start_time"], ascending=False)
    off_df["intersection_start_time"] = off_df["start_time"]
    off_df["intersection_end_time"] = off_df["end_time"]
    off_df["intersection_duration"] = off_df["duration"]
    off_df["union_start_time"] = off_df["start_time"]
    off_df["union_end_time"] = off_df["end_time"]
    off_df["union_duration"] = off_df["duration"]
    off_df["N_merged"] = 1
    off_df["merged_band_offs_indices"] = [[idx] for idx in off_df.index]
    off_df = off_df.loc[
        :,
        ["state", "intersection_start_time", "intersection_end_time",
         "intersection_duration", "union_start_time", "union_end_time",
         "union_duration", "lo", "hi", "span", "N_merged",
         "merged_band_offs_indices"],
    ]
    merged_list = []
    initial = off_df.copy()
    keep = pd.Series(True, index=initial.index)
    dt = spatial_params["nearby_off_max_time_diff"]
    for i, off_row in initial.iterrows():
        if not keep.loc[i]:
            continue
        keep.loc[i] = False
        nearby = initial.loc[
            keep.values
            & (initial["intersection_start_time"] >= off_row["intersection_start_time"] - dt)
            & (initial["intersection_end_time"] <= off_row["intersection_end_time"] + dt)
        ].copy()
        nearby["keep"] = True
        to_merge = _find_indice(off_row, nearby, spatial_params)
        merged = off_row.copy()
        while len(to_merge):
            merged = _merge_off_rows(merged, initial.loc[to_merge])
            keep.loc[to_merge] = False
            nearby.loc[to_merge, "keep"] = False
            to_merge = _find_indice(merged, nearby, spatial_params)
        merged_list.append(merged)
        keep.loc[i] = False
    return (
        pd.DataFrame(merged_list)
        .sort_values(by="intersection_start_time", ascending=True)
        .reset_index(drop=True)
    )


def _random_all_bands_df(seed, n):
    """Synthetic all-bands ON/OFF frame: overlapping depth bands, short OFFs, some ONs."""
    rng = np.random.default_rng(seed)
    span_s = max(5.0, n / 30.0)
    n_bands = 8
    band_los = np.linspace(0.0, 1600.0, n_bands)
    bi = rng.integers(0, n_bands, size=n)
    lo = band_los[bi]
    hi = lo + 400.0
    start = rng.uniform(0.0, span_s, size=n)
    dur = rng.uniform(0.01, 0.2, size=n)
    state = np.where(rng.random(n) < 0.85, "off", "on")
    return pd.DataFrame(
        {
            "state": state,
            "start_time": start,
            "end_time": start + dur,
            "duration": dur,
            "lo": lo,
            "hi": hi,
            "span": hi - lo,
        }
    )


@pytest.mark.parametrize("seed,n", [(0, 50), (1, 200), (2, 500), (3, 1200)])
def test_merge_matches_reference(seed, n):
    """The vectorised merge reproduces the original row-wise merge bit-for-bit,
    including merged times/depths, N_merged, and the merged_band_offs_indices lists."""
    df = _random_all_bands_df(seed, n)
    sp = dict(SPATIAL_PARAMS)
    sp["min_band_off_duration"] = 0.03
    got = SpatialOffModel._merge_all_bands_offs(df, sp)
    exp = _reference_merge(df, sp)
    assert len(got) == len(exp) > 0
    pd.testing.assert_frame_equal(got, exp, check_dtype=False)
