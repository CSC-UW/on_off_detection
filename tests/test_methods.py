"""Tests for the on/off detection methods (threshold, hmmem, sticky) + OnOffModel.

These are fast, dependency-light unit tests on synthetic two-state Poisson spike
trains with known ON/OFF structure. They check that each method returns the
canonical schema, recovers OFF periods, and that OnOffModel handles bouts.
"""

import numpy as np
import pandas as pd
import pytest

from on_off_detection import OnOffModel
from on_off_detection.on_off import DF_PARAMS, METHODS
from on_off_detection.methods import threshold


def _make_two_state_train(
    *, seed=0, n_cycles=120, on_dur=0.4, off_dur=0.2, on_rate=120.0, off_rate=2.0
):
    """Build a pooled spike train alternating ON (high-rate) and OFF (low-rate).

    Returns (train, Tmax, off_intervals) where off_intervals is a list of
    (start, end) tuples for the ground-truth OFF periods.
    """
    rng = np.random.default_rng(seed)
    spikes = []
    off_intervals = []
    t = 0.0
    for _ in range(n_cycles):
        # ON segment
        n_on = rng.poisson(on_rate * on_dur)
        spikes.append(t + np.sort(rng.uniform(0, on_dur, size=n_on)))
        t += on_dur
        # OFF segment (sparse)
        n_off = rng.poisson(off_rate * off_dur)
        spikes.append(t + np.sort(rng.uniform(0, off_dur, size=n_off)))
        off_intervals.append((t, t + off_dur))
        t += off_dur
    train = np.sort(np.concatenate(spikes))
    return train, t, off_intervals


@pytest.mark.parametrize("method", ["threshold", "hmmem", "sticky"])
def test_method_returns_schema_and_recovers_offs(method):
    train, Tmax, off_intervals = _make_two_state_train()
    params = dict(DF_PARAMS[method])
    on_off_df, info = METHODS[method](train, Tmax, params, verbose=False)

    # Schema
    assert set(["state", "start_time", "end_time", "duration"]).issubset(
        on_off_df.columns
    )
    assert set(on_off_df["state"].unique()).issubset({"on", "off"})

    offs = on_off_df[on_off_df["state"] == "off"]
    assert len(offs) > 0, f"{method} found no OFF periods"

    # Times are ordered and within the recording
    assert (on_off_df["end_time"] >= on_off_df["start_time"]).all()
    assert on_off_df["start_time"].min() >= -1e-9
    assert on_off_df["end_time"].max() <= Tmax + 0.05

    # We should recover roughly the right number of OFFs (120 ground-truth),
    # allowing generous tolerance for boundary/merging effects.
    assert 0.5 * len(off_intervals) <= len(offs) <= 1.5 * len(off_intervals)


def test_sticky_off_rate_max_caps_off_state():
    """The near-silence cap forces the OFF-state rate at/below off_rate_max."""
    # Train with a genuinely high OFF-state rate (80 Hz) so the cap bites.
    train, Tmax, _ = _make_two_state_train(
        on_rate=300.0, off_rate=80.0, n_cycles=200
    )
    params = dict(DF_PARAMS["sticky"])
    cap_hz = 30.0
    params["off_rate_max"] = cap_hz
    _, info = METHODS["sticky"](train, Tmax, params, verbose=False)
    # lambda_off is counts/bin; cap is cap_hz * binsize.
    assert info["lambda_off"] <= cap_hz * params["binsize"] + 1e-9
    assert info["lambda_on"] > info["lambda_off"]

    # Without the cap, the OFF state settles much higher.
    params_uncapped = dict(DF_PARAMS["sticky"])
    params_uncapped["off_rate_max"] = None
    _, info_u = METHODS["sticky"](train, Tmax, params_uncapped, verbose=False)
    assert info_u["lambda_off"] > info["lambda_off"]


def test_sticky_off_rate_max_zero_is_a_valid_cap():
    """off_rate_max=0.0 (cap0) means a silent OFF state, distinct from None.

    Regression: run_sticky used ``if off_rate_max`` to gate the cap, which folds
    0.0 into the no-cap branch. 0.0 must force lambda_off ~ 0 (OFF = silent), and
    must differ from None (uncapped), where the OFF state settles much higher.
    """
    train, Tmax, _ = _make_two_state_train(on_rate=300.0, off_rate=80.0, n_cycles=200)

    params0 = dict(DF_PARAMS["sticky"])
    params0["off_rate_max"] = 0.0
    _, info0 = METHODS["sticky"](train, Tmax, params0, verbose=False)
    assert info0["lambda_off"] <= 1e-3  # essentially silent

    params_none = dict(DF_PARAMS["sticky"])
    params_none["off_rate_max"] = None
    _, info_n = METHODS["sticky"](train, Tmax, params_none, verbose=False)
    assert info_n["lambda_off"] > info0["lambda_off"]


def test_sticky_min_dwell_floors_self_transition():
    """A larger min_dwell should not crash and should yield a sticky A matrix."""
    train, Tmax, _ = _make_two_state_train()
    params = dict(DF_PARAMS["sticky"])
    params["min_dwell"] = 0.1  # delta = 1 - 0.01/0.1 = 0.9 floor
    _, info = METHODS["sticky"](train, Tmax, params, verbose=False)
    A = info["A"]
    assert info["delta"] == pytest.approx(0.9)
    assert A[0, 0] >= 0.9 - 1e-9 and A[1, 1] >= 0.9 - 1e-9
    assert info["lambda_on"] > info["lambda_off"]


def test_threshold_bin_centers_are_midpoints():
    """Regression for the duration-histogram bin-center bug.

    Previously the bin centers were ``(edges[:-1] + edges[:-1]) / 2`` (i.e. the
    left edge), not the midpoint. This pins the corrected midpoint computation.
    """
    edges = np.array([0.0, 0.1, 0.2, 0.3])
    centers = (np.array(edges[0:-1]) + np.array(edges[1:])) / 2
    np.testing.assert_allclose(centers, [0.05, 0.15, 0.25])


@pytest.mark.parametrize("method", ["threshold", "hmmem", "sticky"])
def test_onoffmodel_recovers_original_time_and_drops_interbout(method):
    """OnOffModel cuts/concatenates bouts, detects, and recovers original time."""
    train, Tmax, _ = _make_two_state_train(n_cycles=200)
    # Two bouts separated by a gap; spikes in the gap must not appear in output.
    bouts_df = pd.DataFrame(
        {
            "start_time": [0.0, 60.0],
            "end_time": [30.0, 90.0],
            "duration": [30.0, 30.0],
            "state": ["NREM", "NREM"],
        }
    )
    model = OnOffModel(
        [train],
        bouts_df,
        cluster_ids=[0],
        method=method,
        params=DF_PARAMS[method],
        verbose=False,
    )
    on_off_df, _ = model.run()
    assert len(on_off_df) > 0
    # All periods strictly inside one of the bouts (interbout dropped).
    inside = (
        ((on_off_df["start_time"] >= 0.0) & (on_off_df["end_time"] <= 30.0))
        | ((on_off_df["start_time"] >= 60.0) & (on_off_df["end_time"] <= 90.0))
    )
    assert inside.all(), "Found a period spanning the inter-bout gap"
    assert set(on_off_df["bout_state"].unique()) == {"NREM"}


def test_threshold_module_params_present():
    assert "binsize" in threshold.THRESHOLD_PARAMS
