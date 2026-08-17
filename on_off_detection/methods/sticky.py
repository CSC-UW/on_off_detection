"""Sticky Poisson hidden Markov model for ON/OFF detection.

Implements a 2-state Poisson HMM with a "sticky" self-transition floor, after:

    Li & La Camera (2025), "A sticky Poisson hidden Markov model for solving the
    problem of over-segmentation and rapid state switching in cortical datasets",
    PLOS One.

Standard Poisson HMMs applied to binned multiunit activity tend to over-segment
OFF periods and switch state implausibly fast. The sticky variant constrains the
self-transition probability of each state to be at least
``delta = 1 - binsize / min_dwell`` (the paper's stickiness threshold), enforcing
a minimum expected dwell time and suppressing rapid switching.

Unlike the GLM-based HMM-EM method (Chen et al. 2009, see ``hmmem.py``), this is a
plain 2-state Poisson HMM (no spike-history covariate), fit by Baum-Welch with the
self-transition floor applied at each M-step. The sequential forward-backward and
Viterbi scans are JIT-compiled with numba when available (the workspace ships it);
a pure-numpy fallback keeps the package importable without numba. The E/M-step
linear algebra is vectorized numpy.

State convention: state 0 = OFF (low rate), state 1 = ON (high rate). The returned
DataFrame uses the same ``state``/``start_time``/``end_time``/``duration`` schema as
the ``threshold`` and ``hmmem`` methods.
"""

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .. import utils
from .exceptions import FailedInitializationException, NumericalErrorException

try:  # Optional numba acceleration for the sequential HMM scans.
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba installed
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def _decorator(func):
            return func

        return _decorator


STICKY_PARAMS = {
    "binsize": 0.010,  # (s) bin size for spike-count emissions
    # (s) minimum expected state dwell -> self-transition floor
    # delta = 1 - binsize / min_dwell (Li & La Camera 2025 stickiness).
    # Set to None or <= binsize to disable the floor (plain Poisson HMM).
    "min_dwell": 0.050,
    # (Hz) near-silence constraint: cap the OFF-state mean firing rate at
    # off_rate_max during EM, so OFF means "(near-)silent population" rather than
    # merely "lower rate than ON". Matches the morphological method's deep, rare OFFs.
    #   None -> no cap (pure unconstrained 2-state Poisson HMM)
    #   0.0  -> OFF is the silent (zero-count) state; parameter-free and
    #           structure-invariant (no per-structure / per-unit rate to tune)
    #   >0   -> absolute Hz cap (does NOT generalize across structures; see the
    #           offproj cap-scheme sweep)
    "off_rate_max": None,
    "n_iter_EM": 100,  # Max Baum-Welch iterations
    "tol": 1e-4,  # Log-likelihood convergence tolerance
    "min_off_duration": None,  # (s) optional post-hoc merge of OFFs shorter than this
}


def _cap_off_rate(lam, off_cap_counts):
    """Cap the lower (OFF) state's rate at off_cap_counts, keeping states distinct."""
    if off_cap_counts is None:
        return lam
    lo = int(np.argmin(lam))
    hi = 1 - lo
    lam[lo] = min(lam[lo], off_cap_counts)
    if lam[hi] <= lam[lo]:
        lam[hi] = lam[lo] + 1e-3
    return lam

_EPS = np.spacing(1)


@njit(cache=True)
def _forward_backward(B, A, p0):
    """Scaled forward-backward for a 2-state HMM.

    Args:
        B: (2, n) emission likelihoods.
        A: (2, 2) transition matrix.
        p0: (2,) initial state distribution.

    Returns:
        (alpha, beta, c): scaled forward/backward messages (2, n) and the
        per-step scaling factors c (n,). The data log-likelihood is sum(log c).
    """
    n = B.shape[1]
    alpha = np.zeros((2, n))
    beta = np.zeros((2, n))
    c = np.zeros(n)

    a0 = p0[0] * B[0, 0]
    a1 = p0[1] * B[1, 0]
    s = a0 + a1
    if s <= 0.0:
        s = 1e-300
    c[0] = s
    alpha[0, 0] = a0 / s
    alpha[1, 0] = a1 / s
    for k in range(1, n):
        x0 = (alpha[0, k - 1] * A[0, 0] + alpha[1, k - 1] * A[1, 0]) * B[0, k]
        x1 = (alpha[0, k - 1] * A[0, 1] + alpha[1, k - 1] * A[1, 1]) * B[1, k]
        s = x0 + x1
        if s <= 0.0:
            s = 1e-300
        c[k] = s
        alpha[0, k] = x0 / s
        alpha[1, k] = x1 / s

    beta[0, n - 1] = 1.0 / c[n - 1]
    beta[1, n - 1] = 1.0 / c[n - 1]
    for k in range(n - 2, -1, -1):
        f0 = B[0, k + 1] * beta[0, k + 1]
        f1 = B[1, k + 1] * beta[1, k + 1]
        beta[0, k] = (A[0, 0] * f0 + A[0, 1] * f1) / c[k]
        beta[1, k] = (A[1, 0] * f0 + A[1, 1] * f1) / c[k]
    return alpha, beta, c


@njit(cache=True)
def _viterbi(logp0, logA, logB):
    """Viterbi decoding for a 2-state HMM in log space."""
    n = logB.shape[1]
    delta = np.zeros((2, n))
    psi = np.zeros((2, n), dtype=np.int64)
    delta[0, 0] = logp0[0] + logB[0, 0]
    delta[1, 0] = logp0[1] + logB[1, 0]
    for k in range(1, n):
        for j in range(2):
            v0 = delta[0, k - 1] + logA[0, j]
            v1 = delta[1, k - 1] + logA[1, j]
            if v0 >= v1:
                psi[j, k] = 0
                m = v0
            else:
                psi[j, k] = 1
                m = v1
            delta[j, k] = m + logB[j, k]
    states = np.zeros(n, dtype=np.int64)
    states[n - 1] = 0 if delta[0, n - 1] >= delta[1, n - 1] else 1
    for k in range(n - 2, -1, -1):
        states[k] = psi[states[k + 1], k + 1]
    return states


def _poisson_logpmf(k: np.ndarray, lam: float) -> np.ndarray:
    """log P(k | Poisson(lam)) for an integer count vector ``k``."""
    return k * np.log(lam + _EPS) - lam - gammaln(k + 1.0)


def _states_to_df(active_bin: np.ndarray, srate: float) -> pd.DataFrame:
    """Convert a binary ON(1)/OFF(0) bin vector to an on/off-period DataFrame."""
    on_starts = utils.state_starts(active_bin, 1) / srate
    off_starts = utils.state_starts(active_bin, 0) / srate
    on_ends = utils.state_ends(active_bin, 1) / srate
    off_ends = utils.state_ends(active_bin, 0) / srate
    on_durations = utils.state_durations(active_bin, 1, srate=srate)
    off_durations = utils.state_durations(active_bin, 0, srate=srate)
    return (
        pd.DataFrame(
            {
                "state": ["on"] * len(on_starts) + ["off"] * len(off_starts),
                "start_time": list(on_starts) + list(off_starts),
                "end_time": list(on_ends) + list(off_ends),
                "duration": list(on_durations) + list(off_durations),
            }
        )
        .sort_values(by="start_time")
        .reset_index(drop=True)
    )


def _run_sticky_on_counts(
    counts,
    binsize,
    *,
    min_dwell=0.050,
    off_cap_counts=None,
    n_iter_EM=100,
    tol=1e-4,
    min_off_duration=None,
    verbose=False,
):
    """Fit a sticky 2-state Poisson HMM to a pre-binned count vector.

    This is the count-based core shared by :func:`run_sticky` and by external
    callers that have already binned an observable (e.g. pooled spike counts or
    the number of active units per bin, for the fraction-of-units-active variant).

    Args:
        counts (np.ndarray): Integer observation per bin (e.g. pooled spike
            count, or active-unit count). The "OFF" state is the low-mean state.
        binsize (float): Bin size (s); only used for the ``min_dwell`` floor and
            the optional short-OFF merge.
        min_dwell (float | None): Minimum expected dwell (s) -> self-transition
            floor ``delta = 1 - binsize / min_dwell``. None/<=binsize disables it.
        off_cap_counts (float | None): Cap on the OFF-state mean emission, in the
            same units as ``counts`` (NOT Hz). ``None`` disables the cap; ``0.0``
            forces an essentially silent OFF state (OFF = zero-count bins).
        n_iter_EM (int): Max Baum-Welch iterations.
        tol (float): Log-likelihood convergence tolerance.
        min_off_duration (float | None): Post-hoc merge of OFFs shorter than this (s).
        verbose (bool): Print EM progress.

    Returns:
        (np.ndarray, dict): ``active_bin`` (1 = ON, 0 = OFF) and an info dict
        (``lambda_off``/``lambda_on``/``A``/``delta``/``log_L``/``end_iter_EM``/
        ``EM_converged``).
    """
    counts = np.asarray(counts, dtype=np.int64)
    srate = 1.0 / binsize
    nbins = len(counts)
    if nbins < 2:
        raise FailedInitializationException("Fewer than 2 bins; cannot fit HMM.")

    # --- Initialization: split bins by count into low (OFF) / high (ON) states.
    init_off = counts <= np.quantile(counts, 0.5)
    if init_off.all() or (~init_off).all():
        init_off = counts <= counts.mean()  # fall back to mean split
    if init_off.all() or (~init_off).all():
        raise FailedInitializationException(
            "Could not initialize distinct ON/OFF states from spike counts."
        )

    lam = np.array([counts[init_off].mean(), counts[~init_off].mean()], dtype=float)
    lam = np.clip(lam, 1e-6, None)
    if lam[1] <= lam[0]:
        lam[1] = lam[0] + 1e-3
    lam = _cap_off_rate(lam, off_cap_counts)

    A = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    p0 = np.array([init_off.mean(), 1.0 - init_off.mean()], dtype=float)

    # Self-transition floor from min_dwell (Li & La Camera stickiness).
    delta = 1.0 - binsize / min_dwell if (min_dwell and min_dwell > binsize) else None

    log_counts_factorial = gammaln(counts + 1.0)

    def _emissions(lam_vec):
        # (2, nbins) Poisson likelihoods.
        logB = np.empty((2, nbins))
        for st in range(2):
            logB[st] = (
                counts * np.log(lam_vec[st] + _EPS) - lam_vec[st] - log_counts_factorial
            )
        return np.exp(logB)

    prev_ll = -np.inf
    ll = -np.inf
    it = 0
    gamma = None
    for it in range(int(n_iter_EM)):
        B = _emissions(lam)
        alpha, beta, c = _forward_backward(B, A, p0)
        if not np.all(np.isfinite(c)) or np.any(c <= 0):
            raise NumericalErrorException("Non-finite forward scaling in sticky EM.")
        ll = float(np.sum(np.log(c + _EPS)))

        gamma = alpha * beta
        gamma /= gamma.sum(axis=0, keepdims=True) + _EPS

        # Expected transition counts, summed over time (fully vectorized).
        # xi_k[i, j] = alpha[i, k] * A[i, j] * (B[j, k+1] * beta[j, k+1]) / s_k.
        a = alpha[:, :-1]  # (2, K-1)
        bB = B[:, 1:] * beta[:, 1:]  # (2, K-1)
        s_k = np.einsum("ik,ik->k", a, A @ bB) + _EPS  # (K-1,)
        a_scaled = a / s_k[None, :]
        xi_sum = A * (a_scaled @ bB.T)

        # M-step: transition matrix + sticky self-transition floor.
        A = xi_sum / (xi_sum.sum(axis=1, keepdims=True) + _EPS)
        if delta is not None:
            for i in range(2):
                if A[i, i] < delta:
                    others = A[i].copy()
                    others[i] = 0.0
                    s = others.sum()
                    A[i] = others / s * (1.0 - delta) if s > 0 else A[i]
                    A[i, i] = delta

        # M-step: Poisson emission rates (+ near-silence OFF cap).
        denom = gamma.sum(axis=1)
        lam = (gamma * counts[None, :]).sum(axis=1) / (denom + _EPS)
        lam = np.clip(lam, 1e-6, None)
        lam = _cap_off_rate(lam, off_cap_counts)
        p0 = gamma[:, 0].copy()

        if verbose and (it % 20 == 0 or it == int(n_iter_EM) - 1):
            print(
                f"  sticky EM it={it} ll={ll:.1f} lam={np.round(lam, 4)} "
                f"A_diag={np.round(np.diag(A), 4)}"
            )
        if it > 0 and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

    converged = it < int(n_iter_EM) - 1

    # Keep OFF as the low-rate state (index 0); swap if EM reordered.
    if lam[0] > lam[1]:
        lam = lam[::-1].copy()
        A = A[::-1, ::-1].copy()
        p0 = p0[::-1].copy()

    # Viterbi decoding.
    logB = np.vstack([_poisson_logpmf(counts, lam[0]), _poisson_logpmf(counts, lam[1])])
    states = _viterbi(np.log(p0 + _EPS), np.log(A + _EPS), logB)
    active_bin = np.asarray(states, dtype=int)  # 1 = ON, 0 = OFF

    # Optional post-hoc merge of short OFFs.
    if min_off_duration is not None and min_off_duration > 0:
        off_durations = utils.state_durations(active_bin, 0, srate=srate)
        off_starts = utils.state_starts(active_bin, 0)
        off_ends = utils.state_ends(active_bin, 0)
        for i, dur in enumerate(off_durations):
            if dur <= min_off_duration:
                active_bin[off_starts[i] : off_ends[i] + 1] = 1

    info = {
        "lambda_off": float(lam[0]),
        "lambda_on": float(lam[1]),
        "A": A,
        "delta": delta,
        "log_L": ll,
        "end_iter_EM": int(it),
        "EM_converged": bool(converged),
    }
    return active_bin, info


def run_sticky(train, Tmax, params, verbose=True):
    """Detect ON/OFF periods with a sticky 2-state Poisson HMM.

    Args:
        train (array-like): Pooled, sorted spike times (seconds).
        Tmax (float): Recording (or cut-and-concatenated bouts) duration (seconds).
        params (dict): See :data:`STICKY_PARAMS`.
        verbose (bool): Print progress.

    Returns:
        (pd.DataFrame, dict): ``on_off_df`` with 'state'/'start_time'/'end_time'/
        'duration' columns and an ``output_info`` dict.
    """
    binsize = params["binsize"]
    srate = 1.0 / binsize

    bins = np.arange(0, Tmax + binsize, binsize)
    counts = np.histogram(train, bins=bins)[0].astype(np.int64)
    nbins = len(counts)

    cumFR = len(train) / Tmax
    if verbose:
        print(
            f"method=sticky, pop. rate = {cumFR:.1f}Hz, N={len(train)} spikes, "
            f"nbins={nbins}, numba={_HAVE_NUMBA}, params={params}"
        )

    off_rate_max = params.get("off_rate_max", None)
    # off_rate_max=0.0 is a VALID cap (OFF = silent/zero-count state), distinct
    # from None (no cap); don't fold 0.0 into the no-cap branch.
    off_cap_counts = None if off_rate_max is None else off_rate_max * binsize

    active_bin, info = _run_sticky_on_counts(
        counts,
        binsize,
        min_dwell=params.get("min_dwell", None),
        off_cap_counts=off_cap_counts,
        n_iter_EM=params["n_iter_EM"],
        tol=params["tol"],
        min_off_duration=params.get("min_off_duration", None),
        verbose=verbose,
    )

    on_off_df = _states_to_df(active_bin, srate)
    output_info = {"cumFR": cumFR, **info, "params": params}
    if verbose:
        print(
            f"sticky: done. N_off={int((on_off_df['state'] == 'off').sum())}, "
            f"lam_off={info['lambda_off']:.3f}, lam_on={info['lambda_on']:.3f}, "
            f"converged={info['EM_converged']}"
        )
    return on_off_df, output_info
