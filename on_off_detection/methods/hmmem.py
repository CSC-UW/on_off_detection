"""
Implements "HMM-EM" method from:

Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown;
Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.
doi: https://doi.org/10.1162/neco.2009.06-08-799

Translated to python 05/22 by Tom Bugnon from MATLAB code provided by Zhe Sage Chen

Differences from original MATLAB code: 
- Use all bins with shorter window for history at beginning
- Use numpy RNG in newton_ralphson (So different output as MATLAB) ( TODO: https://stackoverflow.com/a/36823993 )
- Max number of iterations is params['n_iter_EM'] rather than n_iter_EM - 1
- normalize "bin_history_spike_count"

Heuristics (tested in rat prelimbic cortex during sleep recovery)
- HMMEM model stays stuck in very low OFF fr region with init_state_off_on_fr_ratio_thresh <= 0.02. Can be used to enforce no spikes during OFFs, but effectively defeats the purpose of HMMEM optimization
- Utterly fails with sumFR <= 60Hz
- With FR = 80Hz, not so good and very  sensitive on initialization:
    - long tail in OFFs duration with init_state_off_on_fr_ratio_thresh = 0.1, looks dubious on visual inspection
    - No/few offs (fails) with init_state_off_on_fr_ratio_thresh = 0.05, which works well for higher firing rate
- With FR = 100Hz, looks pretty good
    - init_state_off_on_fr_ratio_thresh = 0.03 looks overly conservative, some OFFs split around single spikes that should probably not be.
    - init_state_off_on_fr_ratio_thresh = 0.05 looks sane, very low FR OFFs but not null. Can be considered conservative
    - init_state_off_on_fr_ratio_thresh = 0.1 looks ok. Some OFFs with a number of spikes that could be considered false positives. 15% more offs than with 0.05
    - history_window_n_bins = 2 fails
    - history_window_n_bins = 10 looks fine
    - history_window_n_bins >= 20 seems to find more false negatives

For cortical data I recommend 100Hz minimum population rate, init_state_off_on_fr_ratio_thresh=0.05 and history_window_n_bins = 10
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln

from .. import utils
from .exceptions import NumericalErrorException, FailedInitializationException
from .sticky import _forward_backward, _viterbi


HMMEM_PARAMS = {
    "binsize": 0.010,  # (s) (Discrete algorithm)
    "history_window_nbins": 10,  # Size of history window IN BINS
    "n_iter_EM": 200,  # Number of iterations for EM
    "n_iter_newton_ralphson": 100,
    "init_A": np.array(
        [[0.1, 0.9], [0.01, 0.99]]
    ),  # Initial transition probability matrix
    "init_state_off_on_fr_ratio_thresh": 0.05, # During initialization, after removing OFFs shorter than 0.05msec, we remove OFFs with FR less than init_state_off_on_fr_ratio_thresh times the grand mean ON state firing rate
    "init_mu": None,  # ~ OFF rate. Fitted to data if None
    "init_alphaa": None,  # ~ difference between ON and OFF rate. Fitted to data if None
    "init_betaa": None,  # ~ Weight of recent history firing rate. Fitted to data if None,
    "min_off_duration": None,  # Merge active states separated by less than this
}


# TODO: Nx1-array for betaa (one value per "history window")
def run_hmmem(
    train,
    Tmax,
    params,
    output_dir=None,
    filename=None,  # TODO harmonize
    save=None,  # TODO harmonize
    verbose=True,
):
    # Params
    assert set(params.keys()) == set(HMMEM_PARAMS.keys())

    # Merge and bin all trains
    bins = np.arange(0, Tmax + params["binsize"], params["binsize"])
    nbins = len(bins)
    bin_spike_count, _ = np.histogram(train, bins)
    bin_spike_count = bin_spike_count.astype(int)

    # History spike count of each bin (through panda roll)
    # Substract bin_spike_count to exclude current bin
    # (while keeping actual window of size X), so X+1 -1
    # (Stay consistent with matlab)
    # TODO: Modify for N-dim betaa
    bin_history_spike_count = (
        pd.Series(bin_spike_count)
        .rolling(
            params["history_window_nbins"] + 1,  # Sum over window of N bins
            center=False,  # Window left of each sample
            min_periods=1,  # Sum over fewer bins at beginning of array (Unused if we trim)
        )
        .sum()
        .to_numpy(dtype=int)
        - bin_spike_count
    )
    # Normalize (avoid overflows with large history window)
    bin_history_spike_count = bin_history_spike_count / params["history_window_nbins"]

    # Reshape to 1xnbins vectors (MATLAB consistency of _run_hmmem)
    bin_spike_count = bin_spike_count.reshape((1, -1))
    bin_history_spike_count = bin_history_spike_count.reshape((1, -1))

    # Ignore bins without full history
    bin_spike_count_trimmed = bin_spike_count[:, params["history_window_nbins"] :]
    bin_history_spike_count_trimmed = bin_history_spike_count[
        :, params["history_window_nbins"] :
    ]

    # Fit init_alphaa, init_mu and init_betaa ?
    if all([params[p] is None for p in ["init_alphaa", "init_mu", "init_betaa"]]):
        init_mu, init_alphaa, init_betaa = fit_init_poisson_params(
            bin_spike_count_trimmed[0, :],
            bin_history_spike_count_trimmed[0, :],
            params["binsize"],
            init_state_off_on_fr_ratio_thresh=params["init_state_off_on_fr_ratio_thresh"],
            verbose=verbose,
        )
    else:
        if any([params[k] is None for k in ["init_alphaa", "init_mu", "init_betaa"]]):
            raise ValueError(
                "'init_alphaa', 'init_mu' and 'init_betaa' params should either be all floats or all None."
            )
        init_alphaa = params["init_alphaa"]
        init_mu = params["init_mu"]
        init_betaa = params["init_betaa"]

    (
        S,
        prob_S,
        alphaa,
        betaa,
        mu,
        A,
        B,
        p0,
        log_L,
        log_P,
        end_iter_EM,
        EM_converged,
    ) = _run_hmmem(
        bin_spike_count_trimmed,
        bin_history_spike_count_trimmed,
        np.array(params["init_A"]),
        init_alphaa,
        init_betaa,
        init_mu,
        int(params["n_iter_EM"]),
        int(params["n_iter_newton_ralphson"]),
        verbose=verbose,
    )

    # Return identical result from MATLAB original code
    # return S, prob_S, alphaa, betaa, mu, A, B, p0, log_L, log_P

    # 1d-Array of active/inactive bins
    # Broadcast trimmed data to original number of bins
    active_bin = S[0, 0] * np.ones(
        (nbins,)
    )  # Set same value for ignored bins at the beginning as first detected value
    active_bin[params["history_window_nbins"] + 1 :] = S[0, :]  # 1d
    assert all([s in [0, 1] for s in active_bin])
    srate = 1 / params["binsize"]  # "Sampling rate" of returned binned states (Hz)

    ## Remove short OFF states and return as pd.Dataframe

    # Merge active states separated by less than min_off_duration
    if params["min_off_duration"] is not None and params["min_off_duration"] > 0:
        if verbose:
            print("Merge closeby on-periods...", end="")
        off_durations = utils.state_durations(active_bin, 0, srate=srate)
        off_starts = utils.state_starts(active_bin, 0)
        off_ends = utils.state_ends(active_bin, 0)
        N_merged = 0
        for i, off_dur in enumerate(off_durations):
            if off_dur <= params["min_off_duration"]:
                active_bin[off_starts[i] : off_ends[i] + 1] = 1
                N_merged += 1
        if verbose:
            print(f"Merged N={N_merged} active periods")

    # Return df
    # all in (sec)
    if verbose:
        print("Get final on/off periods df...")
    on_starts = utils.state_starts(active_bin, 1) / srate
    off_starts = utils.state_starts(active_bin, 0) / srate
    on_ends = utils.state_ends(active_bin, 1) / srate
    off_ends = utils.state_ends(active_bin, 0) / srate
    on_durations = utils.state_durations(
        active_bin,
        1,
        srate=srate,
    )
    off_durations = utils.state_durations(
        active_bin,
        0,
        srate=srate,
    )
    N_on = len(on_starts)
    N_off = len(off_starts)

    # TODO: Return _run_hmmem info bin by bin?
    on_off_df = pd.DataFrame(
        {
            "state": ["on" for _ in range(N_on)] + ["off" for _ in range(N_off)],
            "start_time": list(on_starts) + list(off_starts),
            "end_time": list(on_ends) + list(off_ends),
            "duration": list(on_durations) + list(off_durations),
        }
    ).sort_values(by="start_time").reset_index(drop=True)

    output_info = {
        "cumFR": len(train) / Tmax,
        "alphaa": alphaa,
        "betaa": betaa,
        "mu": mu,
        "init_alphaa": init_alphaa,
        "init_betaa": init_betaa,
        "init_mu": init_mu,
        "A": A,
        "log_L": log_L,
        "end_iter_EM": end_iter_EM,
        "EM_converged": EM_converged,
        "params": params,
    }

    return on_off_df, output_info


# TODO: Nx1-array for betaa (one value per "history window")
def _run_hmmem(
    bin_spike_count,
    bin_history_spike_count,
    init_A,
    init_alphaa,
    init_betaa,
    init_mu,
    n_iter_EM,
    n_iter_newton_ralphson,
    verbose=True,
):

    # Param check
    bin_spike_count = np.atleast_1d(bin_spike_count).astype(int)
    bin_history_spike_count = np.atleast_1d(bin_history_spike_count).astype(int)
    assert bin_spike_count.shape[0] == 1  # Horizontal vectors
    assert bin_spike_count.shape[1] == bin_history_spike_count.shape[1]
    # TODO: bin_history_spike_count could be dimension (k, nbins) rather than (1, nbins)
    #   if so beta would be dimension k
    if not isinstance(init_betaa, (float, int)):
        raise NotImplementedError()
    if bin_history_spike_count.shape[0] > 1:
        raise NotImplementedError()

    # Constants
    EPS = np.spacing(1)
    STATES = np.array([0.0, 1.0])  # OFF, ON

    A = init_A.copy().astype(float)
    alphaa = float(init_alphaa)
    betaa = float(init_betaa)
    mu = float(init_mu)

    nbins = bin_spike_count.shape[1]
    counts = bin_spike_count[0]  # (nbins,)
    hist = bin_history_spike_count[0].astype(float)  # (nbins,)
    log_factorial = gammaln(counts.astype(float) + 1.0)

    def _emissions(mu, alphaa, betaa):
        # Poisson emission matrix B (2, nbins). Chen et al. 2009 eqs 2.2-2.3:
        # log lambda = mu + alphaa*state + betaa*history; B = Poisson(count|lambda).
        lam = np.exp(mu + alphaa * STATES[:, None] + betaa * hist[None, :])
        logB = counts[None, :] * np.log(lam + EPS) - lam - log_factorial[None, :]
        return np.exp(logB)

    B = _emissions(mu, alphaa, betaa)
    p0 = 0.5 * np.ones(2)

    # Forward-backward E-M. The sequential forward/backward and Viterbi scans are
    # the numba-JIT helpers shared with the sticky method; emissions, posteriors
    # (gamma), and expected transition counts (zeta_sum) are vectorized numpy.
    log_P = np.empty((n_iter_EM,), dtype=float)
    gamma = np.zeros((2, nbins))
    t = 0
    diff_log_P = 10.0
    while t < n_iter_EM and diff_log_P > 0:
        alpha, beta, C = _forward_backward(B, A, p0)
        if not np.all(np.isfinite(C)) or np.any(C <= 0):
            raise NumericalErrorException(
                f"Numerical error in forward-backward scaling (EM step t={t})"
            )
        log_P[t] = np.sum(np.log(C + EPS))

        gamma = alpha * beta
        gamma /= gamma.sum(axis=0, keepdims=True) + EPS

        # Expected transition counts summed over time (vectorized zeta):
        # zeta_sum[i,j] = A[i,j] * sum_k alpha[i,k] (B[j,k+1] beta[j,k+1]) / s_k.
        a = alpha[:, :-1]
        bB = B[:, 1:] * beta[:, 1:]
        s_k = np.einsum("ik,ik->k", a, A @ bB) + EPS
        zeta_sum = A * ((a / s_k[None, :]) @ bB.T)

        # M-step: transition matrix + GLM params (Newton-Raphson).
        p0 = gamma[:, 0].copy()
        A = zeta_sum / (zeta_sum.sum(axis=1, keepdims=True) + EPS)
        Z = gamma[1, :]  # E[S], since STATES = [0, 1]
        alphaa, betaa, mu = newton_ralphson(
            bin_spike_count,
            Z,
            bin_history_spike_count,
            alphaa,
            betaa,
            mu,
            n_iter_newton_ralphson,
        )
        B = _emissions(mu, alphaa, betaa)

        if verbose:
            print(
                f"n_iter_EM={t}, log-likelihood={log_P[t]}, mu={mu}, "
                f"alpha={alphaa}, beta={betaa}, A={A}"
            )
        if t > 1:
            diff_log_P = log_P[t] - log_P[t - 1]
        t += 1

    end_iter_EM = t - 1
    EM_converged = t < n_iter_EM

    prob_S = gamma  # 2 x nbins
    p0 = gamma[:, 0:1]  # 2 x 1

    # Viterbi decoding (numba, log-domain).
    states = _viterbi(np.log(gamma[:, 0] + EPS), np.log(A + EPS), np.log(B + EPS))
    S = states.reshape(1, -1).astype(int)
    log_L = float(log_P[end_iter_EM])

    return (
        S,
        prob_S,
        alphaa,
        betaa,
        mu,
        A,
        B,
        p0,
        log_L,
        log_P,
        end_iter_EM,
        EM_converged,
    )


def get_initial_state_estimate(
    bin_spike_count,
    binsize,
    method="liberal",
    off_min_duration=0.05,
    off_on_fr_ratio_threshold=0.05,
):
    """Compute initial estimate of ON/OFF states.

    Used to fit GLM parameters and initialize HMMEM algorithm.
    Ideally we'd use a ground truth.
    """
    if method is None:
        method = "liberal"
    assert method in ["liberal", "intermediate", "conservative"]

    if binsize > 0.010:
        raise NotImplementedError()

    def _flip_short_periods(active_bin, state, min_duration, binsize):
        assert state in [0, 1]
        srate = 1 / binsize
        state_durations = utils.state_durations(active_bin, state, srate=srate)  # (sec)
        state_starts = utils.state_starts(active_bin, state)
        state_ends = utils.state_ends(active_bin, state)
        for i, dur in enumerate(state_durations):
            if dur <= min_duration:
                active_bin[state_starts[i] : state_ends[i] + 1] = not state
        return active_bin

    def _flip_low_fr_offs(bin_spike_count, active_bin, thresh_fr, binsize):
        srate = 1 / binsize
        state_durations = utils.state_durations(active_bin, 0, srate=srate)  # (sec)
        state_starts = utils.state_starts(active_bin, 0)
        state_ends = utils.state_ends(active_bin, 0)
        for i, dur in enumerate(state_durations):
            n_spikes = np.sum(bin_spike_count[state_starts[i] : state_ends[i]])
            if n_spikes > dur * thresh_fr:
                active_bin[state_starts[i] : state_ends[i]] = 1
        return active_bin
    
    # if method == "liberal":
    #     active_bin = (bin_spike_count > min(bin_spike_count) + 1).astype(bool)
    # elif method == "conservative":
    #     active_bin = (bin_spike_count > min(bin_spike_count)).astype(bool)
    # elif method == "intermediate":
    #     active_bin = (bin_spike_count > min(bin_spike_count)).astype(bool)
    #     # Remove very short ONs
    #     active_bin = _flip_short_periods(active_bin, 1, on_min_duration, binsize)

    active_bin = (bin_spike_count > min(bin_spike_count) + 1).astype(bool)

    # Remove short off periods
    active_bin = _flip_short_periods(active_bin, 0, off_min_duration, binsize)

    # Remove off periods with above-threshold firing
    # FR during ON so far
    on_fr = np.sum(bin_spike_count[active_bin]) / (np.sum(active_bin) * binsize)
    thresh_fr = on_fr * off_on_fr_ratio_threshold
    active_bin = _flip_low_fr_offs(bin_spike_count, active_bin, thresh_fr, binsize)

    if all(active_bin):
        # Found only ONs
        raise FailedInitializationException()
    elif not any(active_bin):
        # Found only OFFs
        raise FailedInitializationException()

    return active_bin.astype(int)


def fit_init_poisson_params(
    bin_spike_count,
    bin_history_spike_count,
    binsize,
    init_state_off_on_fr_ratio_thresh=None,
    verbose=True,
):  # 1D arrays

    ## Result
    endog = pd.DataFrame({"count": bin_spike_count})

    ## Predictors
    state = get_initial_state_estimate(
        bin_spike_count,
        binsize,
        off_on_fr_ratio_threshold=init_state_off_on_fr_ratio_thresh,
    )
    exog = sm.add_constant(
        pd.DataFrame(
            {
                # "state": np.zeros((nbins,)),
                "state": state,
                "history": bin_history_spike_count,
            }
        )
    )
    # count ~ mu + alphaa * state + betaa * bin_history
    mod = sm.GLM(endog, exog, family=sm.families.Poisson(link=sm.families.links.Log()))
    res = mod.fit()

    if verbose:
        print(res.summary())

    return res.params.values


def newton_ralphson(
    bin_spike_count,
    Z,
    bin_history_spike_count,
    init_alphaa,
    init_betaa,
    init_mu,
    n_iter,
):

    assert bin_spike_count.shape[1] > bin_spike_count.shape[0]  # Horizontal
    assert (
        bin_history_spike_count.shape[1] > bin_history_spike_count.shape[0]
    )  # Horizontal

    if bin_history_spike_count.shape[0] > 1:
        raise NotImplementedError()

    mu, alphaa = 0.0, 0.0
    if not isinstance(init_betaa, float):
        raise NotImplementedError()
    else:
        betaa = 0.0

    # Update mu
    temp0 = np.sum(bin_spike_count, axis=None)
    update = init_mu + 0.01 * np.random.randn()  # Differ from MATLAB output here
    for _ in range(n_iter):
        g = (
            np.sum(
                np.exp(
                    update
                    + init_alphaa * Z
                    + init_betaa * bin_history_spike_count
                    # # If betaa is not scalar (use np.dot?)
                    # update + init_alphaa * Z \
                    #     + np.matmul(
                    #         init_betaa.transpose(),
                    #         bin_history_spike_count
                    #     )
                )
            )
            - temp0
        )
        gprime = np.sum(
            np.exp(
                update
                + init_alphaa * Z
                + init_betaa * bin_history_spike_count
                # # # If betaa is not scalar (use np.dot?)
                # update + init_alphaa * Z \
                #     + np.matmul(
                #         init_betaa.transpose(),
                #         bin_history_spike_count
                #     )
            )
        )  # Derivative w.r.t update
        update = update - g / gprime
        mu = update

    # Update alphaa
    temp1 = np.sum(bin_spike_count * Z, axis=None)
    update = init_alphaa + 0.01 * np.random.randn()  # Differ from MATLAB output here
    for _ in range(n_iter):
        g = (
            np.sum(
                Z
                * np.exp(
                    mu
                    + update * Z
                    + init_betaa * bin_history_spike_count
                    # # If betaa is not scalar (use np.dot?)
                    # mu + update * Z \
                    #     + np.matmul(
                    #         init_betaa.transpose(),
                    #         bin_history_spike_count
                    #     )
                )
            )
            - temp1
        )
        gprime = np.sum(
            Z
            * Z
            * np.exp(
                mu
                + update * Z
                + init_betaa * bin_history_spike_count
                # # If betaa is not scalar (use np.dot?)
                # mu + update * Z \
                #     + np.matmul(
                #         init_betaa.transpose(),
                #         bin_history_spike_count
                #     )
            )
        )  # Derivative w.r.t update
        update = update - g / gprime

        alphaa = update

    # Update betaa
    d = bin_history_spike_count.shape[0]
    if d == 1:
        temp2 = np.sum(bin_spike_count * bin_history_spike_count, axis=None)
        update = init_betaa + 0.01 * np.random.randn()  # Differ from MATLAB output here
        for _ in range(n_iter):
            g = (
                np.sum(
                    bin_history_spike_count
                    * np.exp(mu + alphaa * Z + update * bin_history_spike_count)
                )
                - temp2
            )
            gprime = np.sum(
                bin_history_spike_count
                * bin_history_spike_count
                * np.exp(mu + alphaa * Z + update * bin_history_spike_count)
            )  # Derivative w.r.t update
            update = update - g / gprime

            betaa = update
    elif d >= 1:
        # Corresponding piece of matlab code from Zhe Sage Chen:
        # %[d,ydim] = size(count);
        # %[1,ydim] = size(Y);
        # tem2 = sum(repmat(Y,d,1) .* count, 2);
        # update = beta_old + 0.01*randn(d,1);
        # for i = 1:Iter
        #     g = sum(count .* repmat( exp(mu_new + alpha_new * Z + update' * count), d, 1), 2) - tem2;  % vector
        #     gprime = count .* repmat( exp(mu_new + alpha_new * Z + update' * count), d, 1) * count';   % matrix
        #     update = update - inv(gprime) * g;

        #     beta_new = update;
        # end
        raise NotImplementedError()

    return alphaa, betaa, mu
