# on_off_detection: Detect ON/OFF periods from MUA or single-unit activity.

> **Workspace fork notes.** Methods: `threshold`, `hmmem`, and `sticky`
> (a sticky 2-state Poisson HMM after Li & La Camera 2025; numba-accelerated
> forward-backward/Viterbi with a pure-numpy fallback). Also fixed a duration
> histogram bin-center bug in `methods/threshold.py` and gated the statsmodels
> GLM summary print in `methods/hmmem.py` behind `verbose`. Installed editable in
> the `gfys_workspace` uv environment. Tests in `tests/`.

## Examples

### Detect all ON and OFF periods from MUA


```python
from on_off_detection import OnOffModel
import numpy as np

cluster_ids = [
    '1',
    '2',
]
trains_list = [
    np.random.randint(0, 3600, 300), # 300 Spikes for cluster 1
    np.random.randint(0, 3600, 300), # 300 Spikes for cluster 2
]

bouts_df = pd.DataFrame({
    'start_time': [0, 3000],
    'end_time': [2000, 3600],
    'state': ["NREM", "NREM"],
})  # Cut and concatenate spikes within these bouts to do detection
bouts_df["duration"] = bouts_df["end_time"] - bouts_df["start_time"]

#### Threshold based method #######
on_off_method = 'threshold'  # Implements: Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown; Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.  doi: https://doi.org/10.1162/neco.2009.06-08-799
on_off_params = {
	'binsize': 1,
	'smooth_sd_counts': 3, # Units of binsize
	'binsize_smoothed_count_hist': 0.02, # 
	'smooth_sd_smoothed_count_hist': 1, # Units of "binsize_smoothed_count_hist"
	'binsize_duration_hist': 0.01, # 
	'count_threshold': None,
	'gap_threshold': None,
}

#### HMM-EM ########
on_off_method = 'hmmem'  # Implements: Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown; Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.  doi: https://doi.org/10.1162/neco.2009.06-08-799
on_off_params = {
    'binsize': 0.010, # (s) (Discrete algorithm)
    'history_window_nbins': 3, # Size of history window IN BINS
    'n_iter_EM': 200,  # Number of iterations for EM
    'n_iter_newton_ralphson': 100,
    'init_A': np.array([[0.1, 0.9], [0.01, 0.99]]), # Initial transition probability matrix
    'init_state_estimate_method': 'liberal',  # Method to find inital OFF states to fit GLM model with. Ignored if init_mu/alphaa/betaa are specified. Either of 'conservative'/'liberal'/'intermediate'
    'init_mu': None,  # ~ OFF rate. Fitted to data if None
    'init_alphaa': None,  # ~ difference between ON and OFF rate. Fitted to data if None
    'init_betaa': None, # ~ Weight of recent history firing rate. Fitted to data if None,
    'gap_threshold': None, # Merge active states separated by less than gap_threhsold
}
#### Sticky Poisson-HMM ########
on_off_method = 'sticky'  # Implements: Li & La Camera 2025, "A sticky Poisson hidden Markov model...", PLOS One
on_off_params = {
    'binsize': 0.010,       # (s)
    'min_dwell': 0.050,     # (s) minimum expected dwell -> self-transition floor delta = 1 - binsize/min_dwell
    'off_rate_max': 20.0,   # (Hz) near-silence cap on the OFF-state rate (None disables). OFF = (near-)silent population.
    'n_iter_EM': 100,       # max Baum-Welch iterations
    'tol': 1e-4,            # log-likelihood convergence tolerance
    'min_off_duration': None,  # (s) optional post-hoc merge of short OFFs
}
##########

model = OnOffModel(
    trains_list,
    bouts_df,
    cluster_ids=cluster_ids,
    method=on_off_method,
    params=on_off_params,
)
on_off_df = model.run()
on_off_df = model.on_off_df # Pandas frame with 'state', 'start_time', 'end_time', 'duration' columns
```

### Spatial OFF detection

```python
from on_off_detection import SpatialOffModel
import numpy as np

cluster_ids = [
    '1', '2', '3'
]
cluster_depths = [
    1, 2, 3,
]
trains_list = [
    np.random.randint(0, 3600, 300), # 300 Spikes for cluster 1
    np.random.randint(0, 3600, 300), # 300 Spikes for cluster 1
    np.random.randint(0, 3600, 300), # 300 Spikes for cluster 2
]

bouts_df = pd.DataFrame({
    'start_time': [0, 3000],
    'end_time': [2000, 3600],
    'state': ["NREM", "NREM"],
})  # Cut and concatenate spikes within these bouts to do detection
bouts_df["duration"] = bouts_df["end_time"] - bouts_df["start_time"]

# Method used for ON-OFF detection within each spatial window
# (Same as OnOffModel)

#### HMM-EM ########
on_off_method = 'hmmem'  # Implements: Zhe Chen, Sujith Vijayan, Riccardo Barbieri, Matthew A. Wilson, Emery N. Brown; Discrete- and Continuous-Time Probabilistic Models and Algorithms for Inferring Neuronal UP and DOWN States. Neural Comput 2009; 21 (7): 1797–1862.  doi: https://doi.org/10.1162/neco.2009.06-08-799
on_off_params = {
    'binsize': 0.010, # (s) (Discrete algorithm)
    'history_window_nbins': 3, # Size of history window IN BINS
    'n_iter_EM': 200,  # Number of iterations for EM
    'n_iter_newton_ralphson': 100,
    'init_A': np.array([[0.1, 0.9], [0.01, 0.99]]), # Initial transition probability matrix
    'init_state_estimate_method': 'liberal',  # Method to find inital OFF states to fit GLM model with. Ignored if init_mu/alphaa/betaa are specified. Either of 'conservative'/'liberal'/'intermediate'
    'init_mu': None,  # ~ OFF rate. Fitted to data if None
    'init_alphaa': None,  # ~ difference between ON and OFF rate. Fitted to data if None
    'init_betaa': None, # ~ Weight of recent history firing rate. Fitted to data if None,
    'gap_threshold': None, # Merge active states separated by less than gap_threhsold
}

# Parameters for aggregation of OFF states across windows
spatial_params = {
	# Windowing/pooling
	'window_size_min': 200,  # (um) Smallest spatial "grain" for pooling
	'window_overlap': 0.5,  # (no unit) Overlap between windows within each spatial grain 
	'window_size_step': 200,  # (um) Increase in size of windows across successive spatial "grains"
	# Merging of OFF state between and across grain
	'merge_max_time_diff': 0.050, # (s). To be merged, off states need their start & end times to differ by less than this
	'nearby_off_max_time_diff': 3, # (sec). #TODO
	'sort_all_window_offs_by': ['off_area', 'duration', 'start_time', 'end_time'],  # How to sort all OFFs before iteratively merging
	'sort_all_window_offs_by_ascending': [False, False, True, True],
}

spatial_off_model = SpatialOffModel(
    trains_list,
    cluster_depths,
    bouts_df,
    cluster_ids=cluster_ids,
    on_off_method=on_off_method,
    on_off_params=on_off_params,
    spatial_params=spatial_params,
    n_jobs=1,
    verbose=True
)
off_df = spatial_off_model.run()

all_windows_on_off_df = spatial_off_model.all_windows_on_off_df # On/Off across all windows. Pandas frame with 'state', 'start_time', 'end_time', 'duration', 'window_depths' columns
off_df = spatial_off_model.off_df # Offs after merging. Pandas frame with 'state', 'start_time', 'end_time', 'duration', 'window_depths' columns
```


### Detect OFF state evoked by stimulation

#TODO 
Check out the "EvokedOff" class