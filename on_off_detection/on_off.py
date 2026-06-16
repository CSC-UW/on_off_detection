from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .methods.hmmem import HMMEM_PARAMS, run_hmmem
from .methods.sticky import STICKY_PARAMS, run_sticky
from .methods.threshold import THRESHOLD_PARAMS, run_threshold
from .utils import subset_sorted_train, kway_mergesort, slice_and_concat_sorted_train

METHODS = {
    "threshold": run_threshold,
    "hmmem": run_hmmem,
    "sticky": run_sticky,
}

DF_PARAMS = {
    "threshold": THRESHOLD_PARAMS,
    "hmmem": HMMEM_PARAMS,
    "sticky": STICKY_PARAMS,
}


def _run_detection(
    cluster_ids,
    detection_func,
    trains_list,
    bouts_df,
    params,
    verbose=True,
    i=None,
):
    if i is not None:
        if verbose:
            print(f"Run #{i+1}/N, cluster_ids={cluster_ids}")

    if verbose:
        print(f"Merge N={len(trains_list)} spike trains")
    merged_train = kway_mergesort(trains_list)

    Tmax = bouts_df.duration.sum()
    if verbose:
        print(f"Cut and concatenate bouts: subselect T={Tmax} seconds within bouts")
    sliced_concat_train = slice_and_concat_sorted_train(
        merged_train, bouts_df
    )  # Times in cut-and-concatenated bouts
    if not len(sliced_concat_train):
        raise ValueError(
            "Attempting to perform on/off detection on an empty spike train"
        )

    on_off_df, output_info = detection_func(
        sliced_concat_train,
        Tmax,
        params,
        verbose=verbose,
    )
    # on_off_df["cluster_ids"] = [cluster_ids] * len(on_off_df)

    if verbose:
        print("Recover original start/end times from non-cut-and-concat data...")
    # Add bout info for computed on_off periods
    # - 'state' from original bouts_df
    # - Mark on/off periods that span non-consecutive bouts as 'interbout'
    # - Mark first and last bout as 'interbout'
    # - recover start/end time in original (not cut/concatenated) time (Kinda nasty)
    on_off_orig = on_off_df.copy()
    on_off_df["bout_state"] = "interbout"
    bout_concat_start_time = 0  # Start time in cut and concatenated data
    for i, row in bouts_df.iterrows():
        bout_concat_end_time = bout_concat_start_time + row["duration"]
        bout_on_off = (on_off_orig["start_time"] > bout_concat_start_time) & (
            on_off_orig["end_time"] < bout_concat_end_time
        )  # Strict comparison also excludes first and last bout
        # start and end time in cut-concatenated data
        # on_off_df.loc[
        #     bout_on_off, "start_time_relative_to_concatenated_bouts"
        # ] = on_off_df.loc[bout_on_off, "start_time"]
        # on_off_df.loc[
        #     bout_on_off, "end_time_relative_to_concatenated_bouts"
        # ] = on_off_df.loc[bout_on_off, "end_time"]
        # Start and end time in original recording
        bout_offset = (
            -bout_concat_start_time + row["start_time"]
        )  # Offset from concat to real time for this bout
        # print('offset', bout_offset)
        on_off_df.loc[bout_on_off, "start_time"] = (
            on_off_df.loc[bout_on_off, "start_time"] + bout_offset
        )
        on_off_df.loc[bout_on_off, "end_time"] = (
            on_off_df.loc[bout_on_off, "end_time"] + bout_offset
        )
        # bout information
        bout_state = row["state"]
        on_off_df.loc[bout_on_off, "bout_state"] = bout_state
        # on_off_df.loc[bout_on_off, "bout_idx"] = row.name
        # on_off_df.loc[bout_on_off, "bout_concat_start_time"] = row["start_time"]
        # on_off_df.loc[bout_on_off, "bout_concat_end_time"] = row["end_time"]
        # on_off_df.loc[bout_on_off, "bout_duration"] = row["duration"]
        # Go to next bout
        bout_concat_start_time = bout_concat_end_time

        # Total state time per condition
        # for bout_state in
        #     total_state_time = bouts_df[bouts_df["state"] == bout_state].duration.sum()
        #     on_off_df.loc[
        #         on_off_df["bout_state"] == bout_state, "bout_state_total_time"
        #     ] = total_state_time

    on_off_df = on_off_df[on_off_df["bout_state"] != "interbout"].reset_index(
        drop=True
    )

    if verbose:
        print(f"Found N={len(on_off_df)} on/off periods.")

    return on_off_df, output_info


class OnOffModel(object):
    """Run ON and OFF-state detection from MUA data.

    Args:
        trains_list (list of array-like): Sorted MUA spike times for each cluster.
                Spike outside of bouts are ignored.
        bouts_df (pd.DataFrame): Frame containing bouts of interest. Must contain
                'start_time', 'end_time', 'duration' and 'state' columns. We consider
                only spikes within these bouts for on-off detection (by
                cutting-and-concatenating the trains of each cluster).  The
                "state", "start_time" and "end_time" of the bout each on or
                off period pertains to is saved in the "bout_state",
                "bout_start_time" and "bout_end_time" columns. ON or OFF
                periods that are not STRICTLY comprised within bouts are
                dismissed ()

    Kwargs:
            cluster_ids (array-like): Cluster ids. Added to output df if provided
                    (default None)
            method (string): Method used for On-off detection.
            params (dict): Dict of parameters. Recognized params depend of <method>
            verbose (bool): (default True)
    """

    def __init__(
        self,
        trains_list,
        bouts_df,
        cluster_ids=None,
        method="hmmem",
        params=None,
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
            [
                c in bouts_df.columns
                for c in ["start_time", "end_time", "duration", "state"]
            ]
        )
        assert bouts_df.duration.sum(), f"Empty bouts"
        self.bouts_df = bouts_df

        # Method and params
        self.method = method
        if self.method not in METHODS.keys():
            raise ValueError(
                f"Unrecognized method. Available methods are {METHODS.keys()}"
            )
        self.detection_func = METHODS[method]
        if params is None:
            params = {}
        unrecognized_params = set(params.keys()) - set(DF_PARAMS[method].keys())
        if len(unrecognized_params):
            raise ValueError(
                f"Unrecognized parameter keys for on-off detection method `{method}`: "
                f"{unrecognized_params}.\n\n"
                f"Default (recognized) parameters for this method: {DF_PARAMS[method]}"
            )
        self.params = {k: v for k, v in DF_PARAMS[method].items()}
        self.params.update(params)

        # Output stuff
        self.verbose = verbose
        self.on_off_df = None
        self.output_info = None

    def run(self):
        if self.verbose:
            print("Run on-off detection on pooled data")
        self.on_off_df, self.output_info = _run_detection(
            self.cluster_ids,
            self.detection_func,
            self.trains_list,
            self.bouts_df,
            self.params,
            self.verbose,
            i=None,
        )
        return self.on_off_df, self.output_info

    # def save():
    # 	if self.output_dir is None:
    # 		raise ValueError()
    # 	self.output_dir.mkdir(exist_ok=True, parents=True)
    # 	self.res.to_csv(self.output_dir/'on-off-times.csv')
    # 	self.stats.to_csv(self.output_dir/'on-off-stats.csv')
