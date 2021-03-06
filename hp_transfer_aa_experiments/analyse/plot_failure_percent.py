import argparse

from pathlib import Path

import numpy as np

from hp_transfer_aa_experiments.analyse._plot_utils import plot_aggregates
from hp_transfer_aa_experiments.analyse._plot_utils import plot_global_aggregates
from hp_transfer_aa_experiments.analyse.read_results import load_data_to_df


def _df_to_nan_stats(data, groupby_cols):
    grouped = data.groupby(groupby_cols)
    n = 25  # grouped.size().unique()[0]
    nan_percent = 1 - grouped.count() / n
    return nan_percent * 100


# pylint: disable=anomalous-backslash-in-string
def analyse_results(results_path, output_dir, reference_losses):
    df = load_data_to_df(
        results_path, offline_cache=True, reference_losses=reference_losses
    )
    df = df.drop(
        columns=["loss", "losses", "num_hyperparameters", "repeat", "development_step"]
    )

    nan_checks = [
        # ["transfer_top_gp", "transfer_importance_gp"],
        # ["transfer_intersection_model_gp_no_ra", "transfer_best_first_gp", "gp"],
        ["transfer_top", "transfer_importance"],
        ["transfer_tpe_no_best_first", "transfer_tpe_no_ttpe", "tpe2"],
    ]
    for approaches in nan_checks:
        data = df[
            np.logical_or.reduce([df.approach == approach for approach in approaches])
        ]
        nan_percent = _df_to_nan_stats(
            data,
            [
                "benchmark",
                "trajectory",
                "adjustment",
                "runtype",
                "approach",
            ],
        )
        plot_global_aggregates(
            nan_percent,
            output_dir,
            f"global_nan_percent_{'+'.join(approaches)}",
            yline=None,
            ylabel="Failed Runs [%]",
            xlabel="TPE Evaluations for\nReference Objective [\#]",
            approach_split=len(approaches) == 2,
            approach_hue=len(approaches) > 1,
        )

    nan_checks_detail = [
        # ["transfer_tpe", "tpe"],
        # ["transfer_top", "transfer_importance"],
    ]
    for runtype, df_ in df.groupby("runtype"):
        df_ = df_.drop(columns=["runtype"])
        for approach_a, approach_b in nan_checks_detail:
            data = df_[(df_.approach == approach_a) | (df_.approach == approach_b)]
            nan_percent = _df_to_nan_stats(
                data,
                ["benchmark", "trajectory", "adjustment", "approach"],
            )
            plot_aggregates(
                nan_percent,
                output_dir,
                f"nan_percent_{approach_a}_and_{approach_b}_{runtype}",
                yline=None,
                ylabel="Failed Runs [%]",
                xlabel="GP Evaluations for\nReference Objective [\#]",
                approach_hue=True,
                approach_split=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_losses",
        type=Path,
        default="hp_transfer_aa_experiments/reference_losses.yaml",
    )
    args = parser.parse_args()
    analyse_results(args.results_path, args.output_dir, args.reference_losses)
