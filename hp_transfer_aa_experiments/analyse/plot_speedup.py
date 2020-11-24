import argparse

from pathlib import Path

import matplotlib as mpl
import pandas as pd

from hp_transfer_aa_experiments.analyse._plot_utils import get_approach_spelling
from hp_transfer_aa_experiments.analyse._plot_utils import plot_aggregates
from hp_transfer_aa_experiments.analyse._plot_utils import plot_global_aggregates
from hp_transfer_aa_experiments.analyse.read_results import get_approach_data
from hp_transfer_aa_experiments.analyse.read_results import load_data_to_df


def _df_to_mean_ratio_with_tpe(df, data_tpe, groupby_cols):
    df = df.fillna(400)
    data_tpe = data_tpe.fillna(400)

    mean = df.groupby(groupby_cols).mean()
    mean_tpe = data_tpe.groupby(groupby_cols).mean()

    mean["10"] = mean_tpe["10"] / mean["10"]
    mean["20"] = mean_tpe["20"] / mean["20"]
    mean["40"] = mean_tpe["40"] / mean["40"]

    return mean


# pylint: disable=anomalous-backslash-in-string
def analyse_results(results_path, output_dir, reference_losses):
    df = load_data_to_df(
        results_path, offline_cache=True, reference_losses=reference_losses
    )
    df = df.drop(
        columns=["loss", "losses", "repeat", "num_hyperparameters", "development_step"]
    )

    comparisons = [
        # [["tpe2"], "tpe"],
        # [["random"], "tpe"],
        # [["transfer_tpe_no_best_first", "transfer_tpe_no_ttpe"], "tpe"],
        # [["transfer_tpe_no_ttpe", "transfer_tpe"], "tpe"],
        [["gp"], "gp2"],
        [["gp"], "random"],
        [["transfer_intersection_model_gp_no_ra", "transfer_best_first_gp"], "gp"],
        [
            ["transfer_intersection_model_best_first_gp_no_ra", "transfer_best_first_gp"],
            "gp",
        ],
        [["transfer_intersection_model_best_first_gp_no_ra"], "transfer_best_first_gp"],
        [["gp"], "tpe"],
    ]
    for approaches, baseline_approach in comparisons:
        data_dfs = [get_approach_data(df, approach) for approach in approaches]
        data_baseline = get_approach_data(df, baseline_approach)
        mean_ratios = [
            _df_to_mean_ratio_with_tpe(
                data_df,
                data_baseline,
                ["benchmark", "trajectory", "adjustment", "runtype"],
            )
            for data_df in data_dfs
        ]
        for mean_ratio, approach in zip(mean_ratios, approaches):
            mean_ratio["approach"] = approach
        mean_ratio = pd.concat(mean_ratios)
        if baseline_approach == "gp2":
            ymajorlocator = mpl.ticker.MultipleLocator(0.2)
            yminorlocator = mpl.ticker.MultipleLocator(0.2)
        elif (
            approaches[0] == "transfer_tpe_no_best_first"
            and baseline_approach == "transfer_tpe"
        ):
            ymajorlocator = mpl.ticker.MultipleLocator(0.5)
        elif approaches[0] == "random":
            ymajorlocator = mpl.ticker.MultipleLocator(0.5)
        else:
            ymajorlocator = mpl.ticker.MultipleLocator(1)
            yminorlocator = mpl.ticker.MultipleLocator(1)
        plot_global_aggregates(
            mean_ratio,
            output_dir,
            f"global_speedup_{'+'.join(approaches)}_over_{baseline_approach}",
            ylabel=f"Speedup Over {get_approach_spelling(baseline_approach)}",
            xlabel="GP Evaluations for\nReference Objective [\#]",
            yline=1,
            ymajorlocator=ymajorlocator,
            yminorlocator=yminorlocator,
            approach_hue=len(approaches) == 2,
            geometric_mean=True,
        )

    comparisons_detail = [
        [["transfer_intersection_model_gp_no_ra", "transfer_best_first_gp"], "gp"],
        [
            ["transfer_intersection_model_best_first_gp_no_ra", "transfer_best_first_gp"],
            "gp",
        ],
        [
            [
                "transfer_intersection_model_best_first_gp_no_ra",
                "transfer_intersection_model_gp_no_ra",
            ],
            "transfer_best_first_gp",
        ],
    ]
    for runtype, df_runtype in df.groupby("runtype"):
        df_runtype = df_runtype.drop(columns=["runtype"])
        for approaches, baseline_approach in comparisons_detail:
            data_dfs = [
                get_approach_data(df_runtype, approach) for approach in approaches
            ]
            data_baseline = get_approach_data(df_runtype, baseline_approach)
            mean_ratios = [
                _df_to_mean_ratio_with_tpe(
                    data_df,
                    data_baseline,
                    ["benchmark", "trajectory", "adjustment"],
                )
                for data_df in data_dfs
            ]
            for mean_ratio, approach in zip(mean_ratios, approaches):
                mean_ratio["approach"] = approach
            mean_ratio = pd.concat(mean_ratios)
            plot_aggregates(
                mean_ratio,
                output_dir,
                f"speedup_{'+'.join(approaches)}_over_{baseline_approach}_{runtype}",
                ylabel=f"Speedup Over {get_approach_spelling(baseline_approach)}",
                xlabel="GP Evaluations for\nReference Objective [\#]",
                yline=1,
                ymajorlocator=mpl.ticker.MultipleLocator(1),
                yminorlocator=mpl.ticker.MultipleLocator(1),
                approach_hue=len(approaches) == 2,
                geometric_mean=True,
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
