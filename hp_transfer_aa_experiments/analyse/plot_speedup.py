import argparse

from pathlib import Path

import matplotlib as mpl
import pandas as pd

from hp_transfer_aa_experiments.analyse._plot_utils import get_approach_spelling
from hp_transfer_aa_experiments.analyse._plot_utils import plot_aggregates
from hp_transfer_aa_experiments.analyse._plot_utils import plot_global_aggregates
from hp_transfer_aa_experiments.analyse._plot_utils import plot_global_aggregates_stacked
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


def _build_mean_ratio(approaches, baseline_approach, df):
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
    return mean_ratio


# pylint: disable=anomalous-backslash-in-string
def analyse_results(
    results_path, results_path_tpe, output_dir, reference_losses, reference_losses_tpe
):
    df = load_data_to_df(
        results_path, offline_cache=True, reference_losses=reference_losses
    )
    df = df.drop(
        columns=["loss", "losses", "repeat", "num_hyperparameters", "development_step"]
    )
    df_tpe = load_data_to_df(
        results_path_tpe, offline_cache=True, reference_losses=reference_losses_tpe
    )
    df_tpe = df_tpe.drop(
        columns=["loss", "losses", "repeat", "num_hyperparameters", "development_step"]
    )

    df = df_tpe

    comparisons = [
        # [["transfer_tpe_no_ttpe"], "tpe2"],
        # [["transfer_tpe_no_best_first"], "tpe2"],
        # [["transfer_tpe"], "tpe2"],
        # [["gp"], "gp2"],
        [["tpe2"], "tpe3"],
        # [["gp", "tpe"], "random"],
    ]
    for approaches, baseline_approach in comparisons:
        mean_ratio = _build_mean_ratio(approaches, baseline_approach, df)
        if baseline_approach == "gp2" or baseline_approach == "tpe3":
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
            xlabel="TPE Evaluations for\nReference Objective [\#]",
            yline=1,
            ymajorlocator=ymajorlocator,
            yminorlocator=yminorlocator,
            approach_hue=len(approaches) > 1,
            approach_split=len(approaches) == 2,
            geometric_mean=True,
        )

    comparisons_a = [
        # [
        #     [
        #         "transfer_intersection_model_gp_no_ra",
        #         "transfer_best_first_gp",
        #         "transfer_intersection_model_best_first_gp_no_ra",
        #     ],
        #     "gp",
        # ],
    ]
    comparisons_b = [
        # [
        #     [
        #         "transfer_tpe_no_best_first",
        #         "transfer_tpe_no_ttpe",
        #         "transfer_tpe",
        #     ],
        #     "tpe2",
        # ],
    ]
    for (approaches_a, baseline_approach_a), (approaches_b, baseline_approach_b) in zip(
        comparisons_a, comparisons_b
    ):
        mean_ratio_a = _build_mean_ratio(approaches_a, baseline_approach_a, df)
        mean_ratio_b = _build_mean_ratio(approaches_b, baseline_approach_b, df_tpe)
        ymajorlocator = mpl.ticker.MultipleLocator(1)
        yminorlocator = mpl.ticker.MultipleLocator(1)
        plot_global_aggregates_stacked(
            mean_ratio_a,
            mean_ratio_b,
            output_dir,
            f"global_speedup_{'+'.join(approaches_a)}__and_{'+'.join(approaches_b)}_over_{baseline_approach_a}_and_{baseline_approach_b}",
            ylabel=[
                f"Speedup Over {get_approach_spelling(baseline_approach_a)}",
                f"Speedup Over {get_approach_spelling(baseline_approach_b)}",
            ],
            xlabel="GP/TPE Evaluations for\nReference Objective [\#]",
            yline=1,
            ymajorlocator=ymajorlocator,
            yminorlocator=yminorlocator,
            approach_hue=len(approaches_a) > 1,
            approach_split=len(approaches_a) == 2,
            geometric_mean=True,
        )

    comparisons_detail = [
        # [
        #     [
        #         "transfer_intersection_model_gp_no_ra",
        #         "transfer_best_first_gp",
        #         "transfer_intersection_model_best_first_gp_no_ra",
        #     ],
        #     "gp",
        # ],
        [
            [
                "transfer_tpe_no_best_first",
                "transfer_tpe_no_ttpe",
                "transfer_tpe",
            ],
            "tpe2",
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
                xlabel="TPE Evaluations for\nReference Objective [\#]",
                yline=1,
                ymajorlocator=mpl.ticker.MultipleLocator(1),
                yminorlocator=mpl.ticker.MultipleLocator(1),
                approach_hue=len(approaches) > 1,
                approach_split=len(approaches) == 2,
                geometric_mean=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--results_path_tpe", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_losses",
        type=Path,
        default="hp_transfer_aa_experiments/reference_losses.yaml",
    )
    parser.add_argument(
        "--reference_losses_tpe",
        type=Path,
        default="hp_transfer_aa_experiments/reference_losses_tpe.yaml",
    )
    args = parser.parse_args()
    analyse_results(
        args.results_path,
        args.results_path_tpe,
        args.output_dir,
        args.reference_losses,
        args.reference_losses_tpe,
    )
