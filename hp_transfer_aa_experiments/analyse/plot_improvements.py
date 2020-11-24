import argparse

from pathlib import Path

import pandas as pd

from hp_transfer_aa_experiments.analyse._plot_utils import plot_aggregates
from hp_transfer_aa_experiments.analyse._plot_utils import plot_global_aggregates
from hp_transfer_aa_experiments.analyse.read_results import get_approach_data
from hp_transfer_aa_experiments.analyse.read_results import load_data_to_df


def _df_to_normed_performance(df):
    def row_to_performance(row):
        return [
            min(row.losses[:10]),
            min(row.losses[:20]),
            min(row.losses[:40]),
        ]

    df[["10", "20", "40"]] = df.apply(
        row_to_performance,
        axis="columns",
        result_type="expand",
    )
    df = df.drop(
        columns=["loss", "losses", "num_hyperparameters", "repeat", "development_step"]
    )

    grouped = df.groupby(
        [
            "benchmark",
            "trajectory",
            "adjustment",
            "runtype",
            "approach",
        ]
    )
    means = grouped.mean()
    means = means.reset_index().set_index(
        ["benchmark", "trajectory", "adjustment", "runtype"]
    )

    baseline = df[df.approach == "gp"]
    baseline = baseline.drop(columns=["approach"])
    baseline_grouped = baseline.groupby(
        ["benchmark", "trajectory", "adjustment", "runtype"]
    )

    baseline_means = baseline_grouped.mean()
    baseline_stds = baseline_grouped.std()
    baseline_stds += baseline_stds.quantile(0.2)
    for reference in ["10", "20", "40"]:
        means[reference] = baseline_means[reference] - means[reference]
        means[reference] /= baseline_stds[reference]

    means = means.reset_index().set_index(["benchmark", "trajectory", "adjustment"])
    return means


def analyse_results(results_path, output_dir):
    df = load_data_to_df(results_path, offline_cache=True)
    df = _df_to_normed_performance(df)

    ylabel = r"""Improvement Over
    GP [SD\textsubscript{GP} $+ \varepsilon]$"""

    normed_checks = [
        # ["transfer_tpe_no_best_first", "transfer_tpe_no_ttpe"],
        # ["transfer_top", "transfer_importance"],
        ["transfer_top_gp", "transfer_importance_gp"],
        [
            "transfer_intersection_model_gp_no_ra",
            "transfer_best_first_gp",
            "transfer_intersection_model_best_first_gp_no_ra",
        ],
    ]
    for approaches in normed_checks:
        data_dfs = [get_approach_data(df, approach) for approach in approaches]
        for data_df, approach in zip(data_dfs, approaches):
            data_df["approach"] = approach
        data_dfs = pd.concat(data_dfs)
        data_dfs["10"] = data_dfs["10"].clip(lower=-10)
        data_dfs["20"] = data_dfs["20"].clip(lower=-10)
        data_dfs["40"] = data_dfs["40"].clip(lower=-10)
        plot_global_aggregates(
            data_dfs,
            output_dir,
            f"global_improvement_{'+'.join(approaches)}",
            ylabel=ylabel,
            xlabel=r"Evaluations [\#]",
            yline=0,
            approach_hue=len(approaches) > 1,
            clip_to_zero=False,
            approach_split=len(approaches) == 2,
        )

    normed_checks_detail = [
        ["transfer_top_gp", "transfer_importance_gp"],
        [
            "transfer_intersection_model_gp_no_ra",
            "transfer_best_first_gp",
            "transfer_intersection_model_best_first_gp_no_ra",
        ],
    ]
    for runtype, df_runtype in df.groupby("runtype"):
        df_runtype = df_runtype.drop(columns=["runtype"])
        for approaches in normed_checks_detail:
            data_dfs = [
                get_approach_data(df_runtype, approach) for approach in approaches
            ]
            for data_df, approach in zip(data_dfs, approaches):
                data_df["approach"] = approach
            data_dfs = pd.concat(data_dfs)
            data_dfs["10"] = data_dfs["10"].clip(lower=-10)
            data_dfs["20"] = data_dfs["20"].clip(lower=-10)
            data_dfs["40"] = data_dfs["40"].clip(lower=-10)
            plot_aggregates(
                data_dfs,
                output_dir,
                f"improvement_{'+'.join(approaches)}_{runtype}",
                ylabel=ylabel,
                xlabel=r"Evaluations [\#]",
                yline=0,
                clip_to_zero=False,
                approach_hue=len(approaches) > 1,
                approach_split=len(approaches) == 2,
                geometric_mean=False,
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
    analyse_results(args.results_path, args.output_dir)
