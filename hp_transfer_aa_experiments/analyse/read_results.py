import pickle

from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _parse_num_evals(df, reference_losses):
    with Path(reference_losses).open("r") as stream:
        reference_losses = yaml.safe_load(stream)

    def _row_to_num_evals(row, reference_losses):
        def first_true_eval_number(iterable, condition):
            try:
                return next(i for i, v in enumerate(iterable) if condition(v)) + 1
            except StopIteration:
                return None

        reference_losses = reference_losses[row.benchmark][row.trajectory]
        reference_losses = reference_losses[row.adjustment][row.development_step]
        d1_reference, d3_reference, d9_reference = reference_losses.values()

        d1_evals = first_true_eval_number(row.losses, lambda loss: loss <= d1_reference)
        d3_evals = first_true_eval_number(row.losses, lambda loss: loss <= d3_reference)
        d9_evals = first_true_eval_number(row.losses, lambda loss: loss <= d9_reference)
        return [d1_evals, d3_evals, d9_evals]

    df[["10", "20", "40"]] = df.apply(
        partial(_row_to_num_evals, reference_losses=reference_losses),
        axis="columns",
        result_type="expand",
    )
    return df


def get_batch_result_row(
    benchmark,
    runtype,
    approach,
    trajectory,
    adjustment,
    repeat,
    batch_result=None,
    batch_result_path=None,
):
    if batch_result is None:
        with batch_result_path.open("rb") as batch_result_stream:
            batch_result = pickle.load(batch_result_stream)
    result = batch_result.results[0]  # Only one task in batch for now
    losses = [run.loss for run in result.get_all_runs()]
    best_loss = min(losses)
    development_step = batch_result.step
    num_hyperparameters = len(batch_result.configspace.get_hyperparameters())
    return [
        benchmark,
        runtype,
        approach,
        trajectory,
        adjustment,
        repeat,
        development_step,
        best_loss,
        losses,
        num_hyperparameters,
    ]


RESULT_COLUMNS = [
    "benchmark",
    "runtype",
    "approach",
    "trajectory",
    "adjustment",
    "repeat",
    "development_step",
    "loss",
    "losses",
    "num_hyperparameters",
]

_RESULT_COLUMNS_DTYPES = [str, str, str, str, str, np.int, np.int, np.float, str, np.int]

_RESULT_COLUMN_TO_DTYPE = {
    column: dtype for column, dtype in zip(RESULT_COLUMNS, _RESULT_COLUMNS_DTYPES)
}


def _read_from_tree(is_benchmark_path, results_path):
    all_rows = []
    benchmark_paths = (p for p in results_path.iterdir() if is_benchmark_path(p))
    for benchmark_path in benchmark_paths:
        for runtype_path in benchmark_path.iterdir():
            for approach_path in runtype_path.iterdir():
                for trajectory_path in approach_path.iterdir():
                    for adjustment_path in trajectory_path.iterdir():
                        for repeat_path in adjustment_path.iterdir():
                            batch_result_paths = (repeat_path / "train").glob("batch_*")
                            for batch_result_path in batch_result_paths:
                                all_rows.append(
                                    get_batch_result_row(
                                        benchmark_path.name,
                                        runtype_path.name,
                                        approach_path.name,
                                        trajectory_path.name.replace(
                                            "trajectory_id_", ""
                                        ),
                                        adjustment_path.name.replace(
                                            "adjustment_id_", ""
                                        ),
                                        int(repeat_path.name.replace("repeat_", "")),
                                        None,
                                        batch_result_path,
                                    )
                                )
    return all_rows


def load_data_to_df(results_path, offline_cache=True, reference_losses=None):
    cache_path = results_path / "_load_cache.csv"
    if offline_cache and cache_path.exists():
        return pd.read_pickle(cache_path)

    def is_benchmark_path(p):
        return p.is_dir() and not p.name.startswith("cluster_oe") and not p.name == "args"

    individual_result_path = results_path / "results"
    if individual_result_path.exists():
        dfs = []
        for individual_result in individual_result_path.iterdir():
            with individual_result.open() as result_stream:
                df = pd.read_csv(
                    result_stream,
                    sep="\t",
                    header=None,
                    names=RESULT_COLUMNS,
                    index_col=False,
                    dtype=_RESULT_COLUMN_TO_DTYPE,
                )
                dfs.append(df)
        df = pd.concat(dfs, axis=0, ignore_index=True)

        def parse_losses(losses):
            losses = losses.strip("[]")
            losses = losses.split(", ")
            return [float(loss) for loss in losses]

        df["losses"] = df["losses"].apply(parse_losses)
    else:
        all_rows = _read_from_tree(is_benchmark_path, results_path)
        df = pd.DataFrame(
            all_rows,
            columns=RESULT_COLUMNS,
        )

    df = df[df.development_step > 1]
    if reference_losses is not None:
        df = _parse_num_evals(df, reference_losses)

    if offline_cache:
        df.to_pickle(cache_path)

    return df


def get_approach_data(df, approach):
    df = df[df.approach == approach]
    return df.drop(columns="approach")
