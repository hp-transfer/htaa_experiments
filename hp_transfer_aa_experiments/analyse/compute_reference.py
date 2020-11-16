import argparse

from collections import defaultdict
from pathlib import Path

import yaml.representer

from hp_transfer_aa_experiments.analyse.read_results import load_data_to_df


def row_to_references(row):
    return [min(row.losses[:k]) for k in [10, 20, 40]]


def get_reference(results_path, approach):
    df = load_data_to_df(results_path)
    df = df[df.approach == approach]
    df[["10_loss", "20_loss", "40_loss"]] = df.apply(
        row_to_references, axis="columns", result_type="expand"
    )
    df = df.drop(columns=["loss", "losses", "num_hyperparameters", "approach", "runtype"])
    df = df.groupby(["benchmark", "trajectory", "adjustment", "development_step"])
    df = df[["10_loss", "20_loss", "40_loss"]].mean()
    df_dict = df.to_dict("index")
    references = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for index in df_dict.keys():
        benchmark, trajectory, adjustment, development_step = index
        references[benchmark][trajectory][adjustment][development_step] = df_dict[index]
    return references


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--approach", default="random")
    args = parser.parse_args()
    references = get_reference(args.results_path, args.approach)

    yaml.add_representer(defaultdict, yaml.representer.Representer.represent_dict)
    with args.output_file.open("w") as out_stream:
        yaml.dump(references, out_stream)
