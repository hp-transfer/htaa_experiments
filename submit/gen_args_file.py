import argparse
import itertools

from pathlib import Path


parser = argparse.ArgumentParser()

parser.add_argument("--benchmark", required=True)
parser.add_argument("--approaches", required=True, help="In the form 'A B C'")
parser.add_argument("--trajectory_ids", default="0", help="In the form '0 2 3'")
parser.add_argument("--adjustment_ids", default="a", help="In the form 'a b c'")
parser.add_argument("--runtypes", default="online_5", help="In the form 'A B'")
parser.add_argument("--num_workers", default=1, type=int)
parser.add_argument("--experiment_group", default="test", required=True)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--offset_repetition", type=int, default=0)
parser.add_argument("--nic_name", default="eth0")
parser.add_argument("--argfile", required=True)

args = parser.parse_args()

approaches = args.approaches.split(" ")
trajectory_ids = args.trajectory_ids.split(" ")
adjustment_ids = args.adjustment_ids.split(" ")
runtypes = args.runtypes.split(" ")

all_runs = []
for runtype in runtypes:
    for repetition in range(args.repeats):
        repetition += args.offset_repetition
        for run_type_id, (approach, trajectory_id, adjustment_id) in enumerate(
            itertools.product(approaches, trajectory_ids, adjustment_ids)
        ):
            for worker_id in range(args.num_workers):
                experiment_name = (
                    f"{args.benchmark}/"
                    f"{runtype}/"
                    f"{approach}/"
                    f"trajectory_id_{trajectory_id}/"
                    f"adjustment_id_{adjustment_id}/"
                    f"repeat_{repetition}"
                )
                all_runs.append(
                    f"experiment_group={args.experiment_group} "
                    f"benchmark={args.benchmark} "
                    f"approach={approach} "
                    f"benchmark.trajectory_id={trajectory_id} "
                    f"benchmark.adjustment_id={adjustment_id} "
                    f"runtype={runtype} "
                    f"run_id={repetition} "
                    f"worker_id={worker_id} "
                    f"nic_name={args.nic_name} "
                    f"seed={args.repeats * run_type_id + repetition} "
                    f"experiment_name={experiment_name}"
                )

Path(args.argfile).write_text("\n".join(all_runs))
