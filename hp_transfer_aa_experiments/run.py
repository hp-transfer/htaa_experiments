import contextlib
import logging
import logging.config
import random
import time

from pathlib import Path

import hp_transfer_benchmarks  # pylint: disable=unused-import
import hp_transfer_optimizers  # pylint: disable=unused-import
import hydra
import numpy as np
import yaml

from gitinfo import gitinfo
from hp_transfer_optimizers.core import nameserver as hpns
from hp_transfer_optimizers.core import result as result_utils
from hp_transfer_optimizers.core.worker import Worker
from omegaconf import OmegaConf

from hp_transfer_aa_experiments.analyse.read_results import get_batch_result_row


logger = logging.getLogger("hp_transfer_aa_experiments.run")


def _read_reference_losses(args):
    reference_losses = None
    if args.runtype.type.startswith("eval_reference"):
        reference_losses_path = hydra.utils.to_absolute_path(args.reference_losses_path)
        with Path(reference_losses_path).open("r") as stream:
            reference_losses = yaml.safe_load(stream)
        reference_losses = reference_losses[args.benchmark.name]
        reference_losses = reference_losses[str(args.benchmark.benchmark.trajectory_id)]
        reference_losses = reference_losses[str(args.benchmark.benchmark.adjustment_id)]
    return reference_losses


def _get_trial_parameters(args, reference_losses, step):
    if step == 1 and args.runtype.type in ["eval_dim", "eval_reference"]:
        trials_per_task = args.runtype.dim_factor_pre_adjustment
    else:
        trials_per_task = args.runtype.dim_factor
    logger.info(f"Using {trials_per_task} trials per task")

    if step > 1 and args.runtype.type.startswith("eval_reference"):
        trials_until_loss = reference_losses[step][f"{args.runtype.dim_factor}_loss"]
        logger.info(
            f"Also performing trials until loss {trials_until_loss :.4f}"
            f" (max {10 * trials_per_task})"
        )
    else:
        trials_until_loss = None

    return trials_per_task, trials_until_loss


def _write_batch_result(args, result_batch):
    batch_result_row = get_batch_result_row(
        args.benchmark.name,
        args.runtype.dim_factor_pre_adjustment,
        args.approach.name,
        args.benchmark.benchmark.trajectory_id,
        args.benchmark.benchmark.adjustment_id,
        args.run_id,
        result_batch,
    )
    result_path = Path(
        hydra.utils.to_absolute_path("results"),
        args.experiment_group,
        f"results/{args.experiment_name.replace('/', ',')}.csv",
    )
    result_path.parent.mkdir(exist_ok=True, parents=True)
    with result_path.open("a") as result_stream:
        result_stream.write("\t".join([str(value) for value in batch_result_row]) + "\n")


def _run_on_task_batch(
    optimizer,
    task_batch,
    configspace,
    step,
    result_trajectory,
    trials_per_task,
    trials_until_loss,
    args,
):
    do_transfer = args.approach.name.startswith("transfer")
    previous_results = result_trajectory if do_transfer else None
    result_batch = result_utils.BatchResult(step, configspace)
    for task in task_batch:
        logger.info(f"Running on task {task.identifier}")
        task_result = optimizer.run(
            configspace=configspace,
            task=task,
            n_iterations=trials_per_task,
            trials_until_loss=trials_until_loss,
            previous_results=previous_results,
        )
        result_batch.insert(task_result, task)

    if step > 1:
        _write_batch_result(args, result_batch)
    return result_batch


def _train_and_eval(optimizer, benchmark, args):
    reference_losses = _read_reference_losses(args)

    result_trajectory = result_utils.TrajectoryResult()
    for step, (train_batch, configspace) in enumerate(
        zip(benchmark.dev_trajectory, benchmark.configspace_trajectory), 1
    ):
        if args.runtype.type == "reference" and step == 1:
            continue

        logger.info(f"Step ------- {step :04d}")
        trials_per_task, trials_until_loss = _get_trial_parameters(
            args, reference_losses, step
        )
        logger.info(f"Using configspace\n{configspace}".rstrip())

        batch_result = _run_on_task_batch(
            optimizer,
            train_batch,
            configspace,
            step,
            result_trajectory,
            trials_per_task,
            trials_until_loss,
            args,
        )
        result_trajectory.insert(batch_result)


class _HPOWorker(Worker):
    def __init__(self, benchmark, **kwargs):
        super().__init__(**kwargs)

        # Only read task once
        self._benchmark = benchmark
        self._previous_task_identifier = None
        self._previous_development_stage = None
        self._task = None

    # pylint: disable=unused-argument
    def compute(
        self,
        config_id,
        config,
        budget,
        working_directory,
        *args,
        **kwargs,
    ):
        task_identifier = kwargs["task_identifier"]
        development_stage = kwargs["development_stage"]
        task_changed = (
            development_stage != self._previous_development_stage
            or self._previous_task_identifier != task_identifier
        )
        if task_changed:  # Only read task once
            self._previous_task_identifier = task_identifier
            self._previous_development_stage = development_stage
            self._task = self._benchmark.get_task_from_identifier(
                task_identifier, development_stage
            )
        if "development_step" in config:
            del config["development_step"]
        return self._task.evaluate(config)


def _run_worker(args, benchmark, working_directory):
    time.sleep(5)  # short artificial delay to make sure the nameserver is already running
    host = hpns.nic_name_to_host(args.nic_name)
    w = _HPOWorker(
        benchmark,
        run_id=args.run_id,
        host=host,
        logger=logging.getLogger("worker"),
    )
    w.load_nameserver_credentials(working_directory=str(working_directory))
    w.run(background=False)


def _run_master(args, benchmark, working_directory):
    nameserver = hpns.NameServer(
        run_id=args.run_id,
        working_directory=str(working_directory),
        nic_name=args.nic_name,
    )
    ns_host, ns_port = nameserver.start()

    # Start a background worker for the master node
    w = _HPOWorker(
        benchmark,
        run_id=args.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        logger=logging.getLogger("worker"),
    )
    w.run(background=True)

    # Create an optimizer
    optimizer = hydra.utils.instantiate(
        args.approach.approach,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        logger=logging.getLogger("master"),
    )

    # Train and evaluate the optimizer
    try:
        _train_and_eval(optimizer, benchmark, args)
    finally:
        optimizer.shutdown(shutdown_workers=True)
        nameserver.shutdown()


def _set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.manual_seed(seed)
    # tf.random.set_seed(seed)


@hydra.main(config_path="configs", config_name="run")
def run(args):
    _set_seeds(args.seed)
    working_directory = Path().cwd()

    # Log general information
    logger.info(f"Using working_directory={working_directory}")
    with contextlib.suppress(TypeError):
        git_info = gitinfo.get_git_info()
        logger.info(f"Commit hash: {git_info['commit']}")
        logger.info(f"Commit date: {git_info['author_date']}")
    logger.info(f"Arguments:\n{OmegaConf.to_yaml(args)}")

    # Construct benchmark
    if "data_path" in args.benchmark.benchmark:
        args.benchmark.benchmark.data_path = hydra.utils.to_absolute_path(
            args.benchmark.benchmark.data_path
        )
    benchmark = hydra.utils.instantiate(args.benchmark.benchmark)

    # Actually run
    if args.worker_id == 0:
        _run_master(args, benchmark, working_directory)
    else:
        _run_worker(args, benchmark, working_directory)
    logger.info(f"Run finished")


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
