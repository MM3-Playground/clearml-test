import argparse
import json
import subprocess
from pathlib import Path

from clearml import PipelineController, Task

from pipeline_steps import evaluate_step, train_step


PARAMETERS = {
    "execution_backend": (
        "worker_cpu",
        "worker_cpu for the CPU ClearML test worker; alliance for original Slurm trainer",
        "str",
    ),
    "dataset_id": (
        "",
        "ClearML Dataset ID. Leave empty when using a persistent dataset path.",
        "str",
    ),
    "dataset_project": (
        "",
        "Optional ClearML Dataset project metadata.",
        "str",
    ),
    "dataset_name": (
        "",
        "Optional ClearML Dataset name metadata; used by the original Alliance trainer.",
        "str",
    ),
    "persistent_dataset_path": (
        "",
        "Pre-downloaded dataset path such as /workspace/persistent-data/name.",
        "str",
    ),
    "manifest_task_id": (
        "",
        "Task ID produced by upload_manifests.py.",
        "str",
    ),
    "clearml_project_name": (
        "clearml-orchestration-demo",
        "Project for experiment/task tracking.",
        "str",
    ),
    "clearml_task_name": (
        "training",
        "Training task display name used by the original trainer.",
        "str",
    ),
    "run_name": (
        "demo",
        "Run/model display name.",
        "str",
    ),
    "model": (
        "ours",
        "xception, cnndct, cnnpixel, or ours.",
        "str",
    ),
    "image_size": (
        128,
        "Input image size.",
        "int",
    ),
    "batch_size": (
        1,
        "Batch size.",
        "int",
    ),
    "workers": (
        0,
        "DataLoader workers.",
        "int",
    ),
    "n_epochs": (
        2,
        "Training epochs.",
        "int",
    ),
    "lr": (
        0.001,
        "Learning rate.",
        "float",
    ),
    "factor": (
        0.9,
        "Scheduler factor.",
        "float",
    ),
    "patience": (
        5,
        "Scheduler patience.",
        "int",
    ),
    "minimum_accuracy": (
        0.0,
        "Evaluation acceptance threshold.",
        "float",
    ),
    "input_model_id": (
        "",
        "Optional ClearML model ID for retraining.",
        "str",
    ),
}


def git_info():
    def git(*args):
        return subprocess.check_output(
            ["git", *args],
            text=True,
        ).strip()

    repo = git("remote", "get-url", "origin")
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")

    if not repo or repo == ".":
        raise RuntimeError(
            "Git origin must be a cloneable remote URL"
        )

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    if dirty:
        print(
            "WARNING: code changes are uncommitted. "
            "Remote tasks execute the recorded pushed commit; "
            "local JSON settings are captured separately."
        )

    return repo, branch, commit


def load_settings(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def _get_remote_pipeline_parameters():
    """
    Read the effective pipeline parameters from the currently executing
    ClearML controller Task.

    This is used when the controller itself is running remotely.

    ClearML stores PipelineController.add_parameter() values under the
    pipeline hyperparameter section. Values changed through '+ NEW RUN'
    are already applied to this Task before the controller script runs.
    """
    task = Task.current_task()

    if task is None:
        return {}

    parameters = task.get_parameters(cast=True) or {}

    result = {}

    for name in PARAMETERS:
        # ClearML normally stores PipelineController parameters under
        # "pipeline/<name>". Support the capitalized form defensively too.
        for key in (
            f"pipeline/{name}",
            f"Pipeline/{name}",
        ):
            if key in parameters:
                result[name] = parameters[key]
                break

    return result


def _resolve_settings(initial_settings):
    """
    Determine the settings that should be used while constructing the DAG.

    Initial submission:
        values come from the user's local JSON.

    Remote controller / UI '+ NEW RUN':
        values come from the controller Task's effective pipeline parameters.

    The JSON file therefore never needs to be opened on the worker.
    """
    settings = {
        name: default
        for name, (default, _description, _type) in PARAMETERS.items()
    }

    if initial_settings is not None:
        settings.update(initial_settings)
        return settings

    # We are executing the controller remotely.
    remote_settings = _get_remote_pipeline_parameters()

    if not remote_settings:
        raise RuntimeError(
            "The remote pipeline controller did not contain any "
            "pipeline parameters."
        )

    settings.update(remote_settings)
    return settings


def build_pipeline(args, initial_settings=None):
    repo, branch, commit = git_info()

    pipe = PipelineController(
        name=args.name,
        project=args.project,
        version=args.version,
        abort_on_failure=True,
        add_pipeline_tags=True,
        docker=args.controller_docker,

        # Controller itself only needs ClearML.
        packages=["clearml==2.1.11"],

        repo=repo,
        repo_branch=branch or None,
        repo_commit=commit,

        # Important:
        # Function-based pipelines are rebuilt from code when remotely
        # executed / started through '+ NEW RUN'. This lets us read the
        # effective controller parameters and inject them into the steps.
        always_create_from_code=True,

        skip_global_imports=True,
        working_dir=".",
    )

    settings = _resolve_settings(initial_settings)

    # Store the original locally supplied JSON as a ClearML configuration
    # object. Do this only during initial submission. A remotely executing
    # controller already has its recorded configuration.
    if initial_settings is not None:
        pipe.connect_configuration(
            initial_settings,
            name="run_settings",
        )

    # Expose all researcher-facing values as pipeline parameters.
    #
    # On initial submission, defaults come from the local JSON.
    # On remote/UI execution, defaults are the effective values already
    # present on the controller Task.
    for name, (default, description, param_type) in PARAMETERS.items():
        pipe.add_parameter(
            name=name,
            default=settings.get(name, default),
            description=description,
            param_type=param_type,
        )

    pipe.set_default_execution_queue(
        args.execution_queue
    )

    step_common = dict(
        project_name=args.project,
        packages=False,
        repo=repo,
        repo_branch=branch or None,
        repo_commit=commit,
        docker=args.task_docker,
        execution_queue=args.execution_queue,
        cache_executed_step=False,
        working_dir=".",
        output_uri=True,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------
    #
    # Use the values already resolved from the controller Task instead of
    # asking ClearML to perform another ${pipeline.x} interpolation when
    # creating the standalone function Task.
    #
    # This is the important change for the dataset_id problem.
    #
    pipe.add_function_step(
        name="train",
        task_name="pipeline-train",
        task_type="training",
        function=train_step,
        function_kwargs={
            "execution_backend":
                settings["execution_backend"],

            "dataset_id":
                settings["dataset_id"],

            "dataset_project":
                settings["dataset_project"],

            "dataset_name":
                settings["dataset_name"],

            "persistent_dataset_path":
                settings["persistent_dataset_path"],

            "manifest_task_id":
                settings["manifest_task_id"],

            "clearml_project_name":
                settings["clearml_project_name"],

            "clearml_task_name":
                settings["clearml_task_name"],

            "run_name":
                settings["run_name"],

            "model":
                settings["model"],

            "image_size":
                settings["image_size"],

            "batch_size":
                settings["batch_size"],

            "workers":
                settings["workers"],

            "n_epochs":
                settings["n_epochs"],

            "lr":
                settings["lr"],

            "factor":
                settings["factor"],

            "patience":
                settings["patience"],

            "input_model_id":
                settings["input_model_id"],
        },
        function_return=["training"],
        **step_common,
    )

    # ------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------
    #
    # The training output still needs ClearML pipeline interpolation because
    # it does not exist until the training step finishes.
    #
    pipe.add_function_step(
        name="evaluate",
        task_name="pipeline-evaluate",
        task_type="testing",
        function=evaluate_step,
        function_kwargs={
            "training":
                "${train.training}",

            "dataset_id":
                settings["dataset_id"],

            "persistent_dataset_path":
                settings["persistent_dataset_path"],

            "manifest_task_id":
                settings["manifest_task_id"],

            "model":
                settings["model"],

            "image_size":
                settings["image_size"],

            "minimum_accuracy":
                settings["minimum_accuracy"],
        },
        function_return=["evaluation"],
        parents=["train"],
        **step_common,
    )

    return pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description="ClearML training -> evaluation pipeline"
    )

    parser.add_argument(
        "--config",
        default="configs/run.local.json",
    )

    parser.add_argument(
        "--mode",
        choices=["remote", "local"],
        default="remote",
    )

    parser.add_argument(
        "--project",
        default="ClearML Pipelines",
    )

    parser.add_argument(
        "--name",
        default="clearml-training-pipeline",
    )

    parser.add_argument(
        "--version",
        default="1.1.0",
    )

    parser.add_argument(
        "--controller-queue",
        default="services",
    )

    parser.add_argument(
        "--execution-queue",
        default="default",
    )

    parser.add_argument(
        "--controller-docker",
        default="python:3.11",
    )

    parser.add_argument(
        "--task-docker",
        default="python:3.11",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Only the initial/local process reads the JSON file.
    #
    # When ClearML re-executes this controller remotely, the original
    # command line can still contain --config, but the worker does not open
    # that file. Instead, _resolve_settings() retrieves the effective values
    # from the controller Task.
    if Task.running_locally():
        initial_settings = load_settings(
            args.config
        )
    else:
        initial_settings = None

    pipe = build_pipeline(
        args,
        initial_settings,
    )

    if args.mode == "local":
        # Alliance/local execution:
        # controller + steps stay on the current machine/allocation.
        pipe.start_locally(
            run_pipeline_steps_locally=True
        )

    else:
        # Remote ClearML execution:
        # submit controller to the services queue and immediately return
        # from the user's local process.
        pipe.start(
            queue=args.controller_queue,
            wait=False,
        )

        print(
            f"Submitted pipeline controller: {pipe.id}"
        )


if __name__ == "__main__":
    main()