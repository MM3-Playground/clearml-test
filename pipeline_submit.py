import argparse
import json
import subprocess
from pathlib import Path

from clearml import PipelineController, Task

from pipeline_steps import evaluate_step, train_step


PARAMETERS = {
    "execution_backend": (
        "worker_cpu",
        "worker_cpu or alliance",
        "str",
    ),
    "dataset_id": (
        "",
        "ClearML Dataset ID",
        "str",
    ),
    "dataset_project": (
        "",
        "ClearML Dataset project",
        "str",
    ),
    "dataset_name": (
        "",
        "ClearML Dataset name",
        "str",
    ),
    "persistent_dataset_path": (
        "",
        "Pre-downloaded persistent dataset path",
        "str",
    ),
    "manifest_task_id": (
        "",
        "Manifest bundle Task ID",
        "str",
    ),
    "clearml_project_name": (
        "clearml-orchestration-demo",
        "Experiment project",
        "str",
    ),
    "clearml_task_name": (
        "cpu-demo",
        "Training task name",
        "str",
    ),
    "run_name": (
        "cpu-demo",
        "Run name",
        "str",
    ),
    "model": (
        "ours",
        "Model type",
        "str",
    ),
    "image_size": (
        128,
        "Image size",
        "int",
    ),
    "batch_size": (
        1,
        "Batch size",
        "int",
    ),
    "workers": (
        0,
        "DataLoader workers",
        "int",
    ),
    "n_epochs": (
        2,
        "Training epochs",
        "int",
    ),
    "lr": (
        0.001,
        "Learning rate",
        "float",
    ),
    "factor": (
        0.9,
        "Scheduler factor",
        "float",
    ),
    "patience": (
        5,
        "Scheduler patience",
        "int",
    ),
    "minimum_accuracy": (
        0.0,
        "Acceptance threshold",
        "float",
    ),
    "input_model_id": (
        "",
        "Optional input model ID for retraining",
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

    return repo, branch, commit


def load_settings(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def build_pipeline(args, settings):
    repo, branch, commit = git_info()

    pipe = PipelineController(
        name=args.name,
        project=args.project,
        version=args.version,
        abort_on_failure=True,
        add_pipeline_tags=True,

        repo=repo,
        repo_branch=branch or None,
        repo_commit=commit,

        docker=args.controller_docker,
        packages=["clearml==2.1.11"],

        # CRITICAL:
        # Store the DAG/configuration built here.
        # Remote/UI runs use this captured definition instead of
        # executing this Python code again to rebuild the DAG.
        always_create_from_code=False,

        working_dir=".",
    )

    # Store the settings dictionary as a ClearML configuration as well.
    settings = pipe.connect_configuration(
        settings,
        name="run_settings",
    )

    # Pipeline parameters get their initial values from the LOCAL JSON.
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

    step_common = {
        "project_name": args.project,
        "packages": False,
        "repo": repo,
        "repo_branch": branch or None,
        "repo_commit": commit,
        "docker": args.task_docker,
        "execution_queue": args.execution_queue,
        "cache_executed_step": False,
        "working_dir": ".",
        "output_uri": True,
    }

    pipe.add_function_step(
        name="train",
        task_name="pipeline-train",
        task_type="training",
        function=train_step,
        function_kwargs={
            "execution_backend":
                "${pipeline.execution_backend}",

            "dataset_id":
                "${pipeline.dataset_id}",

            "dataset_project":
                "${pipeline.dataset_project}",

            "dataset_name":
                "${pipeline.dataset_name}",

            "persistent_dataset_path":
                "${pipeline.persistent_dataset_path}",

            "manifest_task_id":
                "${pipeline.manifest_task_id}",

            "clearml_project_name":
                "${pipeline.clearml_project_name}",

            "clearml_task_name":
                "${pipeline.clearml_task_name}",

            "run_name":
                "${pipeline.run_name}",

            "model":
                "${pipeline.model}",

            "image_size":
                "${pipeline.image_size}",

            "batch_size":
                "${pipeline.batch_size}",

            "workers":
                "${pipeline.workers}",

            "n_epochs":
                "${pipeline.n_epochs}",

            "lr":
                "${pipeline.lr}",

            "factor":
                "${pipeline.factor}",

            "patience":
                "${pipeline.patience}",

            "input_model_id":
                "${pipeline.input_model_id}",
        },
        function_return=["training"],
        **step_common,
    )

    pipe.add_function_step(
        name="evaluate",
        task_name="pipeline-evaluate",
        task_type="testing",
        function=evaluate_step,
        function_kwargs={
            "training":
                "${train.training}",

            "dataset_id":
                "${pipeline.dataset_id}",

            "persistent_dataset_path":
                "${pipeline.persistent_dataset_path}",

            "manifest_task_id":
                "${pipeline.manifest_task_id}",

            "model":
                "${pipeline.model}",

            "image_size":
                "${pipeline.image_size}",

            "minimum_accuracy":
                "${pipeline.minimum_accuracy}",
        },
        function_return=["evaluation"],
        parents=["train"],
        **step_common,
    )

    return pipe


def parse_args():
    parser = argparse.ArgumentParser()

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
        default="1.2.0",
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

    # Initial pipeline definition is created on the user's machine.
    #
    # This is where the JSON is consumed.
    #
    # Because always_create_from_code=False, ClearML stores the resulting
    # DAG and parameters. Remote/UI executions use the stored definition.
    settings = load_settings(args.config)

    pipe = build_pipeline(
        args,
        settings,
    )

    if args.mode == "local":
        pipe.start_locally(
            run_pipeline_steps_locally=True
        )

    else:
        pipe.start(
            queue=args.controller_queue,
            wait=False,
        )

        print(
            f"Submitted pipeline controller: {pipe.id}"
        )


if __name__ == "__main__":
    main()