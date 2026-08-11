import argparse
import json
import subprocess
from pathlib import Path

from clearml import PipelineController
from clearml import Task

from pipeline_steps import evaluate_step, train_step


PARAMETERS = {
    "execution_backend": (
        "worker_cpu",
        "worker_cpu for the CPU ClearML worker; alliance for the Slurm trainer",
        "str",
    ),
    "dataset_id": (
        "",
        "ClearML Dataset ID. Leave empty when using persistent_dataset_path.",
        "str",
    ),
    "dataset_project": (
        "",
        "Optional ClearML Dataset project.",
        "str",
    ),
    "dataset_name": (
        "",
        "Optional ClearML Dataset name.",
        "str",
    ),
    "persistent_dataset_path": (
        "",
        "Pre-downloaded dataset path, e.g. /workspace/persistent-data/name.",
        "str",
    ),
    "manifest_task_id": (
        "",
        "Task ID containing the uploaded train/val/test manifests.",
        "str",
    ),
    "clearml_project_name": (
        "clearml-orchestration-demo",
        "ClearML project used by the training/evaluation code.",
        "str",
    ),
    "clearml_task_name": (
        "cpu-demo",
        "Training task name.",
        "str",
    ),
    "run_name": (
        "cpu-demo",
        "Training run name.",
        "str",
    ),
    "model": (
        "ours",
        "Model type.",
        "str",
    ),
    "image_size": (
        128,
        "Input image size.",
        "int",
    ),
    "batch_size": (
        1,
        "Training batch size.",
        "int",
    ),
    "workers": (
        0,
        "DataLoader workers.",
        "int",
    ),
    "n_epochs": (
        2,
        "Number of training epochs.",
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
        "Minimum evaluation accuracy.",
        "float",
    ),
    "input_model_id": (
        "",
        "Optional ClearML model ID used for retraining.",
        "str",
    ),
}


TRAIN_PIPELINE_PARAMETERS = (
    "execution_backend",
    "dataset_id",
    "dataset_project",
    "dataset_name",
    "persistent_dataset_path",
    "manifest_task_id",
    "clearml_project_name",
    "clearml_task_name",
    "run_name",
    "model",
    "image_size",
    "batch_size",
    "workers",
    "n_epochs",
    "lr",
    "factor",
    "patience",
    "input_model_id",
)

EVALUATE_PIPELINE_PARAMETERS = (
    "dataset_id",
    "persistent_dataset_path",
    "manifest_task_id",
    "model",
    "image_size",
    "minimum_accuracy",
)


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
            "WARNING: repository contains uncommitted changes. "
            "Remote ClearML tasks execute the recorded Git commit, "
            "not uncommitted source-code changes."
        )

    return repo, branch, commit


def normalize_config_path(path):
    """
    Use forward slashes so a path initially supplied on Windows is also
    meaningful when the controller is recreated in a Linux container.

    connect_configuration() is called before the file is read, so ClearML
    can restore the captured local configuration contents remotely.
    """
    return Path(str(path).replace("\\", "/"))


def load_settings(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def _get_pipeline_value(pipeline, name):
    """
    PipelineController.get_parameters() is the source of truth here.

    Support a few possible key renderings defensively.
    """
    params = pipeline.get_parameters() or {}

    for key in (
        name,
        f"pipeline/{name}",
        f"Pipeline/{name}",
    ):
        if key in params:
            return params[key]

    raise KeyError(
        f"Pipeline parameter {name!r} was not found. "
        f"Available parameters: {params}"
    )


def inject_pipeline_parameters(pipeline, node, parameters):
    """
    Override the generated step Task arguments before the job is created.

    In ClearML 2.1.11, node.job can still be None here, so modify the
    parsed parameters dictionary directly instead.
    """

    if node.name == "train":
        names = TRAIN_PIPELINE_PARAMETERS
    elif node.name == "evaluate":
        names = EVALUATE_PIPELINE_PARAMETERS
    else:
        return True

    pipeline_params = pipeline.get_parameters() or {}

    def get_value(name):
        for key in (
            name,
            f"pipeline/{name}",
            f"Pipeline/{name}",
        ):
            if key in pipeline_params:
                return pipeline_params[key]

        raise KeyError(
            f"Pipeline parameter {name!r} not found. "
            f"Available parameters: {pipeline_params}"
        )

    print(f"[pipeline] preparing step {node.name!r}")
    print(f"[pipeline] parameters before override: {parameters!r}")

    for name in names:
        value = get_value(name)

        print(
            f"[pipeline] {node.name}: "
            f"Args/{name} = {value!r}"
        )

        # These are the hyperparameters that will be used to create
        # the standalone function Task.
        parameters[f"Args/{name}"] = value

    if node.name == "train":
        dataset_id = str(
            get_value("dataset_id") or ""
        ).strip()

        persistent_path = str(
            get_value("persistent_dataset_path") or ""
        ).strip()

        if not dataset_id and not persistent_path:
            raise ValueError(
                "Pipeline has neither dataset_id nor "
                "persistent_dataset_path"
            )

    print(f"[pipeline] parameters after override: {parameters!r}")

    return True


def build_pipeline(args):
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

        # Controller only needs ClearML itself.
        packages=["clearml==2.1.11"],

        # Rebuild from the recorded source code when a controller is
        # launched remotely / through + NEW RUN. The connected config
        # below restores the initial local JSON contents.
        always_create_from_code=True,

        working_dir=".",
    )

    # IMPORTANT:
    # connect_configuration() must happen before reading the file.
    #
    # On the original machine this captures the local JSON contents.
    # On remote execution ClearML restores the captured contents.
    config_path = normalize_config_path(
        args.config
    )

    connected_config_path = pipe.connect_configuration(
        configuration=str(config_path),
        name="run_settings",
        description="Runtime settings used to construct the pipeline",
    )

    settings = load_settings(
        connected_config_path
    )

    # Define actual ClearML pipeline parameters.
    #
    # These values appear under the controller Task's Pipeline parameters
    # and can be changed from + NEW RUN.
    for name, (
        default,
        description,
        param_type,
    ) in PARAMETERS.items():
        pipe.add_parameter(
            name=name,
            default=settings.get(
                name,
                default,
            ),
            description=description,
            param_type=param_type,
        )

    pipe.set_default_execution_queue(
        args.execution_queue
    )

    step_common = {
        "project_name": args.project,

        # Use the repository requirements.txt for training/evaluation.
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

    # ------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------

    pipe.add_function_step(
        name="train",
        task_name="pipeline-train",
        task_type="training",
        function=train_step,

        # Keep normal ClearML pipeline references in the DAG.
        #
        # pre_execute_callback additionally forces the effective values
        # into the generated Task immediately before enqueue.
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

        function_return=[
            "training",
        ],

        pre_execute_callback=inject_pipeline_parameters,

        **step_common,
    )

    # ------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------

    pipe.add_function_step(
        name="evaluate",
        task_name="pipeline-evaluate",
        task_type="testing",
        function=evaluate_step,

        function_kwargs={
            # This remains a normal step-output reference because its value
            # does not exist until train has completed.
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

        function_return=[
            "evaluation",
        ],

        parents=[
            "train",
        ],

        pre_execute_callback=inject_pipeline_parameters,

        **step_common,
    )

    return pipe


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Submit or locally execute the ClearML "
            "training -> evaluation pipeline"
        )
    )

    parser.add_argument(
        "--config",
        default="configs/run.example.json",
        help=(
            "Local runtime settings JSON. "
            "Its contents are captured by ClearML."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "remote",
            "local",
        ],
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
        default="1.3.0",
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

    pipe = build_pipeline(args)

    if args.mode == "local":
        pipe.start_locally(
            run_pipeline_steps_locally=True
        )
        return

    # On the user's machine:
    # submit the controller and return immediately.
    #
    # On the services agent:
    # keep the controller alive until the entire pipeline finishes.
    running_locally = Task.running_locally()

    pipe.start(
        queue=args.controller_queue,
        wait=not running_locally,
    )

    if running_locally:
        print(
            f"Submitted pipeline controller: {pipe.id}"
        )


if __name__ == "__main__":
    main()