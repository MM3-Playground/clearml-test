import argparse
import json
import subprocess
from pathlib import Path

from clearml import PipelineController, Task

from pipeline_steps import evaluate_step, train_step


PARAMETERS = {
    "execution_backend": ("worker_cpu", "worker_cpu for the CPU ClearML test worker; alliance for original Slurm trainer", "str"),
    "dataset_id": ("", "ClearML Dataset ID. Leave empty when using a persistent dataset path.", "str"),
    "dataset_project": ("", "Optional ClearML Dataset project metadata.", "str"),
    "dataset_name": ("", "Optional ClearML Dataset name metadata; used by the original Alliance trainer.", "str"),
    "persistent_dataset_path": ("", "Pre-downloaded dataset path such as /workspace/persistent-data/name.", "str"),
    "manifest_task_id": ("", "Task ID produced by upload_manifests.py.", "str"),
    "clearml_project_name": ("clearml-orchestration-demo", "Project for experiment/task tracking.", "str"),
    "clearml_task_name": ("training", "Training task display name used by the original trainer.", "str"),
    "run_name": ("demo", "Run/model display name.", "str"),
    "model": ("ours", "xception, cnndct, cnnpixel, or ours.", "str"),
    "image_size": (128, "Input image size.", "int"),
    "batch_size": (1, "Batch size.", "int"),
    "workers": (0, "DataLoader workers.", "int"),
    "n_epochs": (2, "Training epochs.", "int"),
    "lr": (0.001, "Learning rate.", "float"),
    "factor": (0.9, "Scheduler factor.", "float"),
    "patience": (5, "Scheduler patience.", "int"),
    "minimum_accuracy": (0.0, "Evaluation acceptance threshold.", "float"),
    "input_model_id": ("", "Optional ClearML model ID for retraining.", "str"),
}


def git_info():
    def git(*args):
        return subprocess.check_output(["git", *args], text=True).strip()

    repo = git("remote", "get-url", "origin")
    branch = git("branch", "--show-current")
    commit = git("rev-parse", "HEAD")
    if not repo or repo == ".":
        raise RuntimeError("Git origin must be a cloneable remote URL")
    dirty = subprocess.run(["git", "status", "--porcelain"], text=True, capture_output=True, check=True).stdout.strip()
    if dirty:
        print("WARNING: code changes are uncommitted. Remote tasks execute the recorded pushed commit; local JSON settings are still captured separately.")
    return repo, branch, commit


def load_settings(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pipeline(args, initial_settings=None):
    repo, branch, commit = git_info()
    pipe = PipelineController(
        name=args.name,
        project=args.project,
        version=args.version,
        abort_on_failure=True,
        add_pipeline_tags=True,
        docker=args.controller_docker,
        # Controller only needs ClearML; there is still a single repository requirements.txt.
        packages=["clearml==2.1.11"],
        repo=repo,
        repo_branch=branch or None,
        repo_commit=commit,
        always_create_from_code=False,
        skip_global_imports=True,
        working_dir=".",
    )

    # On the initial local submission this stores the user's current JSON contents
    # in ClearML. On remote/UI reruns {} is replaced by the stored configuration.
    settings = pipe.connect_configuration(initial_settings or {}, name="run_settings")
    for name, (default, description, param_type) in PARAMETERS.items():
        pipe.add_parameter(name=name, default=settings.get(name, default), description=description, param_type=param_type)

    pipe.set_default_execution_queue(args.execution_queue)
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

    pipe.add_function_step(
        name="train",
        task_name="pipeline-train",
        task_type="training",
        function=train_step,
        function_kwargs={
            "execution_backend": "${pipeline.execution_backend}",
            "dataset_id": "${pipeline.dataset_id}",
            "dataset_project": "${pipeline.dataset_project}",
            "dataset_name": "${pipeline.dataset_name}",
            "persistent_dataset_path": "${pipeline.persistent_dataset_path}",
            "manifest_task_id": "${pipeline.manifest_task_id}",
            "clearml_project_name": "${pipeline.clearml_project_name}",
            "clearml_task_name": "${pipeline.clearml_task_name}",
            "run_name": "${pipeline.run_name}",
            "model": "${pipeline.model}",
            "image_size": "${pipeline.image_size}",
            "batch_size": "${pipeline.batch_size}",
            "workers": "${pipeline.workers}",
            "n_epochs": "${pipeline.n_epochs}",
            "lr": "${pipeline.lr}",
            "factor": "${pipeline.factor}",
            "patience": "${pipeline.patience}",
            "input_model_id": "${pipeline.input_model_id}",
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
            "training": "${train.training}",
            "dataset_id": "${pipeline.dataset_id}",
            "persistent_dataset_path": "${pipeline.persistent_dataset_path}",
            "manifest_task_id": "${pipeline.manifest_task_id}",
            "model": "${pipeline.model}",
            "image_size": "${pipeline.image_size}",
            "minimum_accuracy": "${pipeline.minimum_accuracy}",
        },
        function_return=["evaluation"],
        parents=["train"],
        **step_common,
    )
    return pipe


def parse_args():
    p = argparse.ArgumentParser(description="ClearML training -> evaluation pipeline")
    p.add_argument("--config", default="configs/run.local.json")
    p.add_argument("--mode", choices=["remote", "local"], default="remote")
    p.add_argument("--project", default="ClearML Pipelines")
    p.add_argument("--name", default="clearml-training-pipeline")
    p.add_argument("--version", default="1.0.0")
    p.add_argument("--controller-queue", default="services")
    p.add_argument("--execution-queue", default="default")
    p.add_argument("--controller-docker", default="python:3.11")
    p.add_argument("--task-docker", default="python:3.11")
    return p.parse_args()


def main():
    args = parse_args()
    # Crucial: remote controller re-execution never opens the repo copy of the JSON.
    initial_settings = load_settings(args.config) if Task.running_locally() else None
    pipe = build_pipeline(args, initial_settings)
    if args.mode == "local":
        pipe.start_locally(run_pipeline_steps_locally=True)
    else:
        # Submit controller to services and return immediately. Training/evaluation
        # dependency installation and execution happen entirely on remote agents.
        pipe.start(queue=args.controller_queue, wait=False)
        print(f"Submitted pipeline controller: {pipe.id}")


if __name__ == "__main__":
    main()
