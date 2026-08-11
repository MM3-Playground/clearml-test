
import json
import os
import subprocess
import tempfile
from pathlib import Path

from clearml import Dataset, Task
from clearml.automation.controller import PipelineDecorator

DEFAULT_QUEUE = os.getenv("CLEARML_PIPELINE_QUEUE", "default")
DEFAULT_DOCKER = os.getenv("CLEARML_TASK_DOCKER", "python:3.11")
PIPELINE_PROJECT = os.getenv("CLEARML_PIPELINE_PROJECT", "ClearML Pipelines")
PIPELINE_NAME = os.getenv("CLEARML_PIPELINE_NAME", "clearml-training-pipeline")


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return ""


def _code_repository() -> str:
    return os.getenv("CLEARML_CODE_REPO", "").strip() or _git_output(
        "remote", "get-url", "origin"
    )


def _code_commit() -> str:
    return os.getenv("CLEARML_CODE_COMMIT", "").strip() or _git_output(
        "rev-parse", "HEAD"
    )


def _code_branch() -> str:
    return os.getenv("CLEARML_CODE_BRANCH", "").strip() or _git_output(
        "branch", "--show-current"
    )


CODE_REPO = _code_repository()
CODE_COMMIT = _code_commit()
CODE_BRANCH = _code_branch()


def _require_remote_git() -> None:
    if not CODE_REPO:
        raise RuntimeError(
            "Remote mode requires a reachable Git repository. Run from a Git checkout "
            "with an origin remote, or set CLEARML_CODE_REPO explicitly."
        )
    if not CODE_COMMIT:
        raise RuntimeError(
            "Remote mode requires an exact Git commit. Commit the code first, or set "
            "CLEARML_CODE_COMMIT explicitly."
        )


def _read_manifest(path: str | Path):
    rows = []
    for n, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.rstrip().split("\t")
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest line {n}: expected PATH<TAB>LABEL")
        rows.append((parts[0], int(parts[1])))
    return rows


def _portable_manifest(source, destination, dataset_root=""):
    root = Path(dataset_root).expanduser().resolve() if dataset_root else None
    dest = Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for raw_path, label in _read_manifest(source):
            path = Path(raw_path).expanduser()
            rendered = path.as_posix()
            if root is not None and path.is_absolute():
                rendered = path.resolve().relative_to(root).as_posix()
            handle.write(f"{rendered}\t{label}\n")
    return dest


def upload_manifest_bundle(config):
    task = Task.init(
        project_name=config["clearml_project_name"],
        task_name=config.get("task_name", "manifest-bundle"),
        task_type=Task.TaskTypes.data_processing,
        output_uri=True,
    )
    root = str(config.get("dataset_root", "") or "")
    with tempfile.TemporaryDirectory(prefix="clearml-manifests-") as tmp:
        for key, name in [
            ("train_paths_file", "train.txt"),
            ("val_paths_file", "val.txt"),
            ("test_paths_file", "test.txt"),
        ]:
            portable = _portable_manifest(config[key], Path(tmp) / name, root)
            task.upload_artifact(
                f"manifest_{key}",
                artifact_object=str(portable),
                wait_on_upload=True,
            )
    task.close()
    return task.id



@PipelineDecorator.component(
    name="train",
    return_values=["training"],
    cache=False,
    execution_queue=DEFAULT_QUEUE,
    docker=DEFAULT_DOCKER,
    repo=CODE_REPO or None,
    repo_commit=CODE_COMMIT or None,
    packages=False,
)
def train_component(
    dataset_id: str,
    persistent_dataset_path: str,
    manifest_task_id: str,
    clearml_project_name: str,
    clearml_task_name: str,
    run_name: str,
    model: str,
    image_size: int,
    batch_size: int,
    workers: int,
    n_epochs: int,
    lr: float,
    factor: float,
    patience: int,
):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from clearml import Task
    from pipeline_helpers import materialize_manifests, resolve_dataset_root

    task = Task.current_task()
    root, mode = resolve_dataset_root(dataset_id, persistent_dataset_path)
    task.set_parameter("dataset/mode", mode)
    task.set_parameter("dataset/id", dataset_id)
    task.set_parameter("dataset/persistent_path", persistent_dataset_path)
    work = Path(os.getenv("TMPDIR", "/tmp")) / "clearml-pipeline" / task.id
    manifests = materialize_manifests(manifest_task_id, root, work / "manifests")
    save = work / "runs"
    cmd = [
        sys.executable,
        "-u",
        "train_torch_test.py",
        "--run_name",
        run_name,
        "--save_dir",
        str(save),
        "--paths_file",
        manifests["train"],
        "--val_paths_file",
        manifests["val"],
        "--image_size",
        str(image_size),
        "--batch_size",
        str(batch_size),
        "--workers",
        str(workers),
        "--model",
        model,
        "--lr",
        str(lr),
        "--n_epochs",
        str(n_epochs),
        "--factor",
        str(factor),
        "--patience",
        str(patience),
        "--clearml_project_name",
        clearml_project_name,
        "--clearml_task_name",
        clearml_task_name,
    ]
    if dataset_id:
        cmd += ["--clearml_dataset_id", dataset_id]
    subprocess.run(cmd, check=True)
    result = json.loads((save / "training-result.json").read_text())
    result["dataset_mode"] = mode
    return result


@PipelineDecorator.component(
    name="evaluate",
    return_values=["evaluation"],
    cache=False,
    execution_queue=DEFAULT_QUEUE,
    docker=DEFAULT_DOCKER,
    repo=CODE_REPO or None,
    repo_commit=CODE_COMMIT or None,
    packages=False,
)
def evaluate_component(
    dataset_id: str,
    persistent_dataset_path: str,
    manifest_task_id: str,
    training: dict,
    clearml_project_name: str,
    clearml_task_name: str,
    model: str,
    image_size: int,
    minimum_accuracy: float,
):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from clearml import InputModel, Task
    from pipeline_helpers import materialize_manifests, resolve_dataset_root

    task = Task.current_task()
    root, mode = resolve_dataset_root(dataset_id, persistent_dataset_path)
    work = Path(os.getenv("TMPDIR", "/tmp")) / "clearml-pipeline" / task.id
    manifests = materialize_manifests(manifest_task_id, root, work / "manifests")
    model_path = InputModel(model_id=training["model_id"]).get_local_copy()
    out = work / "evaluation"
    subprocess.run(
        [
            sys.executable,
            "-u",
            "eval.py",
            "--iut_paths_file",
            manifests["test"],
            "--image_size",
            str(image_size),
            "--out_dir",
            str(out),
            "--model",
            model,
            "--load_path",
            str(model_path),
            "--device",
            "cpu",
            "--clearml_project_name",
            clearml_project_name,
            "--clearml_task_name",
            f"{clearml_task_name}-evaluation",
            "--parent_training_task_id",
            str(training["training_task_id"]),
        ],
        check=True,
    )
    result = json.loads((out / "result.json").read_text())
    result["minimum_accuracy"] = float(minimum_accuracy)
    result["accepted"] = float(result["accuracy"]) >= float(minimum_accuracy)
    result["dataset_mode"] = mode
    task.get_logger().report_single_value("accepted", int(result["accepted"]))
    return result


@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=PIPELINE_PROJECT,
    version="5.1.0",
    default_queue=DEFAULT_QUEUE,
    pipeline_execution_queue=DEFAULT_QUEUE,
    abort_on_failure=True,
    docker=DEFAULT_DOCKER,
    repo=CODE_REPO or None,
    repo_commit=CODE_COMMIT or None,
    packages=False,
    # Initial bootstrap runs only the lightweight controller locally;
    # train/evaluate components are still submitted to ClearML queues.
    start_controller_locally=True,
)
def training_pipeline(
    dataset_id: str = "",
    persistent_dataset_path: str = "",
    manifest_task_id: str = "",
    clearml_project_name: str = "clearml-orchestration-demo",
    clearml_task_name: str = "cpu-demo",
    run_name: str = "cpu-demo",
    model: str = "ours",
    image_size: int = 128,
    batch_size: int = 1,
    workers: int = 0,
    n_epochs: int = 2,
    lr: float = 0.001,
    factor: float = 0.9,
    patience: int = 5,
    minimum_accuracy: float = 0.0,
):
    training = train_component(
        dataset_id,
        persistent_dataset_path,
        manifest_task_id,
        clearml_project_name,
        clearml_task_name,
        run_name,
        model,
        image_size,
        batch_size,
        workers,
        n_epochs,
        lr,
        factor,
        patience,
    )
    return evaluate_component(
        dataset_id,
        persistent_dataset_path,
        manifest_task_id,
        training,
        clearml_project_name,
        clearml_task_name,
        model,
        image_size,
        minimum_accuracy,
    )
