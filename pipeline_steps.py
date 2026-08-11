def train_step(
    execution_backend="worker_cpu",
    dataset_id="",
    dataset_project="",
    dataset_name="",
    persistent_dataset_path="",
    manifest_task_id="",
    clearml_project_name="clearml-orchestration-demo",
    clearml_task_name="training",
    run_name="demo",
    model="ours",
    image_size=128,
    batch_size=1,
    workers=0,
    n_epochs=2,
    lr=0.001,
    factor=0.9,
    patience=5,
    input_model_id="",
):
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from clearml import InputModel, OutputModel, Task
    from pipeline_common import materialize_manifests, resolve_dataset_root

    task = Task.current_task()
    root, dataset_mode = resolve_dataset_root(dataset_id, persistent_dataset_path)
    work = Path(tempfile.gettempdir()) / "clearml-orchestration" / task.id
    manifests = materialize_manifests(manifest_task_id, root, work / "manifests")
    save_dir = work / "runs"
    save_dir.mkdir(parents=True, exist_ok=True)

    load_path = ""
    if input_model_id:
        load_path = str(InputModel(model_id=input_model_id).get_local_copy())

    if execution_backend == "alliance":
        # Preserve the user's original Slurm/NCCL/GPU training entry point.
        cmd = [
            sys.executable, "-u", "train_torch_test.py",
            "--run_name", str(run_name),
            "--save_dir", str(save_dir),
            "--paths_file", manifests["train"],
            "--image_size", str(image_size),
            "--batch_size", str(batch_size),
            "--workers", str(workers),
            "--model", str(model),
            "--lr", str(lr),
            "--n_epochs", str(n_epochs),
            "--factor", str(factor),
            "--patience", str(patience),
            "--clearml_project_name", str(clearml_project_name),
            "--clearml_task_name", str(clearml_task_name),
        ]
        if manifests.get("val"):
            cmd += ["--val_paths_file", manifests["val"]]
        # Original code only uses this for a metadata association. Persistent
        # path mode intentionally leaves it empty and skips the lookup.
        if dataset_name:
            cmd += ["--clearml_dataset_name", str(dataset_name)]
        if load_path:
            cmd += ["--load_path", load_path]
        subprocess.run(cmd, check=True)
        checkpoints = list(save_dir.glob("checkpoints/**/*_best_*.pth"))
        if not checkpoints:
            checkpoints = list(save_dir.glob("checkpoints/**/*_last_*.pth"))
        if not checkpoints:
            raise FileNotFoundError(f"Original trainer produced no checkpoint under {save_dir}")
        checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    elif execution_backend == "worker_cpu":
        cmd = [
            sys.executable, "-u", "train_worker_cpu.py",
            "--run_name", str(run_name),
            "--save_dir", str(save_dir),
            "--paths_file", manifests["train"],
            "--image_size", str(image_size),
            "--batch_size", str(batch_size),
            "--workers", str(workers),
            "--model", str(model),
            "--lr", str(lr),
            "--n_epochs", str(n_epochs),
            "--factor", str(factor),
            "--patience", str(patience),
        ]
        if manifests.get("val"):
            cmd += ["--val_paths_file", manifests["val"]]
        if load_path:
            cmd += ["--load_path", load_path]
        subprocess.run(cmd, check=True)
        result = json.loads((save_dir / "training-result.json").read_text(encoding="utf-8"))
        checkpoint = Path(result["checkpoint"])
    else:
        raise ValueError("execution_backend must be 'worker_cpu' or 'alliance'")

    output_model = OutputModel(task=task, name=f"{run_name}-model")
    output_model.update_weights(
        weights_filename=str(checkpoint),
        target_filename=checkpoint.name,
        auto_delete_file=False,
        async_enable=False,
    )
    result = {
        "training_task_id": task.id,
        "model_id": output_model.id,
        "dataset_mode": dataset_mode,
        "dataset_id": dataset_id,
        "dataset_project": dataset_project,
        "dataset_name": dataset_name,
        "execution_backend": execution_backend,
    }
    task.upload_artifact("training_result", artifact_object=result, wait_on_upload=True)
    return result


def evaluate_step(
    training,
    dataset_id="",
    persistent_dataset_path="",
    manifest_task_id="",
    model="ours",
    image_size=128,
    minimum_accuracy=0.0,
):
    import csv
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from clearml import InputModel, Task
    from pipeline_common import materialize_manifests, resolve_dataset_root

    task = Task.current_task()
    root, dataset_mode = resolve_dataset_root(dataset_id, persistent_dataset_path)
    work = Path(tempfile.gettempdir()) / "clearml-orchestration" / task.id
    manifests = materialize_manifests(manifest_task_id, root, work / "manifests")
    out_dir = work / "evaluation"
    model_path = InputModel(model_id=training["model_id"]).get_local_copy()

    subprocess.run(
        [
            sys.executable, "-u", "eval.py",
            "--iut_paths_file", manifests["test"],
            "--image_size", str(image_size),
            "--out_dir", str(out_dir),
            "--model", str(model),
            "--load_path", str(model_path),
        ],
        check=True,
    )

    pred_csv = out_dir / "pred.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(f"Evaluation did not produce {pred_csv}")
    total = 0
    correct = 0
    with pred_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            total += 1
            correct += str(row["Correct"]).strip().lower() == "true"
    if not total:
        raise RuntimeError("Evaluation produced no predictions")
    accuracy = correct / total
    accepted = accuracy >= float(minimum_accuracy)
    logger = task.get_logger()
    logger.report_single_value("accuracy", accuracy)
    logger.report_single_value("minimum_accuracy", float(minimum_accuracy))
    logger.report_single_value("accepted", int(accepted))
    result = {
        "evaluation_task_id": task.id,
        "training_task_id": training["training_task_id"],
        "model_id": training["model_id"],
        "accuracy": accuracy,
        "minimum_accuracy": float(minimum_accuracy),
        "accepted": accepted,
        "dataset_mode": dataset_mode,
    }
    task.upload_artifact("evaluation_result", artifact_object=result, wait_on_upload=True)
    return result
