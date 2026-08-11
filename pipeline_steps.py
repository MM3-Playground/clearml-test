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
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from clearml import InputModel, OutputModel, Task
    from pipeline_common import materialize_manifests, resolve_dataset_root

    task = Task.current_task()
    if task is None:
        raise RuntimeError("train_step is not running inside a ClearML Task")

    # ClearML function-step wrapper kwargs can contain the original defaults
    # even when the generated Task's Args/* parameters have been overridden
    # correctly by the pipeline controller.
    #
    # Treat the actual Task parameters as the source of truth.
    task_parameters = task.get_parameters(cast=True) or {}

    effective = {
        "execution_backend": execution_backend,
        "dataset_id": dataset_id,
        "dataset_project": dataset_project,
        "dataset_name": dataset_name,
        "persistent_dataset_path": persistent_dataset_path,
        "manifest_task_id": manifest_task_id,
        "clearml_project_name": clearml_project_name,
        "clearml_task_name": clearml_task_name,
        "run_name": run_name,
        "model": model,
        "image_size": image_size,
        "batch_size": batch_size,
        "workers": workers,
        "n_epochs": n_epochs,
        "lr": lr,
        "factor": factor,
        "patience": patience,
        "input_model_id": input_model_id,
    }

    for name in tuple(effective):
        key = f"Args/{name}"
        if key in task_parameters:
            effective[name] = task_parameters[key]

    execution_backend = str(effective["execution_backend"])
    dataset_id = str(effective["dataset_id"] or "")
    dataset_project = str(effective["dataset_project"] or "")
    dataset_name = str(effective["dataset_name"] or "")
    persistent_dataset_path = str(
        effective["persistent_dataset_path"] or ""
    )
    manifest_task_id = str(effective["manifest_task_id"] or "")
    clearml_project_name = str(effective["clearml_project_name"])
    clearml_task_name = str(effective["clearml_task_name"])
    run_name = str(effective["run_name"])
    model = str(effective["model"])

    image_size = int(effective["image_size"])
    batch_size = int(effective["batch_size"])
    workers = int(effective["workers"])
    n_epochs = int(effective["n_epochs"])
    lr = float(effective["lr"])
    factor = float(effective["factor"])
    patience = int(effective["patience"])

    input_model_id = str(effective["input_model_id"] or "")

    print(
        "[train_step] effective parameters:\n"
        f"  execution_backend={execution_backend!r}\n"
        f"  dataset_id={dataset_id!r}\n"
        f"  dataset_project={dataset_project!r}\n"
        f"  dataset_name={dataset_name!r}\n"
        f"  persistent_dataset_path={persistent_dataset_path!r}\n"
        f"  manifest_task_id={manifest_task_id!r}\n"
        f"  run_name={run_name!r}\n"
        f"  model={model!r}\n"
        f"  image_size={image_size!r}\n"
        f"  batch_size={batch_size!r}\n"
        f"  workers={workers!r}\n"
        f"  n_epochs={n_epochs!r}\n"
        f"  lr={lr!r}"
    )

    if not dataset_id and not persistent_dataset_path:
        raise ValueError(
            "Neither dataset_id nor persistent_dataset_path is set "
            "on the ClearML training Task"
        )

    root, dataset_mode = resolve_dataset_root(
        dataset_id,
        persistent_dataset_path,
    )

    work = (
        Path(tempfile.gettempdir())
        / "clearml-orchestration"
        / task.id
    )

    manifests = materialize_manifests(
        manifest_task_id,
        root,
        work / "manifests",
    )

    save_dir = work / "runs"
    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_path = ""

    if input_model_id:
        load_path = str(
            InputModel(
                model_id=input_model_id
            ).get_local_copy()
        )

    if execution_backend == "alliance":
        # Preserve the original Slurm/NCCL/GPU training entry point.
        cmd = [
            sys.executable,
            "-u",
            "train_torch_test.py",
            "--run_name",
            run_name,
            "--save_dir",
            str(save_dir),
            "--paths_file",
            manifests["train"],
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

        if manifests.get("val"):
            cmd += [
                "--val_paths_file",
                manifests["val"],
            ]

        # The original trainer uses dataset_name for metadata association.
        # In persistent-path mode it can remain empty.
        if dataset_name:
            cmd += [
                "--clearml_dataset_name",
                dataset_name,
            ]

        if load_path:
            cmd += [
                "--load_path",
                load_path,
            ]

        subprocess.run(
            cmd,
            check=True,
        )

        checkpoints = list(
            save_dir.glob(
                "checkpoints/**/*_best_*.pth"
            )
        )

        if not checkpoints:
            checkpoints = list(
                save_dir.glob(
                    "checkpoints/**/*_last_*.pth"
                )
            )

        if not checkpoints:
            raise FileNotFoundError(
                "Original trainer produced no checkpoint "
                f"under {save_dir}"
            )

        checkpoint = max(
            checkpoints,
            key=lambda p: p.stat().st_mtime,
        )

    elif execution_backend == "worker_cpu":
        cmd = [
            sys.executable,
            "-u",
            "train_worker_cpu.py",
            "--run_name",
            run_name,
            "--save_dir",
            str(save_dir),
            "--paths_file",
            manifests["train"],
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
        ]

        if manifests.get("val"):
            cmd += [
                "--val_paths_file",
                manifests["val"],
            ]

        if load_path:
            cmd += [
                "--load_path",
                load_path,
            ]

        subprocess.run(
            cmd,
            check=True,
        )

        result_path = (
            save_dir
            / "training-result.json"
        )

        if not result_path.exists():
            raise FileNotFoundError(
                "CPU trainer did not produce "
                f"{result_path}"
            )

        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        checkpoint = Path(
            result["checkpoint"]
        )

    else:
        raise ValueError(
            "execution_backend must be "
            "'worker_cpu' or 'alliance'"
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Training checkpoint does not exist: {checkpoint}"
        )

    output_model = OutputModel(
        task=task,
        name=f"{run_name}-model",
    )

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

    task.upload_artifact(
        "training_result",
        artifact_object=result,
        wait_on_upload=True,
    )

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
    if task is None:
        raise RuntimeError(
            "evaluate_step is not running inside a ClearML Task"
        )

    # Same issue as train_step:
    # use the generated Task's effective Args/* parameters rather than
    # trusting the defaults embedded in the generated function wrapper.
    task_parameters = task.get_parameters(
        cast=True
    ) or {}

    effective = {
        "dataset_id": dataset_id,
        "persistent_dataset_path": persistent_dataset_path,
        "manifest_task_id": manifest_task_id,
        "model": model,
        "image_size": image_size,
        "minimum_accuracy": minimum_accuracy,
    }

    for name in tuple(effective):
        key = f"Args/{name}"
        if key in task_parameters:
            effective[name] = task_parameters[key]

    dataset_id = str(
        effective["dataset_id"] or ""
    )

    persistent_dataset_path = str(
        effective["persistent_dataset_path"]
        or ""
    )

    manifest_task_id = str(
        effective["manifest_task_id"]
        or ""
    )

    model = str(
        effective["model"]
    )

    image_size = int(
        effective["image_size"]
    )

    minimum_accuracy = float(
        effective["minimum_accuracy"]
    )

    print(
        "[evaluate_step] effective parameters:\n"
        f"  dataset_id={dataset_id!r}\n"
        f"  persistent_dataset_path={persistent_dataset_path!r}\n"
        f"  manifest_task_id={manifest_task_id!r}\n"
        f"  model={model!r}\n"
        f"  image_size={image_size!r}\n"
        f"  minimum_accuracy={minimum_accuracy!r}"
    )

    if not dataset_id and not persistent_dataset_path:
        raise ValueError(
            "Neither dataset_id nor persistent_dataset_path is set "
            "on the ClearML evaluation Task"
        )

    # Do NOT replace `training` from Args/training here.
    #
    # `training` is the output artifact/reference from the previous
    # pipeline step and the generated ClearML function wrapper is
    # responsible for deserializing it into the Python dictionary.
    if not isinstance(training, dict):
        raise TypeError(
            "Expected training output to be a dictionary, "
            f"got {type(training).__name__}: {training!r}"
        )

    root, dataset_mode = resolve_dataset_root(
        dataset_id,
        persistent_dataset_path,
    )

    work = (
        Path(tempfile.gettempdir())
        / "clearml-orchestration"
        / task.id
    )

    manifests = materialize_manifests(
        manifest_task_id,
        root,
        work / "manifests",
    )

    out_dir = work / "evaluation"

    model_path = InputModel(
        model_id=training["model_id"]
    ).get_local_copy()

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
            str(out_dir),
            "--model",
            model,
            "--load_path",
            str(model_path),
        ],
        check=True,
    )

    pred_csv = out_dir / "pred.csv"

    if not pred_csv.exists():
        raise FileNotFoundError(
            f"Evaluation did not produce {pred_csv}"
        )

    total = 0
    correct = 0

    with pred_csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            total += 1

            if (
                str(row["Correct"])
                .strip()
                .lower()
                == "true"
            ):
                correct += 1

    if not total:
        raise RuntimeError(
            "Evaluation produced no predictions"
        )

    accuracy = correct / total

    accepted = (
        accuracy
        >= minimum_accuracy
    )

    logger = task.get_logger()

    logger.report_single_value(
        "accuracy",
        accuracy,
    )

    logger.report_single_value(
        "minimum_accuracy",
        minimum_accuracy,
    )

    logger.report_single_value(
        "accepted",
        int(accepted),
    )

    result = {
        "evaluation_task_id": task.id,
        "training_task_id": training[
            "training_task_id"
        ],
        "model_id": training[
            "model_id"
        ],
        "accuracy": accuracy,
        "minimum_accuracy": minimum_accuracy,
        "accepted": accepted,
        "dataset_mode": dataset_mode,
    }

    task.upload_artifact(
        "evaluation_result",
        artifact_object=result,
        wait_on_upload=True,
    )

    return result