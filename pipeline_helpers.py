from pathlib import Path

from clearml import Dataset, Task


def resolve_dataset_root(dataset_id: str, persistent_dataset_path: str):
    """Return a worker-visible dataset root and its mode.

    Persistent datasets are administrator-provisioned and used directly.
    Ordinary ClearML datasets are materialized with get_local_copy().
    """
    if persistent_dataset_path:
        path = Path(persistent_dataset_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Persistent dataset is not available: {path}")
        return path, "persistent"

    if not dataset_id:
        raise ValueError("Either dataset_id or persistent_dataset_path is required")

    dataset = Dataset.get(dataset_id=dataset_id, alias="dataset", overridable=True)
    return Path(dataset.get_local_copy()).resolve(), "clearml"


def materialize_manifests(manifest_task_id: str, root: Path, dest: Path):
    """Download portable manifest artifacts and resolve entries against root."""
    task = Task.get_task(task_id=manifest_task_id)
    dest.mkdir(parents=True, exist_ok=True)
    output = {}

    for key, filename in [
        ("train", "train.txt"),
        ("val", "val.txt"),
        ("test", "test.txt"),
    ]:
        artifact_name = f"manifest_{key}_paths_file"
        if artifact_name not in task.artifacts:
            raise KeyError(
                f"Manifest task {manifest_task_id} does not contain artifact {artifact_name!r}"
            )

        source = Path(task.artifacts[artifact_name].get_local_copy())
        target = dest / filename
        with target.open("w", encoding="utf-8") as handle:
            for line_number, raw in enumerate(
                source.read_text(encoding="utf-8").splitlines(), 1
            ):
                if not raw.strip():
                    continue
                parts = raw.rstrip().split("\t", 1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid manifest line {line_number} in {source}: "
                        "expected PATH<TAB>LABEL"
                    )
                relative, label = parts
                path = Path(relative).expanduser()
                actual = path if path.is_absolute() else root / path
                handle.write(f"{actual.resolve()}\t{label}\n")

        output[key] = str(target.resolve())

    return output
