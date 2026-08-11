import tempfile
from pathlib import Path


def read_manifest(path):
    rows = []
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        parts = raw.rstrip().split("\t")
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest line {line_no}: expected PATH<TAB>LABEL")
        rows.append((parts[0], int(parts[1])))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def resolve_dataset_root(dataset_id="", persistent_dataset_path=""):
    from clearml import Dataset

    if persistent_dataset_path:
        root = Path(persistent_dataset_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Persistent dataset is not available: {root}")
        return root, "persistent"

    if not dataset_id:
        raise ValueError("Either dataset_id or persistent_dataset_path is required")

    dataset = Dataset.get(dataset_id=dataset_id, alias="dataset", overridable=True)
    return Path(dataset.get_local_copy()).resolve(), "clearml"


def materialize_manifests(manifest_task_id, dataset_root, output_dir=None):
    from clearml import Task

    if not manifest_task_id:
        raise ValueError("manifest_task_id is required")
    source_task = Task.get_task(task_id=manifest_task_id)
    out_dir = Path(output_dir or tempfile.mkdtemp(prefix="clearml-manifests-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    for kind in ("train", "val", "test"):
        artifact_name = f"{kind}_manifest"
        if artifact_name not in source_task.artifacts:
            if kind == "val":
                continue
            raise KeyError(f"Manifest task {manifest_task_id} has no artifact {artifact_name!r}")
        src = Path(source_task.artifacts[artifact_name].get_local_copy())
        dst = out_dir / f"{kind}.txt"
        with dst.open("w", encoding="utf-8") as handle:
            for raw_path, label in read_manifest(src):
                p = Path(raw_path).expanduser()
                actual = p if p.is_absolute() else Path(dataset_root) / p
                handle.write(f"{actual.resolve()}\t{label}\n")
        result[kind] = str(dst.resolve())
    return result
