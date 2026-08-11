import argparse
import json
import tempfile
from pathlib import Path

from clearml import Task

from pipeline_common import read_manifest


def portable_manifest(source, destination, dataset_root=""):
    root = Path(dataset_root).expanduser().resolve() if dataset_root else None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for raw_path, label in read_manifest(source):
            p = Path(raw_path).expanduser()
            rendered = p.as_posix()
            if root is not None and p.is_absolute():
                rendered = p.resolve().relative_to(root).as_posix()
            handle.write(f"{rendered}\t{label}\n")
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    task = Task.init(
        project_name=cfg["clearml_project_name"],
        task_name=cfg.get("task_name", "manifest-bundle"),
        task_type=Task.TaskTypes.data_processing,
        output_uri=True,
    )
    with tempfile.TemporaryDirectory(prefix="clearml-manifest-upload-") as tmp:
        for kind in ("train", "val", "test"):
            p = portable_manifest(
                cfg[f"{kind}_paths_file"],
                Path(tmp) / f"{kind}.txt",
                cfg.get("dataset_root", ""),
            )
            task.upload_artifact(f"{kind}_manifest", artifact_object=str(p), wait_on_upload=True)
    task.close()
    print(json.dumps({"manifest_task_id": task.id}, indent=2))


if __name__ == "__main__":
    main()
