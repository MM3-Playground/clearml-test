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

    dataset = Dataset.get(
        dataset_id=dataset_id,
        alias="dataset",
    )

    # TEMPORARY DEBUGGING
    files = dataset.list_files()

    print(f"[dataset] id={dataset_id}")
    print(f"[dataset] number of files={len(files)}")
    print(f"[dataset] first files={files[:20]}")

    root = Path(
        dataset.get_local_copy(
            use_soft_links=False,
            raise_on_error=True,
        )
    ).resolve()

    print(f"[dataset] local root={root}")

    if root.exists():
        print(
            f"[dataset] local entries="
            f"{list(root.iterdir())[:20]}"
        )
    else:
        print("[dataset] local root DOES NOT EXIST")

    return root, "clearml"


def materialize_manifests(
    manifest_task_id,
    dataset_root,
    output_dir=None,
):
    from pathlib import Path

    from clearml import Task

    if not manifest_task_id:
        raise ValueError(
            "manifest_task_id is required"
        )

    source_task = Task.get_task(
        task_id=manifest_task_id
    )

    dataset_root = Path(
        dataset_root
    ).expanduser().resolve()

    if output_dir is None:
        output_dir = Path.cwd() / "manifests"

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Support both the current uploader and manifest Tasks
    # produced by previous versions of this demo.
    artifact_candidates = {
        "train": [
            "train_manifest",
            "manifest_train_paths_file",
        ],
        "val": [
            "val_manifest",
            "manifest_val_paths_file",
        ],
        "test": [
            "test_manifest",
            "manifest_test_paths_file",
        ],
    }

    result = {}

    for kind, candidates in artifact_candidates.items():
        artifact_name = next(
            (
                name
                for name in candidates
                if name in source_task.artifacts
            ),
            None,
        )

        if artifact_name is None:
            available = sorted(
                source_task.artifacts.keys()
            )

            raise KeyError(
                f"Manifest task {manifest_task_id} "
                f"has no artifact for {kind!r}. "
                f"Expected one of {candidates}; "
                f"available artifacts: {available}"
            )

        print(
            f"[manifests] {kind}: "
            f"using artifact {artifact_name!r}"
        )

        source = Path(
            source_task.artifacts[
                artifact_name
            ].get_local_copy()
        )

        target = (
            output_dir
            / f"{kind}.txt"
        )

        with target.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for raw in source.read_text(
                encoding="utf-8"
            ).splitlines():
                if not raw.strip():
                    continue

                relative_path, label = (
                    raw.rstrip().split(
                        "\t",
                        1,
                    )
                )

                path = Path(
                    relative_path
                ).expanduser()

                if path.is_absolute():
                    actual = path
                else:
                    actual = (
                        dataset_root
                        / path
                    )

                handle.write(
                    f"{actual.resolve()}"
                    f"\t{label}\n"
                )

        result[kind] = str(
            target.resolve()
        )

    return result
