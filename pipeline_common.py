import os
import shutil
import tempfile
from pathlib import Path


def read_manifest(path):
    rows = []

    for line_no, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not raw.strip():
            continue

        parts = raw.rstrip().split("\t")

        if len(parts) != 2:
            raise ValueError(
                f"Invalid manifest line {line_no}: "
                "expected PATH<TAB>LABEL"
            )

        rows.append(
            (
                parts[0],
                int(parts[1]),
            )
        )

    if not rows:
        raise ValueError(
            f"Manifest is empty: {path}"
        )

    return rows


def _materialize_external_link_dataset(
    dataset,
    dataset_id,
):
    """
    Workaround for ClearML external-link-only datasets.

    On the current Dockerized ClearML worker, Dataset.get_local_copy()
    successfully downloads HTTPS objects into ClearML's StorageManager
    cache, but the assembled dataset directory remains empty.

    We therefore:
      1. let ClearML StorageManager resolve/cache each external URL;
      2. create our own dataset-relative view under /tmp;
      3. symlink the cached files into their registered relative paths.

    Researchers do not need to call StorageManager themselves.
    """

    from clearml import StorageManager, Task

    task = Task.current_task()

    task_id = (
        task.id
        if task is not None
        else "local"
    )

    root = (
        Path(tempfile.gettempdir())
        / "clearml-external-datasets"
        / task_id
        / dataset_id
    )

    # This directory is task-specific, so recreating it is safe.
    if root.exists():
        shutil.rmtree(root)

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    link_entries = dataset.link_entries_dict

    if not link_entries:
        raise RuntimeError(
            f"Dataset {dataset_id} has no external link entries"
        )

    print(
        f"[dataset] materializing "
        f"{len(link_entries)} external links into {root}"
    )

    for relative_path, entry in link_entries.items():
        remote_url = str(entry.link)

        cached_file = StorageManager.get_local_copy(
            remote_url=remote_url,
        )

        if not cached_file:
            raise RuntimeError(
                f"Failed to retrieve external dataset file: "
                f"{remote_url}"
            )

        cached_file = Path(
            cached_file
        ).resolve()

        if not cached_file.is_file():
            raise FileNotFoundError(
                f"ClearML StorageManager returned {cached_file}, "
                "but it is not a file"
            )

        destination = (
            root
            / relative_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Prefer a symlink so we do not duplicate the cached data.
        #
        # We already verified that the ClearML task/container environment
        # can create and follow symlinks under its cache filesystem.
        try:
            destination.symlink_to(
                cached_file
            )
        except OSError:
            # Defensive fallback in case a different worker filesystem
            # does not permit symlinks.
            shutil.copy2(
                cached_file,
                destination,
            )

    materialized_files = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    ]

    print(
        f"[dataset] materialized files="
        f"{materialized_files[:20]}"
    )

    if len(materialized_files) != len(link_entries):
        raise RuntimeError(
            f"Dataset {dataset_id} has {len(link_entries)} link entries "
            f"but only {len(materialized_files)} files were materialized"
        )

    return root


def resolve_dataset_root(
    dataset_id="",
    persistent_dataset_path="",
):
    from clearml import Dataset

    # =========================================================
    # Mode 1: administrator-provisioned persistent dataset
    # =========================================================
    #
    # Already downloaded onto the worker VM and mounted into
    # task containers, e.g.
    #
    # /workspace/persistent-data/my-dataset
    #
    # No ClearML download occurs.
    #
    if persistent_dataset_path:
        root = Path(
            persistent_dataset_path
        ).expanduser().resolve()

        if not root.is_dir():
            raise FileNotFoundError(
                f"Persistent dataset is not available: {root}"
            )

        print(
            f"[dataset] using persistent dataset: {root}"
        )

        return root, "persistent"

    # =========================================================
    # Mode 2: normal ClearML Dataset
    # =========================================================

    dataset_id = str(
        dataset_id or ""
    ).strip()

    if not dataset_id:
        raise ValueError(
            "Either dataset_id or "
            "persistent_dataset_path is required"
        )

    print(
        f"[dataset] resolving ClearML Dataset "
        f"{dataset_id}"
    )

    dataset = Dataset.get(
        dataset_id=dataset_id,
        alias="dataset",
    )

    file_entries = dataset.file_entries_dict
    link_entries = dataset.link_entries_dict

    print(
        f"[dataset] file entries={len(file_entries)}"
    )

    print(
        f"[dataset] link entries={len(link_entries)}"
    )

    # =========================================================
    # External-link-only dataset
    # =========================================================
    #
    # This is the case for your current Alliance Swift dataset:
    #
    # file_entries == 0
    # link_entries == 12
    #
    # ClearML 2.1.11 on this Docker worker downloads the HTTPS
    # objects into StorageManager's cache but does not assemble the
    # Dataset.get_local_copy() directory.
    #
    if not file_entries and link_entries:
        root = _materialize_external_link_dataset(
            dataset=dataset,
            dataset_id=dataset_id,
        )

        return root.resolve(), "clearml_external"

    # =========================================================
    # Ordinary ClearML-managed dataset
    # =========================================================
    #
    # For datasets containing uploaded/managed files, use the
    # normal ClearML Dataset API.
    #
    root = Path(
        dataset.get_local_copy(
            use_soft_links=True,
            raise_on_error=True,
        )
    ).resolve()

    files = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    ]

    print(
        f"[dataset] local root={root}"
    )

    print(
        f"[dataset] local files={files[:20]}"
    )

    if not files:
        raise RuntimeError(
            f"ClearML Dataset {dataset_id} resolved to "
            f"{root}, but contains no materialized files"
        )

    return root, "clearml"


def materialize_manifests(
    manifest_task_id,
    dataset_root,
    output_dir=None,
):
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

    out_dir = Path(
        output_dir
        or tempfile.mkdtemp(
            prefix="clearml-manifests-"
        )
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {}

    # Support both naming conventions used by manifest Tasks
    # created during our ClearML testing.
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
            # Validation manifests may legitimately be absent.
            if kind == "val":
                continue

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

        destination = (
            out_dir
            / f"{kind}.txt"
        )

        with destination.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for raw_path, label in read_manifest(source):
                path = Path(
                    raw_path
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
            destination.resolve()
        )

    return result