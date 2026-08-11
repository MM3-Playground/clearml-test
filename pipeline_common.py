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


def resolve_dataset_root(
    dataset_id="",
    persistent_dataset_path="",
):
    from clearml import Dataset

    # ---------------------------------------------------------
    # Administrator-provisioned persistent dataset
    # ---------------------------------------------------------
    #
    # The dataset is already physically downloaded on the worker
    # and mounted into the task container.
    #
    # Do NOT ask ClearML to download anything in this mode.
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

    # ---------------------------------------------------------
    # Normal ClearML Dataset
    # ---------------------------------------------------------

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

    # ClearML's Dataset consists of external HTTPS links in our
    # test case.
    #
    # The underlying HTTP objects are downloaded into ClearML's
    # storage cache. On Linux, get_local_copy() can construct the
    # dataset view using symbolic links to those cached objects.
    #
    # We deliberately use soft links here instead of
    # get_mutable_local_copy(), because the latter depends on the
    # assembled local dataset copy anyway.
    root = Path(
        dataset.get_local_copy(
            use_soft_links=True,
            raise_on_error=True,
        )
    ).resolve()

    print(
        f"[dataset] local root={root}"
    )

    downloaded_files = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    ]

    print(
        f"[dataset] local files="
        f"{downloaded_files[:20]}"
    )

    if not downloaded_files:
        raise RuntimeError(
            f"ClearML Dataset {dataset_id} resolved to "
            f"{root}, but the assembled dataset directory "
            "contains no files"
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

    # Support both the current manifest uploader and the older
    # manifest Tasks created during the previous tests.
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
            # Validation is allowed to be absent.
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

        src = Path(
            source_task.artifacts[
                artifact_name
            ].get_local_copy()
        )

        dst = (
            out_dir
            / f"{kind}.txt"
        )

        with dst.open(
            "w",
            encoding="utf-8",
        ) as handle:
            for raw_path, label in read_manifest(src):
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
            dst.resolve()
        )

    return result