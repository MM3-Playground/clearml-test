# ClearML orchestration test — recovered research code

This package intentionally keeps the uploaded research implementation intact and layers ClearML orchestration around it.

## What was restored

The following come from the original `clearml-test.zip` and were restored instead of using the simplified replacements from earlier iterations:

- `datasets/dataset.py`
- all original `models/`
- all original `utils/`
- `eval.py`
- `train_base_test.py`
- `train_torch_test.py`

`train_torch_test.py` has one small compatibility change only: the two `ClearMLDataset.get(...)` metadata lookups run only when `--clearml_dataset_name` is supplied. This lets an administrator-provisioned persistent dataset use its filesystem path without requiring a ClearML Dataset name. The Slurm/NCCL/DDP training path itself is unchanged.

The CPU worker test is additive in `train_worker_cpu.py`; it does not replace the original Alliance trainer.

## Files added for ClearML orchestration

- `pipeline_submit.py` — defines/submits the training -> evaluation pipeline.
- `pipeline_steps.py` — small remote step wrappers. They call the existing training/evaluation programs.
- `pipeline_common.py` — dataset and manifest path resolution.
- `upload_manifests.py` — one-time upload of private manifest files.
- `train_worker_cpu.py` — CPU-only adapter used only by the non-Slurm test worker.
- `configs/` — example local configuration files.
- `worker/` — ClearML worker/services-agent Compose example.
- `requirements.txt` — the single project requirements file.

There is deliberately no `requirements-submit.txt`.

## Manifest contract

Manifest files contain a dataset-relative path and integer label separated by a tab:

```text
train/real/001.png<TAB>0
train/fake/002.png<TAB>1
```

If the manifests are already relative, keep `dataset_root` as an empty string in `configs/manifests.example.json`.

Upload them once from a machine that has the files:

```bash
python upload_manifests.py --config configs/manifests.local.json
```

The command prints a `manifest_task_id`. Put that ID in the run settings.

## Dataset modes

### Normal ClearML Dataset

Set:

```json
{
  "dataset_id": "CLEARML_DATASET_ID",
  "persistent_dataset_path": ""
}
```

The step calls `Dataset.get(...).get_local_copy()` inside the execution environment.

### Administrator-provisioned persistent dataset

The administrator downloads the dataset beforehand to the worker VM, for example:

```text
/development-volume/persistent-data/anime-dataset
```

The worker exposes the parent directory read-only inside every task container as:

```text
/workspace/persistent-data
```

The run then uses:

```json
{
  "dataset_id": "",
  "persistent_dataset_path": "/workspace/persistent-data/anime-dataset"
}
```

No `get_local_copy()` is called in this mode.

## Two execution backends, one repository and one pipeline

`execution_backend` controls only the training entry point.

### `worker_cpu`

Used for the current non-Slurm CPU ClearML worker test. It calls the additive `train_worker_cpu.py`, then the original `eval.py`.

```json
"execution_backend": "worker_cpu"
```

### `alliance`

Used inside an Alliance Slurm allocation. It calls the original `train_torch_test.py`, preserving its Slurm/NCCL/DDP path, then the original `eval.py`.

```json
"execution_backend": "alliance"
```

Launch locally inside the submitted Slurm job with:

```bash
python pipeline_submit.py --mode local --config configs/run.local.json
```

`start_locally(run_pipeline_steps_locally=True)` keeps both controller and steps on the current allocation. ClearML is used for tracking, not for scheduling the Alliance job.

## Remote ClearML-worker mode

The initial pipeline submission is:

```bash
python pipeline_submit.py --mode remote --config configs/run.local.json
```

The local JSON is read only by the initial local process. Its dictionary contents are stored as ClearML pipeline configuration and pipeline parameters. On remote controller execution, the code does not reopen the repository copy of the JSON.

Remote submission uses a controller on the `services` queue and training/evaluation on the `default` queue. `wait=False` means the submitting process does not wait for dependency installation, training, or evaluation.

After the initial pipeline exists, use **Pipelines -> + NEW RUN** in the ClearML UI to change the exposed pipeline parameters and launch another worker run.

## Why there are two ClearML agents in the example worker Compose

A pipeline controller lives for the duration of the pipeline. A normal ClearML agent handles one task at a time, so using the same single worker for the controller and its training step can block execution. The example therefore uses:

- `clearml-services-agent` listening on `services` in services mode for controllers;
- `clearml-worker` listening on `default` for training/evaluation.

Before starting the example Compose, create the two agent state directories and persistent-data directory if used:

```bash
sudo mkdir -p /opt/clearml/agent /opt/clearml/agent-services
sudo mkdir -p /development-volume/persistent-data
```

The persistent dataset directory is mounted only into task containers via `CLEARML_AGENT_EXTRA_DOCKER_ARGS`; it is read-only (`:ro`).

## Git requirement for remote execution

Remote Tasks execute a recorded Git remote URL and commit. Before submitting a new remote pipeline definition, commit and push code changes:

```bash
git status
git add .
git commit -m "Update ClearML pipeline"
git push
```

Local JSON settings can remain gitignored because their contents are transferred through ClearML rather than relying on the repository copy.


## Configuration handoff (important)

`pipeline_submit.py` calls `PipelineController.connect_configuration()` **before** it reads the JSON file. The `--config` path must be repository-relative (for example `configs/run.local.json`; Windows backslashes are normalized). ClearML stores the exact local file contents in the controller Task. When the controller is re-executed by the `services` agent, ClearML restores those stored contents to the same relative path before the script reads it.

This means the JSON file can be gitignored or locally modified; remote execution uses the captured contents, not the committed copy. The Git repository remains the source of code, while the captured configuration/pipeline parameters are the source of runtime settings.

Remote submission uses `wait=False` only on the submitting machine. The remotely executed services controller uses `wait=True` so it remains alive for the lifetime of the train -> evaluate pipeline.
