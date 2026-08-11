# ClearML pipeline orchestration — unified codebase

This repository uses **one training/evaluation codebase and one ClearML-decorated DAG** for both execution models:

1. **Alliance / Slurm or another machine:** execute the pipeline locally; ClearML only tracks it.
2. **ClearML worker:** bootstrap the controller from a user machine, then execute the controller and steps through ClearML queues. After the first run exists, researchers normally use **Pipelines → + NEW RUN**.

## Key design rule: code comes from Git, settings come from ClearML

Remote execution separates two things:

- **Code**: the pushed Git repository + exact commit.
- **Runtime settings**: the contents of the local JSON file supplied at bootstrap.

The JSON file itself is **not** treated as the source of truth on the worker. `pipeline_clearml.py` calls `Task.connect_configuration()` **before reading it**. On the first local/bootstrap execution, ClearML stores the local file contents in the controller Task. When the controller is recreated remotely, ClearML restores those stored contents before the script reads the path.

This means local, uncommitted edits to `configs/run.example.json` are preserved as runtime configuration even though the worker clones the committed Git repository.

The resulting dictionary is then passed as:

```python
training_pipeline(**settings)
```

so its values are also exposed as normal pipeline `Args/*` parameters and can be edited from **+ NEW RUN**.

## Required for remote mode: pushed code

The code itself must still be committed and reachable by the worker.

Check:

```bash
git rev-parse --show-toplevel
git remote get-url origin
git rev-parse HEAD
git branch --show-current
```

The launcher resolves the remote URL and exact commit automatically. You can override them with:

```bash
CLEARML_CODE_REPO=https://github.com/ORG/REPO.git
CLEARML_CODE_COMMIT=<sha>
CLEARML_CODE_BRANCH=main
```

Do **not** use `repo="."` for remote execution. The decorators use the real remote URL and exact commit.

## 1. Alliance / Slurm: execute locally, ClearML tracks

Inside the allocation:

```bash
python pipeline_clearml.py run \
  --mode local \
  --config configs/run.example.json
```

`PipelineDecorator.run_locally()` keeps the pipeline components on the current machine/allocation. No ClearML Agent executes the workload.

The same `train_torch_test.py` and `eval.py` are used by both modes. The included trainer is CPU/single-process for the orchestration demo; a real research Slurm/GPU implementation can replace its internals without changing the pipeline interface.

## 2. ClearML worker: bootstrap once, then use the UI

From a local Git checkout:

```bash
uv run pipeline_clearml.py run \
  --mode remote \
  --queue default \
  --config configs/run.example.json
```

Forward slashes are recommended even on Windows. The path is normalized by the launcher, but the important part is that **the local JSON contents are uploaded to ClearML before the remote controller reads them**.

The controller and components use:

- the actual Git remote URL;
- the exact current commit;
- `python:3.11` by default;
- the repository-root `requirements.txt` (`packages=False`).

After the pipeline has been captured, researchers normally use:

```text
ClearML → Pipelines → clearml-training-pipeline → + NEW RUN
```

The exposed pipeline arguments include dataset selection, manifest task, epochs, learning rate, batch size, etc.

## Manifest bootstrap

Private manifest files are uploaded once:

```bash
python pipeline_clearml.py manifests --config configs/manifests.example.json
```

Manifest paths should normally be relative to the logical dataset root:

```text
train/real/001.png<TAB>0
train/fake/002.png<TAB>1
```

If they are already relative, keep:

```json
"dataset_root": ""
```

## Dataset modes

### Ordinary ClearML dataset

Set:

```json
"dataset_id": "CLEARML_DATASET_ID",
"persistent_dataset_path": ""
```

The task calls `Dataset.get(...).get_local_copy()`.

### Administrator-provisioned persistent dataset

The administrator downloads the dataset in advance on a particular worker VM, e.g.:

```text
<vm-path>/persistent-data/anime-dataset
```

The worker makes the parent directory visible read-only in every task container:

```yaml
CLEARML_AGENT_EXTRA_DOCKER_ARGS: >-
  -v <vm-path>/persistent-data:/workspace/persistent-data:ro
```

Then the run settings use:

```json
"dataset_id": "",
"persistent_dataset_path": "/workspace/persistent-data/anime-dataset"
```

In persistent mode the pipeline **does not call `get_local_copy()`**.

## Dockerized ClearML worker

See `compose.worker.example.yaml` and `Dockerfile.agent`.

Important details:

- `/var/run/docker.sock` lets the agent launch sibling task containers.
- `CLEARML_AGENT_DOCKER_HOST_MOUNT` exposes the agent's host-side ClearML configuration path to spawned task containers.
- `CLEARML_AGENT_EXTRA_DOCKER_ARGS` is only needed for mounts such as persistent datasets that must be visible in every task container.
- The worker is configured CPU-only for this demo.
- Remote fileserver access uses host port `9081` (`9081:8081` on the server).

## Requirements

Remote controller and component tasks use the repository `requirements.txt`, not the Python environment of the laptop that bootstrapped the pipeline.

## Files

- `pipeline_clearml.py` — shared pipeline, local/remote launcher, configuration capture
- `train_torch_test.py` — shared demo trainer
- `eval.py` — shared evaluator
- `configs/run.example.json` — normal ClearML dataset example
- `configs/persistent.example.json` — pre-provisioned persistent dataset example
- `configs/manifests.example.json` — one-time manifest upload
- `compose.worker.example.yaml` / `Dockerfile.agent` — worker deployment
