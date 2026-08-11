# ClearML pipeline orchestration — unified codebase

This repository uses the **same training/evaluation code and the same ClearML-decorated DAG** in two modes:

1. **Alliance / Slurm / local mode** — execute the DAG on the current machine/allocation. ClearML tracks the run, but no ClearML worker executes it.
2. **ClearML worker mode** — a user runs a lightweight bootstrap locally. The pipeline controller stays on the user's machine for that first bootstrap, while `train` and `evaluate` are submitted to ClearML queues. After the pipeline exists, users normally launch subsequent runs from **ClearML → Pipelines → + NEW RUN**.

## Why `bootstrap.py` and `pipeline_clearml.py` are separate

`bootstrap.py` is only the user-facing launcher. It reads the local JSON file into a Python dictionary and calls:

```python
pipeline.training_pipeline(**settings)
```

`pipeline_clearml.py` contains the actual pipeline definition and **has no required CLI arguments**. Therefore a remotely executed ClearML task never needs to reopen `configs/run.example.json`, and local uncommitted JSON changes are passed as pipeline parameter values rather than being read from the Git checkout on the worker.

Code and settings are intentionally separate:

```text
Git repository + pushed commit  -> code executed by workers
local JSON contents             -> pipeline Args / runtime settings
```

## Initial worker-pipeline bootstrap

From a local Git checkout whose current commit has been pushed:

```bash
uv run bootstrap.py run \
  --mode remote \
  --queue default \
  --config configs/run.example.json
```

The bootstrap process itself is lightweight. Because the pipeline decorator uses `start_controller_locally=True`, only the controller logic runs locally during this first launch; the `train` and `evaluate` components are submitted to the configured ClearML queue and run in worker task containers.

After the pipeline has been created, use:

```text
ClearML → Pipelines → clearml-training-pipeline → + NEW RUN
```

The pipeline function arguments (dataset, manifest task, epochs, LR, batch size, etc.) are exposed as pipeline parameters and can be edited in the UI.

## Alliance / local execution

Inside the current machine or Slurm allocation:

```bash
python bootstrap.py run \
  --mode local \
  --config configs/run.example.json
```

`PipelineDecorator.run_locally()` executes the same decorated DAG locally. No ClearML Agent is required for execution; ClearML can still track the tasks when the machine can reach the ClearML server.

## Git / requirements behavior

Remote component tasks use the actual Git remote URL and exact current commit resolved when the pipeline module is imported. You can override them with:

```bash
export CLEARML_CODE_REPO=https://github.com/ORG/REPO.git
export CLEARML_CODE_COMMIT=<sha>
export CLEARML_CODE_BRANCH=main
```

The decorators use `packages=False`, so worker tasks use the repository-root `requirements.txt` rather than auto-capturing the user's local Python environment.

Before bootstrapping remote execution, make sure the commit is pushed:

```bash
git remote get-url origin
git rev-parse HEAD
git ls-remote origin | grep "$(git rev-parse HEAD)"
```

## Manifest bootstrap

Private manifest files are uploaded once:

```bash
uv run bootstrap.py manifests --config configs/manifests.example.json
```

Manifest entries should normally be relative to the logical dataset root:

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

```json
"dataset_id": "CLEARML_DATASET_ID",
"persistent_dataset_path": ""
```

The task calls `Dataset.get(...).get_local_copy()`.

### Administrator-provisioned persistent dataset

The administrator pre-downloads the dataset on a particular worker VM, for example:

```text
<vm-path>/persistent-data/anime-dataset
```

The agent exposes the parent directory read-only to every task container:

```yaml
CLEARML_AGENT_EXTRA_DOCKER_ARGS: >-
  -v <vm-path>/persistent-data:/workspace/persistent-data:ro
```

Then run settings use:

```json
"dataset_id": "",
"persistent_dataset_path": "/workspace/persistent-data/anime-dataset"
```

Persistent mode does **not** call `get_local_copy()`.

## Worker deployment

See `compose.worker.example.yaml` and `Dockerfile.agent`.

Important details:

- `/var/run/docker.sock` lets the Dockerized agent create sibling task containers.
- `CLEARML_AGENT_DOCKER_HOST_MOUNT` propagates the host-side ClearML config into spawned task containers.
- `CLEARML_AGENT_EXTRA_DOCKER_ARGS` is used for optional mounts such as read-only persistent datasets.
- This demo worker is CPU-only.
- Remote fileserver access uses host port `9081` because the server publishes `9081:8081`.

## Main files

- `bootstrap.py` — local CLI/bootstrap only
- `pipeline_clearml.py` — CLI-free pipeline definition
- `train_torch_test.py` — shared CPU demo trainer
- `eval.py` — shared evaluator
- `requirements.txt` — worker task dependencies
- `configs/run.example.json` — ordinary ClearML dataset example
- `configs/persistent.example.json` — pre-provisioned persistent dataset example
- `configs/manifests.example.json` — one-time manifest upload
- `compose.worker.example.yaml` / `Dockerfile.agent` — ClearML worker deployment
