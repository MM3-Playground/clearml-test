# ClearML pipeline orchestration — unified codebase

## Required for remote mode: run from a Git checkout

The remote ClearML mode is Git-native. Run the bootstrap command from an actual Git repository that has a reachable remote and a committed/pushed revision. The worker uses the repository metadata recorded by ClearML to clone the code.

Check before launching:

```bash
git rev-parse --show-toplevel
git remote -v
git status
git rev-parse HEAD
```

If this ZIP was only extracted into a normal directory, ClearML cannot infer a repository from it. Put the files in the intended Git repository, commit them, and push the commit before using remote mode.


This project uses **one decorated ClearML pipeline and one training/evaluation implementation** for both execution models.

## 1. Alliance / Slurm: execute locally, ClearML tracks

Submit the repository with your normal `sbatch` script and, inside the allocation, run:

```bash
python pipeline_clearml.py run --mode local --config configs/run.example.json
```

`PipelineDecorator.run_locally()` keeps the decorated DAG on the current machine/allocation. No ClearML Agent/queue is used for the pipeline components. ClearML still tracks tasks if the Alliance node can reach the ClearML server.

For Alliance storage, set `persistent_dataset_path` to the dataset path visible in the allocation (and leave `dataset_id` empty) when you do not want ClearML to download data. The same manifest contract is used: paths in the manifest are relative to the selected dataset root.

> The demo trainer in this package is deliberately CPU/single-process so it can also run on the test VM. For a real GPU/Slurm research trainer, replace the internals of `train_torch_test.py` with the research training implementation; the pipeline/orchestration code does not need to fork into a second workflow.

## 2. ClearML worker: bootstrap once, then use the UI

From a Git checkout that has been committed and pushed:

```bash
python pipeline_clearml.py run --mode remote --queue default --config configs/run.example.json
```

The bootstrap task is handed to the ClearML queue; the laptop does not perform training. The component decorators use `repo="."` and `packages=False`, so ClearML records the repository/commit and the Agent uses the repository `requirements.txt`.

After the pipeline has been captured, researchers normally use **Pipelines → clearml-training-pipeline → + NEW RUN** and edit the exposed pipeline parameters in the UI.

## Manifest bootstrap

Private manifest files are uploaded once:

```bash
python pipeline_clearml.py manifests --config configs/manifests.example.json
```

A manifest line is:

```text
train/real/001.png<TAB>0
train/fake/002.png<TAB>1
```

If these are already relative to the dataset root, keep `dataset_root` as `""` in `manifests.example.json`.

## Dataset modes

### Ordinary ClearML dataset

Set `dataset_id` and leave `persistent_dataset_path` empty. The component calls `Dataset.get(...).get_local_copy()`.

### Administrator-provisioned persistent dataset

Pre-download it on the worker VM, for example under:

```text
<vm-path>/persistent-data/anime-dataset
```

Expose the parent directory read-only to every task container:

```yaml
CLEARML_AGENT_EXTRA_DOCKER_ARGS: >-
  -v <vm-path>/persistent-data:/workspace/persistent-data:ro
```

Then set:

```json
"dataset_id": "",
"persistent_dataset_path": "/workspace/persistent-data/anime-dataset"
```

The pipeline does **not** call `get_local_copy()` for this mode.

## Files

- `pipeline_clearml.py` — one pipeline; `--mode local|remote`
- `train_torch_test.py` — shared demo trainer
- `eval.py` — shared evaluator
- `configs/run.example.json` — normal dataset example
- `configs/persistent.example.json` — pre-provisioned dataset example
- `configs/manifests.example.json` — one-time manifest upload
- `compose.worker.example.yaml` / `Dockerfile.agent` — ClearML worker
