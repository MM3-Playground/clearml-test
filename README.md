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
