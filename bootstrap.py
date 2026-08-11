import argparse
import json
import os
from pathlib import Path


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap or locally execute the ClearML training pipeline."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    manifests = sub.add_parser("manifests")
    manifests.add_argument("--config", required=True)

    run = sub.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--mode", choices=["local", "remote"], default="remote")
    run.add_argument("--queue", default=os.getenv("CLEARML_PIPELINE_QUEUE", "default"))

    args = parser.parse_args()

    # Import only after local CLI/config handling. The imported module contains the
    # actual pipeline definition and has no required command-line interface.
    from clearml.automation.controller import PipelineDecorator
    import pipeline_clearml as pipeline

    if args.cmd == "manifests":
        config = _load_json(args.config)
        print(json.dumps({"manifest_task_id": pipeline.upload_manifest_bundle(config)}, indent=2))
        return

    settings = _load_json(args.config)

    if args.mode == "local":
        # Alliance/local execution: same decorated functions, no ClearML worker.
        PipelineDecorator.run_locally()
    else:
        # Bootstrap mode: controller stays on this machine (lightweight), while
        # components are queued remotely. The JSON file is never opened by the
        # worker; only this in-memory settings dictionary is passed to the pipeline.
        if args.queue != pipeline.DEFAULT_QUEUE:
            raise ValueError(
                f"Pipeline components were defined for queue {pipeline.DEFAULT_QUEUE!r}. "
                "Set CLEARML_PIPELINE_QUEUE before launching if you need another queue."
            )
        pipeline._require_remote_git()

    pipeline.training_pipeline(**settings)


if __name__ == "__main__":
    main()
