
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from clearml import Dataset as ClearMLDataset, OutputModel, Task

from datasets.dataset import AnimeDataset
from models.A import Attributor
from models.CNNDCT import CNNDCT
from models.Xception import Xception
from train_base_test import collate_fn, parse_args


def build_model(args, device: torch.device):
    if args.model == "xception":
        model = Xception()
        dct = False
    elif args.model in ("cnndct", "cnnpixel"):
        model = CNNDCT(args.image_size)
        dct = args.model == "cnndct"
    elif args.model == "ours":
        model = Attributor(args.image_size)
        dct = True
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    return model.to(device), dct


def build_loader(args, paths_file: str, dct: bool, val: bool):
    dataset = AnimeDataset(
        0,
        paths_file,
        args.id,
        args.image_size,
        dct,
        args.val_n_c_samples if val else args.n_c_samples,
        val,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not val,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=not val,
        collate_fn=collate_fn,
    )


def batch_loss(model, batch, criterion, device):
    images, labels = batch
    images = images.to(device)
    labels = labels.to(device)
    logits = model(images)
    return criterion(logits, labels.float().unsqueeze(1))


def main():
    args = parse_args()
    if args.id is None:
        from datetime import datetime
        args.id = datetime.now().strftime("%Y%m%d%H%M%S")

    task = Task.init(
        project_name=args.clearml_project_name,
        task_name=args.clearml_task_name,
        task_type=Task.TaskTypes.training,
        output_uri=True,
        auto_connect_frameworks={"pytorch": False},
        auto_connect_arg_parser=True,
    )

    if args.clearml_dataset_id:
        # Records exact dataset lineage on this task. The pipeline component already
        # materialized the dataset/manifests in this same container.
        ClearMLDataset.get(dataset_id=args.clearml_dataset_id, alias="training", overridable=False)

    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    checkpoint_dir = Path(args.save_dir) / "checkpoints" / f"{args.id}_{args.run_name}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.save_dir) / "logs" / f"{args.id}_{args.run_name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))

    model, dct = build_model(args, device)
    if args.load_path and args.load_path != "timm":
        model.load_state_dict(torch.load(args.load_path, map_location=device))

    train_loader = build_loader(args, args.paths_file, dct, val=False)
    val_loader = build_loader(args, args.val_paths_file, dct, val=True) if args.val_paths_file else None

    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = None
    if val_loader is not None and args.patience is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=args.factor, patience=args.patience
        )
    elif args.decay_epoch:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.decay_epoch, gamma=args.factor
        )

    if args.cond_state:
        state = torch.load(args.cond_state, map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("lr_scheduler"):
            scheduler.load_state_dict(state["lr_scheduler"])

    criterion = nn.BCEWithLogitsLoss()
    best_val = float("inf")
    best_checkpoint = None
    epochs_without_improvement = 0

    for epoch in range(args.cond_epoch, args.n_epochs):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = batch_loss(model, batch, criterion, device)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())
            train_count += 1

        train_loss = train_total / max(train_count, 1)
        writer.add_scalar("Epoch Loss/Train", train_loss, epoch)
        task.get_logger().report_scalar("loss", "train", train_loss, epoch)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_count = 0
            with torch.no_grad():
                for batch in val_loader:
                    loss = batch_loss(model, batch, criterion, device)
                    val_total += float(loss.item())
                    val_count += 1
            val_loss = val_total / max(val_count, 1)
            writer.add_scalar("Epoch Loss/Val", val_loss, epoch)
            task.get_logger().report_scalar("loss", "validation", val_loss, epoch)

        metric = val_loss if val_loss is not None else train_loss
        improved = metric <= best_val
        if improved:
            best_val = metric
            epochs_without_improvement = 0
            best_checkpoint = checkpoint_dir / f"{args.id}_best_{epoch}.pth"
            torch.save(model.state_dict(), best_checkpoint)
        else:
            epochs_without_improvement += 1

        last_checkpoint = checkpoint_dir / f"{args.id}_last_{epoch}.pth"
        torch.save(model.state_dict(), last_checkpoint)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metric)
            else:
                scheduler.step()

        print(
            f"Epoch {epoch}/{args.n_epochs - 1}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss if val_loss is not None else 'n/a'}"
        )
        if args.n_early is not None and epochs_without_improvement > args.n_early:
            print("Early stopping")
            break

    writer.close()
    if best_checkpoint is None:
        raise RuntimeError("Training produced no checkpoint")

    output_model = OutputModel(task=task, name=f"{args.clearml_task_name}-model")
    output_model.update_weights(
        weights_filename=str(best_checkpoint),
        target_filename=best_checkpoint.name,
        auto_delete_file=False,
        async_enable=False,
    )

    result = {
        "training_task_id": task.id,
        "model_id": output_model.id,
        "best_checkpoint": str(best_checkpoint),
        "device": "cpu",
    }
    result_path = Path(args.save_dir) / "training-result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    task.upload_artifact("training-result", artifact_object=str(result_path), wait_on_upload=True)
    task.close()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
