"""CPU-only test adapter for the ClearML worker.

This file is additive. The original Slurm/NCCL trainer remains in train_torch_test.py.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.dataset import AnimeDataset
from models.A import Attributor
from models.CNNDCT import CNNDCT
from models.Xception import Xception


def collate_fn(batch):
    batch = [x for x in batch if x is not None]
    return torch.utils.data.dataloader.default_collate(batch)


def build_model(name, image_size):
    if name == "xception":
        return Xception()
    if name in {"cnndct", "cnnpixel"}:
        return CNNDCT(image_size)
    if name == "ours":
        return Attributor(image_size)
    raise ValueError(f"Unsupported model: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_name", default="cpu-demo")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--paths_file", required=True)
    p.add_argument("--val_paths_file")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--model", default="ours", choices=["xception", "cnndct", "cnnpixel", "ours"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_epochs", type=int, default=2)
    p.add_argument("--factor", type=float, default=0.9)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--load_path")
    args = p.parse_args()

    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    dct = args.model in {"cnndct", "ours"}

    # Use the original dataset implementation; CPU adapter only replaces distributed execution.
    train_ds = AnimeDataset(0, args.paths_file, args.image_size, "worker", dct, None, False)
    val_ds = AnimeDataset(0, args.val_paths_file, args.image_size, "worker", dct, None, True) if args.val_paths_file else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn) if val_ds else None

    model = build_model(args.model, args.image_size).to(device)
    if args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience) if val_loader else None
    criterion = nn.BCEWithLogitsLoss().to(device)

    best_loss = float("inf")
    best_path = save_dir / "best.pth"
    for _epoch in range(args.n_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        if val_loader:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.float().unsqueeze(1).to(device)
                    total += float(criterion(model(images), labels))
            if scheduler:
                scheduler.step(total)
            if total <= best_loss:
                best_loss = total
                torch.save(model.state_dict(), best_path)
        else:
            torch.save(model.state_dict(), best_path)

    (save_dir / "training-result.json").write_text(
        json.dumps({"checkpoint": str(best_path)}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
