import os
from pathlib import Path

import torch

from services.ml.data.dataloader import create_train_val_loaders
from services.ml.models.unet import UNet


def train_model(
    root_dir,
    output_dir="./artifacts",
    batch_size=16,
    epochs=5,
    learning_rate=1e-4,
    val_ratio=0.2,
    checkpoint_interval=1,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = create_train_val_loaders(
        root_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
    )

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.L1Loss()

    best_val_loss = float("inf")
    best_checkpoint_path = output_path / "best_model.pt"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for degraded, clean in train_loader:
            degraded = degraded.to(device, dtype=torch.float32)
            clean = clean.to(device, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = model(degraded)
            loss = criterion(outputs, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for degraded, clean in val_loader:
                degraded = degraded.to(device, dtype=torch.float32)
                clean = clean.to(device, dtype=torch.float32)
                outputs = model(degraded)
                val_loss += criterion(outputs, clean).item()

        val_loss = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if (epoch + 1) % checkpoint_interval == 0 or val_loss < best_val_loss:
            checkpoint_path = output_path / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                best_checkpoint_path,
            )
            print(f"Saved best checkpoint: {best_checkpoint_path}")

    return best_checkpoint_path


if __name__ == "__main__":
    root_dir = os.environ.get("DEEPHORIZON_DATA_ROOT", "./data/raw/simulated")
    train_model(root_dir=root_dir)
