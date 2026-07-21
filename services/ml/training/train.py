from pathlib import Path

import torch

from services.ml.data.dataloader import create_train_val_loaders
from services.ml.losses.loss import get_loss
from services.ml.models.unet import UNet


def train(
    root_dir="data/train",
    output_dir="checkpoints",
    batch_size=16,
    epochs=10,
    learning_rate=1e-4,
    val_ratio=0.2,
):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, val_loader = create_train_val_loaders(
        root_dir=root_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
    )

    model = UNet().to(device)

    criterion = get_loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):

        # TRAIN

        model.train()

        running_train_loss = 0.0

        for degraded, clean in train_loader:

            degraded = degraded.to(
                device,
                dtype=torch.float32,
            )

            clean = clean.to(
                device,
                dtype=torch.float32,
            )

            prediction = model(degraded)

            loss = criterion(
                prediction,
                clean,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_train_loss += loss.item()

        train_loss = running_train_loss / len(train_loader)

        # VALIDATION

        model.eval()

        running_val_loss = 0.0

        with torch.no_grad():

            for degraded, clean in val_loader:

                degraded = degraded.to(
                    device,
                    dtype=torch.float32,
                )

                clean = clean.to(
                    device,
                    dtype=torch.float32,
                )

                prediction = model(degraded)

                loss = criterion(
                    prediction,
                    clean,
                )

                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)

        # LOG

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {val_loss:.6f}"
        )

        # CHECKPOINT


        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }

        torch.save(
            checkpoint,
            output_dir / f"epoch_{epoch + 1}.pt",
        )

        # BEST MODEL

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                checkpoint,
                output_dir / "best_model.pt",
            )

            print(
                f"Best model updated! "
                f"Validation Loss: {val_loss:.6f}"
            )


if __name__ == "__main__":

    train()
