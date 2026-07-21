from pathlib import Path

import torch

from services.ml.evaluation.benchmark import (
    ValidationSummary,
    evaluate_validation_loader,
    save_sample_outputs,
    save_validation_summary,
    update_best_model,
)
from services.ml.data.dataloader import create_train_val_loaders
from services.ml.models.unet import UNet
from services.ml.losses.loss import get_loss


def train(
    root_dir="data/train",
    output_dir="checkpoints",
    batch_size=16,
    epochs=10,
    learning_rate=1e-4,
    val_ratio=0.2,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = create_train_val_loaders(
        root_dir=root_dir,
        batch_size=batch_size,
        val_ratio=val_ratio,
    )

    model = UNet().to(device)
    criterion = get_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0

        for degraded, clean in train_loader:
            degraded = degraded.to(device, dtype=torch.float32)
            clean = clean.to(device, dtype=torch.float32)

            prediction = model(degraded)
            loss = criterion(prediction, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += float(loss.item())

        train_loss = running_train_loss / max(1, len(train_loader))

        validation_summary, sample_batch = evaluate_validation_loader(
            model=model,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
        )
        validation_summary = ValidationSummary(
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=validation_summary.val_loss,
            psnr=validation_summary.psnr,
            ssim=validation_summary.ssim,
        )

        if sample_batch is not None:
            save_sample_outputs(
                degraded=sample_batch[0],
                prediction=sample_batch[1],
                clean=sample_batch[2],
                output_dir=output_dir,
                epoch=epoch + 1,
            )

        save_validation_summary(validation_summary, output_dir)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} "
            f"Val Loss: {validation_summary.val_loss:.6f} "
            f"PSNR: {validation_summary.psnr:.4f} "
            f"SSIM: {validation_summary.ssim:.4f}"
        )

        checkpoint_path = output_dir / f"epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": validation_summary.val_loss,
                "psnr": validation_summary.psnr,
                "ssim": validation_summary.ssim,
            },
            checkpoint_path,
        )

        best_val_loss, is_best, best_checkpoint_path = update_best_model(
            summary=validation_summary,
            model=model,
            optimizer=optimizer,
            output_dir=output_dir,
            best_val_loss=best_val_loss,
        )
        if is_best:
            print(f"Best model updated: {best_checkpoint_path}")

    return best_checkpoint_path


if __name__ == "__main__":
    train()
