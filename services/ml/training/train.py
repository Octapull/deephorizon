import torch

from models.unet import UNet
from data.dataloader import create_dataloader
from losses.loss import get_loss


def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)

    dataloader = create_dataloader(
        root_dir="data/train",
        batch_size=16
    )

    criterion = get_loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    epochs = 10

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for degraded, clean in dataloader:

            degraded = degraded.to(device)
            clean = clean.to(device)

            prediction = model(degraded)

            loss = criterion(
                prediction,
                clean
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {epoch_loss:.6f}"
        )


if __name__ == "__main__":
    train()
