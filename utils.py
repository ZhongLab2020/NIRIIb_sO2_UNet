import torch
import tifffile as tiff
from dataset import NIRDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def save_checkpoint(state, filename='NIR_UNet.pth'):
    print('==> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('==> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
        batch_size,
        num_workers,
        pin_memory,
        train_transform,
        val_transform,
        root_dir,
        train_image_dir,
        train_mask_dir,
        val_image_dir,
        val_mask_dir
):
    train_ds = NIRDataset(
        root_dir=root_dir,
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = NIRDataset(
        root_dir=root_dir,
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        transform=val_transform
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()
            num_correct += (y == predictions).sum().item()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * y).sum().item()) / ((predictions + y).sum().item())

    print(f'Accuracy: {num_correct / num_pixels * 100:.2f}')
    print(f'Dice score: {dice_score / len(loader)}')
    model.train()
    return dice_score / len(loader)


def save_predictions_as_images(loader, model, folder, device):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            predictions = torch.sigmoid(model(x))
            predictions = (predictions > 0.5).float()

        # 乘以255并转换为uint8
        predictions = (predictions.squeeze(0).cpu().numpy() * 255).astype('uint8')
        ground_truth = (y.squeeze(0).cpu().numpy() * 255).astype('uint8')

        tiff.imwrite(f'{folder}/predictions_{idx}.tif', predictions)
        tiff.imwrite(f'{folder}/gt_{idx}.tif', ground_truth)

    model.train()


def plot_metrics(losses, dice_scores, epoch, folder):
    plt.figure(figsize=(10, 5))
    epochs_range = list(range(1, len(losses) + 1))
    dice_epoch_range = list(range(10, epoch + 1, 10))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)

    # Dice Score
    plt.subplot(1, 2, 2)
    plt.plot(dice_epoch_range, dice_scores, label='Validation Dice Score', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title('Validation Dice Score')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{folder}/metrics_curve.png')
    plt.close()
