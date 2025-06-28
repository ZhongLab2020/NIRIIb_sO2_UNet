import torch
import albumentations as al
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNet
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_images,
    plot_metrics,
)

# Hyperparameters etc.
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
num_epochs = 2000
num_workers = 2
image_height = 480
image_width = 480
pin_memory = True
load_model = True
root_dir = 'D:/PycharmProjects/Training_Dataset'
train_image_dir = 'gut (closed + opened belly)_images_train'
train_mask_dir = 'gut (closed + opened belly)_masks_train'
val_image_dir = 'gut (closed + opened belly)_images_val'
val_mask_dir = 'gut (closed + opened belly)_masks_val'


def main():
    train_transform = al.Compose([
        al.Resize(height=image_height, width=image_width),
        al.Rotate(limit=35, p=1.0),
        al.HorizontalFlip(p=0.5),
        al.VerticalFlip(p=0.5),
        al.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,  # depends on the bit depth of the images
        ),
        ToTensorV2(),
    ])

    val_transform = al.Compose([
        al.Resize(height=image_height, width=image_width),
        al.Normalize(
            mean=[0.0],
            std=[1.0],
            max_pixel_value=255.0,  # depends on the bit depth of the masks
        ),
        ToTensorV2(),
    ])

    model = UNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(
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
    )

    if load_model:
        load_checkpoint(torch.load('NIR_UNet.pth'), model)

    # check_accuracy(val_loader, model, device=device)
    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    dice_scores = []

    for epoch in range(num_epochs):
        model.train()
        loop = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        epoch_loss = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.float().unsqueeze(1).to(device=device)

            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            # save model
            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            # check accuracy and log dice score
            dice = check_accuracy(val_loader, model, device=device)
            dice_scores.append(dice)

            # save prediction image
            save_predictions_as_images(val_loader, model, folder='saved_images/', device=device)

            # plot loss and dice
            plot_metrics(train_losses, dice_scores, epoch + 1, folder='saved_images')


if __name__ == '__main__':
    main()
