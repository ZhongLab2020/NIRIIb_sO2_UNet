import os
import tifffile as tiff
from torch.utils.data import Dataset


class NIRDataset(Dataset):
    def __init__(self, root_dir, image_dir, mask_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.file_names = os.listdir(os.path.join(root_dir, image_dir))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_dir, self.file_names[index])
        mask_path = os.path.join(self.root_dir, self.mask_dir, self.file_names[index])
        image = tiff.imread(image_path)
        mask = tiff.imread(mask_path)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
