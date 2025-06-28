import torch
import torchvision.transforms as transforms
import tifffile as tiff
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import UNet

# parameters
input_path_global = 'D:/PycharmProjects/Practical_Dataset/input/GutMicro3.tif'  # Input TIFF
model_path_global = 'NIR_UNet.pth'  # Pretrained model path
output_path_global = 'D:/PycharmProjects/Practical_Dataset/output/GutMicro3.tif'  # Output TIFF
device_global = 'cuda'  # 'cuda' if torch.cuda.is_available() else 'cpu'


def predict_mask_for_tiff(input_path, model_path, output_path, device):

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()

    # Load tiff file
    tiff_images = tiff.imread(input_path)

    predicted_masks = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0], std=[1.0]),  # Adjust according to your dataset
    ])

    # Use tqdm for progress display
    for image in tqdm(tiff_images, desc="Predicting masks"):
        pil_image = Image.fromarray(image).convert('L')  # Convert to grayscale
        tensor_image = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(tensor_image)
            predictions = torch.sigmoid(predictions)
            predictions = (predictions > 0.5).float()

        mask_array = (predictions.squeeze().cpu().numpy() * 255).astype(np.uint8)
        predicted_masks.append(mask_array)

    # Save predicted masks as a TIFF
    tiff.imwrite(output_path, predicted_masks, dtype='uint8')


if __name__ == '__main__':
    predict_mask_for_tiff(input_path_global, model_path_global, output_path_global, device_global)
