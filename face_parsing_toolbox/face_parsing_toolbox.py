import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from rich import print
from rich.progress import track
import fire

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(root, filename)
                            for root, _, filenames in os.walk(image_dir)
                            for filename in filenames if filename.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

class FaceParsingToolbox:
    def __init__(self, input_path, model_name, image_size=512, batch_size=4, num_workers=4, device=None, save_mask=None, save_color_mask=None):
        self.input_path = input_path
        self.model_name = model_name
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_mask = save_mask
        self.save_color_mask = save_color_mask

        self.transformer = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])

        self.palette = np.array([
            (0,  0,  0), (204, 0,  0), (76, 153, 0),
            (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
            (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
            (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
            (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)
        ], dtype=np.uint8)

        self.data = self.load_data()
        self.model = self.load_model()

    def load_data(self):
        if os.path.isdir(self.input_path):
            dataset = ImageDataset(self.input_path, transform=self.transformer)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
            return dataloader
        else:
            image = Image.open(self.input_path).convert('RGB')
            if self.transformer:
                image = self.transformer(image)
            return [(image.unsqueeze(0).to(self.device), [self.input_path])]

    def load_model(self):
        if self.model_name == 'example_model':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
        model.to(self.device)
        model.eval()
        return model

    def inference(self):
        if self.save_mask is not None and not os.path.exists(self.save_mask):
            os.makedirs(self.save_mask)
        if self.save_color_mask is not None and not os.path.exists(self.save_color_mask):
            os.makedirs(self.save_color_mask)
        
        with torch.no_grad():
            for images, img_paths in track(self.data, description="Processing"):
                images = images.to(self.device)
                outputs = self.model(images)['out']
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()

                for i, img_path in enumerate(img_paths):
                    mask = predictions[i]

                    if self.save_mask is not None:
                        mask_img = Image.fromarray(mask.astype(np.uint8))
                        mask_img.save(os.path.join(self.save_mask, os.path.splitext(os.path.basename(img_path))[0] + '.png'))

                    if self.save_color_mask is not None:
                        color_mask = self.palette[mask]
                        color_mask_img = Image.fromarray(color_mask.astype(np.uint8))
                        color_mask_img.save(os.path.join(self.save_color_mask, os.path.splitext(os.path.basename(img_path))[0] + '.png'))

    def run(self):
        self.inference()

if __name__ == '__main__':
    fire.Fire(FaceParsingToolbox)
