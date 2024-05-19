import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

class CustomDataset(Dataset):
    def __init__(self, root_dir_real, root_dir_drawing, transform=None):
        self.root_dir_real = root_dir_real
        self.root_dir_drawing = root_dir_drawing
        self.transform = transform

        # Get list of filenames for real and drawing images
        self.real_images = sorted(os.listdir(root_dir_real))
        self.drawing_images = sorted(os.listdir(root_dir_drawing))
        
        # Check if the number of real and drawing images match
        assert len(self.real_images) == len(self.drawing_images), "Number of real and drawing images must match"

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        # Read images
        real_img_name = os.path.join(self.root_dir_real, self.real_images[idx])
        drawing_img_name = os.path.join(self.root_dir_drawing, self.drawing_images[idx])
        real_image = Image.open(real_img_name).convert("RGB")
        drawing_image = Image.open(drawing_img_name).convert("RGB")
        
        # Apply transformations if provided
        if self.transform:
            real_image = self.transform(real_image)
            drawing_image = self.transform(drawing_image)
        
        return real_image, drawing_image

#Larger batch size trains faster but might not capture the important details?
batch_size = 8
#imageSize = (280, 190)
imageSize = (256, 256)

# Define transforms
transform = transforms.Compose([
    transforms.Resize(imageSize),  # Resize images
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])  # Normalize images
])


# Path to your dataset directories
real_images_dir = "C:/WorkingSets/TrainingSets/Valid"
drawing_images_dir = "C:/WorkingSets/TrainingSets/Drawing"

# Create dataset instance
custom_dataset = CustomDataset(root_dir_real=real_images_dir, root_dir_drawing=drawing_images_dir, transform=transform)

# Create DataLoader instance
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
