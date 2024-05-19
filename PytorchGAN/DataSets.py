from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

#Larger batch size trains faster but might not capture the important details?
batch_size = 8
#imageSize = (280, 190)
imageSize = (256, 256)


# Define transforms
transform = transforms.Compose([
    transforms.Resize(imageSize),  # Resize images
    transforms.ToTensor(),           # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Create ImageFolder dataset
dataset = ImageFolder(root='C:/WorkingSets/TrainingSets', transform=transform)



# Assuming you have a dataset called 'my_dataset' containing your data
dataset_size = len(dataset)
train_size = int(0.8 * dataset_size)
test_size = dataset_size - train_size

# Split the dataset into training and testing
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoader instances for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



manualTestDataSet = ImageFolder(root='C:/ManualTestSet', transform=transform)
manualTestDataloader = DataLoader(manualTestDataSet, batch_size=batch_size, shuffle=False)