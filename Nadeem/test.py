import torch
import torchvision.models as models

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    # If CUDA is available, use GPU
    device = torch.device("cuda")
    print("CUDA is available! You have a GPU.")
    print("GPU Device Name:", torch.cuda.get_device_name(0))  # Print GPU device name
else:
    # If CUDA is not available, use CPU
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")





# Enable PyTorch bottleneck profiler


# Example function to profile
def my_function():
    # Load a pretrained ResNet model
    model = models.resnet18(pretrained=True)
    model.eval()

    # Example input tensor (batch size 1, 3 channels, 224x224 image)
    input_tensor = torch.randn(1, 3, 224, 224)

    # Forward pass through the model
    output = model(input_tensor)



# Profile using torch.autograd.profiler
with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
    my_function()

print(prof)
