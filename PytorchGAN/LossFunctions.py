import torch
import torch.nn.functional as F

# Define the loss functions
def generator_loss(fake_output):
    # The generator tries to minimize the binary cross entropy loss
    # with the target labels being all ones (indicating real images)
    target = torch.ones_like(fake_output)
    return F.binary_cross_entropy_with_logits(fake_output, target)

def discriminator_loss(real_output, fake_output):
    # The discriminator tries to distinguish between real and fake images
    # so it minimizes the sum of binary cross entropy losses for both
    # real and fake images
    real_target = torch.ones_like(real_output)
    fake_target = torch.zeros_like(fake_output)
    real_loss = F.binary_cross_entropy_with_logits(real_output, real_target)
    fake_loss = F.binary_cross_entropy_with_logits(fake_output, fake_target)
    return real_loss + fake_loss
