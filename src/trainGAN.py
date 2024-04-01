import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from utils.GAN import Generator,Discriminator
from utils.dataloader import loaders
from skimage.metrics import peak_signal_noise_ratio as psnrcalc

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
torch.manual_seed(4)


def show_images(images, title=None, nrow=5):
    """
    Utility function for showing images with matplotlib
    """
    images = torchvision.utils.make_grid(images, nrow=nrow)
    np_images = images.numpy()
    plt.figure(figsize=(20, 10))
    plt.imshow(np.transpose(np_images, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
def validate_and_calculate_psnr(val_loader, generator, device):
    generator.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images,labels = batch['images'].to(device), batch['targets'].to(device)
            outputs = generator(images)
            psnr = psnrcalc(outputs.cpu().numpy(), labels.cpu().numpy())
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(val_loader)
    return avg_psnr

def train_GAN(generator, discriminator, train_loader, val_loader, device, num_epochs):
    # Losses & optimizers
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    highest_psnr = 0.0
    for epoch in range(num_epochs):
        t=0
        generator.train()
        discriminator.train()
        train_psnr_total = 0.0
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['target'].to(device)
            valid = torch.ones((labels.size(0), 1), device=device, requires_grad=False)
            fake = torch.zeros((labels.size(0), 1), device=device, requires_grad=False)

            # Train Generator
            optimizer_G.zero_grad()
            generated_imgs = generator(images)
            g_loss = adversarial_loss(discriminator(generated_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(labels), valid)
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            epoch_loss_g += g_loss.item()
            epoch_loss_d += d_loss.item()
            t=t+1
            if t==1:
                show_images(images.cpu(), title="Input Images", nrow=5)
                show_images(generated_imgs.cpu(), title="Generated Images", nrow=5)
                show_images(labels.cpu(), title="Target Images", nrow=5)

        avg_psnr = validate_and_calculate_psnr(val_loader, generator, device)
        print(f"Epoch {epoch + 1}, G_loss: {epoch_loss_g:.4f}, D_loss: {epoch_loss_d:.4f}, Avg PSNR: {avg_psnr:.2f} dB")
        if avg_psnr > highest_psnr:
            highest_psnr = avg_psnr
            torch.save(generator.state_dict(), 'best_generator_model.pth')
            print(f"Saved better generator model with PSNR: {highest_psnr:.2f} dB")
            
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    train_loader = loaders['train']
    val_loader = loaders['val']
    num_epochs = 10
    train_GAN(generator, discriminator, train_loader, val_loader, device, num_epochs)