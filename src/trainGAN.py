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
import wandb
torch.manual_seed(4)
import argparse
parser = argparse.ArgumentParser(description='Input hyperparameters')
parser.add_argument('--k',metavar='API Key', type=str, help='Enter the API Key')
parser.add_argument('-num_epochs',metavar='Number of Epochs', type=int, help='Enter the number of epochs')
parser.add_argument('-lr1',metavar='Learning Rate', type=float, help='Enter the learning rate')
parser.add_argument('-lr2',metavar='Learning Rate', type=float, help='Enter the learning rate')

args = parser.parse_args()

def validate_and_calculate_psnr(val_loader, generator, device):
    generator.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images,labels = batch['image'].to(device), batch['target'].to(device)
            outputs = generator(images)
            psnr = psnrcalc(outputs.cpu().numpy(), labels.cpu().numpy())
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(val_loader)
    return avg_psnr

def train_GAN(generator, discriminator, train_loader, val_loader, device, num_epochs):
    # Losses & optimizers
    adversarial_loss = nn.BCELoss()
    pixelwise_loss = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr1)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr2)
    highest_psnr = 0.0
    for epoch in range(num_epochs):
        
        generator.train()
        discriminator.train()
        train_psnr_total = 0.0
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['target'].to(device)
            valid = torch.ones(labels.size(0), 1).to(device=device)
            fake = torch.zeros(labels.size(0), 1).to(device=device)

            # Train Generator
            optimizer_G.zero_grad()
            generated_imgs = generator(images)
            g_loss = 0.001 * adversarial_loss(discriminator(generated_imgs), valid.squeeze(1)) + pixelwise_loss(generated_imgs, labels)
            wandb.log({"Generator Loss": g_loss.item(), "Epoch": num_epochs+1})
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(labels), valid.squeeze(1))
            fake_loss = adversarial_loss(discriminator(generated_imgs.detach()), fake.squeeze(1)  )
            d_loss = (real_loss + fake_loss)
            wandb.log({"Discriminator Loss": d_loss.item(), "Epoch": num_epochs+1})
            d_loss.backward()
            optimizer_D.step()

            epoch_loss_g += g_loss.item()
            epoch_loss_d += d_loss.item()
            train_psnr_total += psnrcalc(labels.cpu().numpy(), generated_imgs.detach().cpu().numpy())
        train_psnr_avg = train_psnr_total / len(loaders['train'])
        avg_psnr = validate_and_calculate_psnr(val_loader, generator, device)
        print(f"Epoch {epoch + 1}, G_loss: {epoch_loss_g:.4f}, D_loss: {epoch_loss_d:.4f},Train PSNR: {train_psnr_avg:.2f}, Avg Val PSNR: {avg_psnr:.2f} dB")
        if avg_psnr > highest_psnr:
            highest_psnr = avg_psnr
            torch.save(generator.state_dict(),'best_model_weights.pth')
            print(f"Saved better generator model with PSNR: {highest_psnr:.2f} dB")
        torch.save(generator.state_dict(), 'last_model_weights.pth')
wandb.login(key=args.k)
wandb.init(
    # set the wandb project where this run will be logged
    project="GANproject",
    
    # track hyperparameters and run metadata
    config={
    "num_epochs": args.num_epochs,
    "lr1": args.lr1,
    "lr2": args.lr2,
    }
)            
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    train_loader = loaders['train']
    val_loader = loaders['val']
    num_epochs = args.num_epochs
    train_GAN(generator, discriminator, train_loader, val_loader, device, num_epochs)