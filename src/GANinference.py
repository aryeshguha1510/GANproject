import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from utils.GAN import Generator, Discriminator
from utils.dataloader import loaders
from skimage.metrics import peak_signal_noise_ratio as calcpsnr


def load_model(model_path, device):
    model = Generator().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def inference(test_loader, model, device):
    total_psnr = 0.0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch['image'].to(device), batch['target'].to(device)
            outputs = model(images)
            psnr = calcpsnr(outputs.cpu().numpy(), labels.cpu().numpy())
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(test_loader)
    print(f'Average PSNR: {avg_psnr:.2f} dB')
    
def visualize_sample(data_loader, model, device, num_images=3):
    batch = next(iter(data_loader))
    data, targets = batch['image'].to(device), batch['target'].to(device)
    with torch.no_grad():
        outputs = model(data)
    imgs = torch.cat([data[:num_images], outputs[:num_images], targets[:num_images]], dim=0)
    grid = make_grid(imgs, nrow=num_images)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.cpu().permute(1, 2, 0))
    plt.axis('off')
    plt.title("Original - Generated - Target")
    plt.show()
    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_path = 'C:/Users/aryes/Desktop/YEAR 2/mrm/GANproject/weights (4).pth'  # Update this path
    model = load_model(model_path, device)
    test_loader = loaders['test']
    inference(test_loader, model, device)
    visualize_sample(test_loader, model, device)