import torch
import torch.nn as nn
from utils.dataloader import loaders
from utils.model import unet
import torch.optim as optim
from skimage.metrics import peak_signal_noise_ratio as psnrcalc
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import wandb 
torch.manual_seed(4)
import argparse
parser = argparse.ArgumentParser(description='Input hyperparameters')
parser.add_argument('--k',metavar='API Key', type=str, help='Enter the API Key')
parser.add_argument('-num_epochs',metavar='Number of Epochs', type=int, help='Enter the number of epochs')
parser.add_argument('-lr',metavar='Learning Rate', type=float, help='Enter the learning rate')

args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = unet().to(device)
optimizer = optim.Adam(model.parameters(),lr=args.lr,betas=(0.9,0.999))

num_epochs = args.num_epochs

total_step = len(loaders['train'])

def validate_and_calculate_psnr(val_loader, model, device):
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images,labels = batch['image'].to(device), batch['target'].to(device)
            outputs = model(images)
            psnr = psnrcalc(outputs.cpu().numpy(), labels.cpu().numpy())
            total_psnr += psnr.item()
    avg_psnr = total_psnr / len(val_loader)
    return avg_psnr


        
        
def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs):
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_psnr_total = 0.0
        train_ssim_total = 0.0
        losslist = []
        totalLoss = 0.0
        for batch in train_dataloader:
            images = batch['image'].to(device)
            labels = batch['target'].to(device)
            outputs = model(images)
            loss = F.l1_loss(outputs, labels)
            wandb.log({"train_loss": loss.item(), "Epoch": num_epochs+1})
            
            # clear gradients for this training step
            optimizer.zero_grad()
            
            # backpropagation, compute gradients
            loss.backward()
            
            # apply gradients
            optimizer.step()
            totalLoss += loss.item()
            
            for i in range(len(outputs)):
                train_psnr_total += psnrcalc(labels[i].cpu().numpy(), outputs[i].detach().cpu().numpy())
           # train_ssim_total += ssim(labels[i].numpy(), outputs[i].detach().numpy(),wiz_size=11, channel_axis=3, multichannel=True)
        losslist.append(totalLoss/len(loaders['train']))
        train_psnr_avg = train_psnr_total / len(loaders['train'])
        #train_ssim_avg = train_ssim_total / len(loaders['train'])
        
        val_psnr_avg = validate_and_calculate_psnr(val_dataloader, model, device)
      # val_ssim_avg = val_ssim_total / len(loaders['val'])
        # Print training and validation PSNR and SSIM
        print(f'Epoch [{epoch+1}/{num_epochs}], Train PSNR: {train_psnr_avg:.2f}, Val PSNR: {val_psnr_avg:.2f}')
        if (val_psnr_avg > best_val_psnr_avg):
            best_val_psnr_avg = val_psnr_avg
           #best_val_ssim_avg = val_ssim_avg
            torch.save(model.state_dict(), 'weights.pth')
            print('Weights Saved')
        
wandb.login(key=args.k)
wandb.init(
    # set the wandb project where this run will be logged
    project="GANproject",
    
    # track hyperparameters and run metadata
    config={
    "num_epochs": args.num_epochs,
    "lr": args.lr,
    }
)        
if __name__ == '__main__':
    # Define and load datasets and dataloaders
    
    # Instantiate your model, define loss function and optimizer
    
    # Call the train_model function
    train_model(model, loaders['train'], loaders['val'], optimizer, num_epochs)
        

