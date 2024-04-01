import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root_dir,transform=None):
        self.root_dir = root_dir
        self.image_filenames = os.listdir(root_dir)
        self.transform=transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = plt.imread(image_path)
        # Make a copy of the NumPy array to ensure it's writable

        # Assuming filename format is 'noisy_<index>.jpg' and 'ground_truth_<index>.jpg'
        if 'noisy' in image_name:
            target_name = image_name.replace('noisy', 'ground_truth')
            target_path = os.path.join(self.root_dir, target_name)
            target = plt.imread(target_path)
        elif 'ground_truth' in image_name:
            target_name = image_name.replace('ground_truth', 'noisy')
            target_path = os.path.join(self.root_dir, target_name)
            target = plt.imread(target_path)
    
        else:
            raise ValueError("Invalid filename format")
        
        image = np.copy(image)
        target = np.copy(target)
        
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        return {'image': image, 'target': target}

transform = transforms.Compose([
    transforms.ToTensor(),           # Convert images to tensors
])

#Train_dataset = CustomDataset(root_dir='C:/Users/aryes/Desktop/YEAR 2/mrm/GANproject/dataset/train_data', transform=transform)
#Test_dataset = CustomDataset(root_dir='C/Users/aryes/Desktop/YEAR 2/mrm/GANproject/dataset/test_data', transform=transform)
Train_dataset = CustomDataset(root_dir='kaggle/working/GANproject/dataset/train_data', transform=transform)
Test_dataset = CustomDataset(root_dir='kaggle/working/GANproject/dataset/test_data', transform=transform)
        
    
     
