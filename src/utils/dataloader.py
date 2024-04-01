import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.DatasetMaker import Train_dataset, Test_dataset

train_size=int(0.8 * len(Train_dataset))
val_size = len(Train_dataset) - train_size

train_sampler = SubsetRandomSampler(range(train_size))
val_sampler = SubsetRandomSampler(range(train_size, len(Train_dataset)))

loaders = {
    'train' : torch.utils.data.DataLoader(Train_dataset, 
                                          batch_size=8,
                                          sampler=train_sampler, 
                                          ),
    'val' : torch.utils.data.DataLoader(Train_dataset, 
                                          batch_size=8,
                                          sampler=val_sampler, 
                                          ),
    
    'test'  : torch.utils.data.DataLoader(Test_dataset, 
                                          batch_size=1, 
                                          shuffle=True, 
                                          ),
}

