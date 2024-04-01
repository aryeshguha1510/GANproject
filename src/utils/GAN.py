import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.encoder1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(32, 64, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.encoder4=   nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.encoder5=   nn.Conv2d(256, 512, 3, stride=1, padding=1)
        
        self.decoder1 = nn.Conv2d(512, 256, 3, stride=1,padding=1)  # b, 16, 5, 5
        self.decoder2 =   nn.Conv2d(256, 128, 3, stride=1, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.Conv2d(128, 64, 3, stride=1, padding=1)  # b, 1, 28, 28
        self.decoder4 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(32, 3, 3, stride=1, padding=1)
        
        self.tan = nn.Tanh()
        # self.soft = nn.Softmax(dim =1)

    def forward(self, x):

        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        t1 = out
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))
        t2 = out
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))
        t3 = out
        out = F.relu(F.max_pool2d(self.encoder4(out),2,2))
        t4 = out
        out = F.relu(F.max_pool2d(self.encoder5(out),2,2))
        
        # t2 = out
        out = F.relu(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape,t4.shape)
        out = torch.add(out,t4)
        out = F.relu(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)
        out = F.relu(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        out = F.relu(F.interpolate(self.decoder4(out),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        # print(out.shape)
        
        # out = self.soft(out)
        return self.tan(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.3),
            # Flatten before going to linear layer
            nn.Flatten(),
            # Adjust the following layer to match the output of the last Conv2d layer
            nn.Linear(64 * 78 * 58, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.model(x)
        return x