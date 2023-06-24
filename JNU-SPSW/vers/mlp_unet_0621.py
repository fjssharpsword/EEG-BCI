import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):#smooth=1e-5
        super(DiceLoss, self).__init__()

        self.smooth = smooth
            
    def	forward(self, input, target):
        N = target.size(0)
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
    
        intersection = input_flat * target_flat
    
        loss = (2 * intersection.sum(1) + self.smooth) / (input_flat.sum(1) + target_flat.sum(1) + self.smooth)
        loss = 1 - loss.sum() / N
        #loss = loss.sum() / N
        return loss

class mlp_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mlp1 = nn.Linear(in_c, out_c)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.mlp2 = nn.Linear(out_c, out_c)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.mlp1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.mlp2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class build_unet(nn.Module):
    def __init__(self, in_c = 250):
        super().__init__()
        """ Encoder """
        self.e1 = mlp_block(in_c, 128)
        self.e2 = mlp_block(128, 64)
        self.e3 = mlp_block(64, 32)
        self.e4 = mlp_block(32, 16)
        """ Decoder """
        self.d1 = mlp_block(16, 32)
        self.d2 = mlp_block(32, 64)
        self.d3 = mlp_block(64, 128)
        self.d4 = mlp_block(128, in_c)
        self.conv = nn.Conv1d(2, 1, kernel_size=3, padding=1, bias=False)
        """ Classifier """
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ Encoder """
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        """ Decoder """
        d1 = self.d1(e4)

        d1 = self.conv(torch.cat([d1.unsqueeze(1), e3.unsqueeze(1)], dim=1)).squeeze(1)
        d2 = self.d2(d1)
        d2 = self.conv(torch.cat([d2.unsqueeze(1), e2.unsqueeze(1)], dim=1)).squeeze(1)
        d3 = self.d3(d2)
        d3 = self.conv(torch.cat([d3.unsqueeze(1), e1.unsqueeze(1)], dim=1)).squeeze(1)
        d4 = self.d4(d3)
        
        """ Classifier """
        y = self.sigmoid(d4)
        return y
    
#https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn((8, 250)).to(device)
    model = build_unet(in_c = 250).to(device)
    y = model(inputs)
    print(y.shape)