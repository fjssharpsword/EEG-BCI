import torch.nn as nn
from torch.nn import functional as F
import torch

class SelfAttention_layer(nn.Module): 
    def __init__(self, in_ch=18, k=1):
        super(SelfAttention_layer, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        self.f = nn.Sequential(
            nn.Conv1d(self.in_ch, self.mid_ch, 1, 1),
            nn.BatchNorm1d(self.mid_ch),
            nn.ReLU())
        self.g = nn.Sequential(
            nn.Conv1d(self.in_ch, self.mid_ch, 1, 1),
            nn.BatchNorm1d(self.mid_ch),
            nn.ReLU())
        self.h = nn.Conv1d(self.in_ch, self.mid_ch, 1, 1)
        self.v = nn.Conv1d(self.mid_ch, self.out_ch, 1, 1)

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]: 
            conv.apply(weights_init)
        self.v.apply(constant_init)

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def forward(self, x):
        B, C, D = x.shape

        f_x = self.f(x).view(B, self.mid_ch, D)  # B * mid_ch * D
        g_x = self.g(x).view(B, self.mid_ch, D)  # B * mid_ch * D
        h_x = self.h(x).view(B, self.mid_ch, D)  # B * mid_ch * D

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * D * D
        attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * D * mid_ch
        z = z.permute(0, 2, 1).view(B, self.mid_ch, D)  # B * mid_ch * D

        z = self.v(z)
        x = torch.add(z, x) # z + x
        return x

## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass

class Spatial_layer(nn.Module):#spatial attention layer
    def __init__(self):
        super(Spatial_layer, self).__init__()

        self.conv1 = nn.Conv1d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*identity
    
class EEGSACNN(nn.Module):
    def __init__(self, in_ch = 22, num_classes=2):
        # We optimize dropout rate in a convolutional neural network.
        super(EEGSACNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool1d(kernel_size = 3)

        self.dropout = nn.Dropout(p=0.2) 

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size = 3)

        self.pool3 = nn.AdaptiveAvgPool2d((32,32))
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

        self.global_sa = SelfAttention_layer()
        self.local_sa = Spatial_layer()

    def forward(self, x):

        x = self.global_sa(x) #context

        x = self.conv1(x)
        x = self.local_sa(x) #spatial
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.local_sa(x) #spatial
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
class EEG2DConvNet(nn.Module):
    def __init__(self, in_ch = 22, num_classes=2):
        # We optimize dropout rate in a convolutional neural network.
        super(EEG2DConvNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=32, kernel_size=3, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size = 3)

        self.dropout = nn.Dropout(p=0.2) 

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size = 3)

        self.pool3 = nn.AdaptiveAvgPool2d((4,4))
        self.fc1 = nn.Linear(1024, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = self.pool3(x)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
    
if __name__ == "__main__":
    #Paper: Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J]. 
    # Journal of neural engineering, 2018, 15(5): 056013.
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    x = torch.rand(8, 18, 512).to(device)
    model = EEGSACNN(in_ch = 18, num_classes=2).to(device)
    out = model(x)
    print(out.shape)