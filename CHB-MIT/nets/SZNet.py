import torch
import torch.nn as nn

class Spatial_layer(nn.Module):#spatial attention layer
    def __init__(self):
        super(Spatial_layer, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        identity = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)*identity
    
class Channel_layer(nn.Module):
    """Constructs a channel layer.
    Args:k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(Channel_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
    
class Temporal_layer(nn.Module):
    """Constructs a Temporal layer.
    Args:k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(Temporal_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = torch.mean(x, dim=1, keepdim=True).squeeze(1) #b*c*d*n -> b*d*n
        y = self.avg_pool(y) #b*d*n->b*d*1
        #b*d*1->b*1*d->b*d*1->b*1*d*1
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2).unsqueeze(1) 
        #on channels
        y = self.sigmoid(y)
        y = y.expand_as(x)
        x = x*y #redidual
        return x
    
class EEGSZNet(nn.Module):
    '''
    Args:
        num_electrodes (int): The number of electrodes. (defualt: :obj:`28`)
        num_T (int): The number of multi-scale 1D temporal kernels in the dynamic temporal layer, i.e., :math:`T` kernels in the paper. (defualt: :obj:`15`)
        num_S (int): The number of multi-scale 1D spatial kernels in the asymmetric spatial layer. (defualt: :obj:`15`)
        in_channels (int): The number of channels of the signal corresponding to each electrode. If the original signal is used as input, in_channels is set to 1; if the original signal is split into multiple sub-bands, in_channels is set to the number of bands. (defualt: :obj:`1`)
        hid_channels (int): The number of hidden nodes in the first fully connected layer. (defualt: :obj:`32`)
        num_classes (int): The number of classes to predict. (defualt: :obj:`2`)
        sampling_rate (int): The sampling rate of the EEG signals, i.e., :math:`f_s` in the paper. (defualt: :obj:`128`)
        dropout (float): Probability of an element to be zeroed in the dropout layers. (defualt: :obj:`0.5`)
    '''
    def __init__(self,
                 num_electrodes: int = 28,
                 num_T: int = 15, 
                 num_S: int = 15, 
                 in_channels: int = 1,
                 hid_channels: int = 32,
                 num_classes: int = 2,
                 sampling_rate: int = 128,
                 dropout: float = 0.5):
        # input_size: 1 x EEG channel x datapoint
        super(EEGSZNet, self).__init__()
        self.num_electrodes = num_electrodes
        self.num_T = num_T
        self.num_S = num_S
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.sampling_rate = sampling_rate
        self.dropout = dropout

        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8
        # by setting the convolutional kernel being (1,lenght) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(in_channels, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)

        self.Sception1 = self.conv_block(num_T, num_S, (int(num_electrodes), 1), 1, int(self.pool * 0.25))
        self.Sception2 = self.conv_block(num_T, num_S, (int(num_electrodes * 0.5), 1), (int(num_electrodes * 0.5), 1),
                                         int(self.pool * 0.25))
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_s = nn.BatchNorm2d(num_S)
        self.BN_fusion = nn.BatchNorm2d(num_S)

        self.fc = nn.Sequential(nn.Linear(num_S, hid_channels), nn.ReLU(), nn.Dropout(dropout),
                                nn.Linear(hid_channels, num_classes))
        #spatial attention
        self.att_layer = Temporal_layer() #Channel_layer() #Spatial_layer()

    def conv_block(self, in_channels: int, out_channels: int, kernel: int, stride: int, pool_kernel: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride),
            nn.LeakyReLU(), nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_kernel)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 1, 18, 512]`. Here, :obj:`n` corresponds to the batch size, :obj:`1` corresponds to number of channels for convolution, :obj:`28` corresponds to :obj:`num_electrodes`, and :obj:`512` corresponds to the input dimension for each electrode.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = x.unsqueeze(1)
        y = self.Tception1(x)
        y = self.att_layer(y) #attention layer
        out = y
        y = self.Tception2(x)
        y = self.att_layer(y) #attention layer
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        y = self.att_layer(y) #attention layer
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        z = self.Sception1(out)
        out_ = z
        z = self.Sception2(out)
        out_ = torch.cat((out_, z), dim=2)
        out = self.BN_s(out_)
        out = self.fusion_layer(out)
        out = self.BN_fusion(out)
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)
        out = self.fc(out)
        return out
    
if __name__ == "__main__":
    #Ding, Yi, et al. "Tsception: Capturing temporal dynamics and spatial asymmetry from EEG for emotion recognition." IEEE Transactions on Affective Computing (2022).
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    x = torch.rand(8, 18, 512).to(device)
    model = EEGSZNet(num_electrodes=18, num_classes=2).to(device)
    out = model(x)
    print(out.shape)