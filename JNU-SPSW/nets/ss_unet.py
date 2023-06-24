import torch
import numpy as np
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

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool1d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p
    
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)

        # input is CHW
        diff = skip.size()[2] - x.size()[2]

        x = F.pad(x, [diff // 2, diff - diff // 2])
    
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class singular_spectrum_operator(nn.Module):
    def __init__(self, L=30, Ip=2):
        super(singular_spectrum_operator, self).__init__()
        self.L = L #window length of Hankel matrix
        self.Ip = Ip #epochs of power iteration
        self.sigmoid = nn.Sigmoid()
    
    def _batch_power_iteration(self, W):
        """
        power iteration for max_singular_value
        """
        device = W.get_device()
        v = torch.FloatTensor(W.size(0), W.size(2), 1).normal_(0, 1).to(device)
        W_s = torch.bmm(W.permute(0, 2, 1), W)
        for _ in range(self.Ip):
            v_t = v
            v = torch.bmm(W_s, v_t)
            v_norm = torch.norm(v.squeeze(), dim=1).unsqueeze(-1).unsqueeze(-1)
            v_norm = v_norm.expand_as(v)
            v = torch.div(v, v_norm)

        u = torch.bmm(W, v)
        return u, v #left vector, right vector
    
    def _batch_Hankel_matrix(self, W):
        W = W.squeeze(1) #B*N
        B, N = W.shape
        K = N - self.L + 1
        device = W.get_device()

        W=W.detach().cpu().numpy()
        W = np.array([W[:, i:i+K] for i in range(self.L)]) #B*N -> L*B*K
        W = W.transpose(1, 0, 2) #L*B*K -> B*L*K
        W = torch.Tensor(W).to(device)
        return W
    
    #https://github.com/johannfaouzi/pyts/blob/main/pyts/decomposition/ssa.py
    def _batch_transformer(self, X):
        B, _, N = X.shape
        device = X.get_device()
     
        W_hk = self._batch_Hankel_matrix(X) #B*1*N->B*L*K
        W_ss = torch.bmm(W_hk, W_hk.permute(0,2,1)) #B*L*K->B*L*L
        u, v = self._batch_power_iteration(W_ss)
        W_ss = torch.bmm(u, v.permute(0, 2, 1))#B* L* L
        W_ss = torch.bmm(W_ss, W_hk) #B*L*L->B*L*K

        K = N - self.L + 1
        first_row = [(0, col) for col in range(K)]
        last_col = [(row, K-1) for row in range(1, self.L)]
        indices = first_row + last_col
        X_hat = torch.empty((B, N))
        for i in range(B):
            for (j, k) in indices:
                X_hat[i, j + k] = W_ss[i, j, k]

        X_hat = torch.FloatTensor(X_hat).unsqueeze(1).to(device)
        return X_hat
    
    def _power_iteration(self, W):
        """
        power iteration for max_singular_value
        """
        device = W.get_device()
        v = torch.FloatTensor(W.size(1), 1).normal_(0, 1).to(device)
        W_s = torch.matmul(W.T, W)
        for _ in range(self.Ip):
            v_t = v
            v = torch.matmul(W_s, v_t)
            v = v/torch.norm(v)

        u = torch.matmul(W, v)#left vector, right vector
        W_hat = torch.matmul(u, v.T)
        return W_hat.unsqueeze(1)

    def forward(self, x):

        x_ss, _ = torch.max(x, dim=1, keepdim=True)# B*1*N
        #x_ss = self._batch_transformer(x_ss)
        x_ss = self._power_iteration(x_ss.squeeze(1))
        x_ss = self.sigmoid(x_ss)*x

        return x_ss

class build_unet(nn.Module):
    def __init__(self, in_ch =1, n_classes=1):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(in_ch, 16)
        self.e2 = encoder_block(16, 32)
        self.e3 = encoder_block(32, 64)
        self.e4 = encoder_block(64, 128)
        """ Bottleneck """
        self.b = conv_block(128, 256)
        """ Decoder """
        self.d1 = decoder_block(256, 128)
        self.d2 = decoder_block(128, 64)
        self.d3 = decoder_block(64, 32)
        self.d4 = decoder_block(32, 16)
        """ Classifier """
        self.outputs = nn.Conv1d(16, n_classes, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        #singular-spectrum operator
        self.ss_layer = singular_spectrum_operator()

    def forward(self, inputs):
        """ Encoder """
        inputs = self.ss_layer(inputs)
        s1, p1 = self.e1(inputs)
        
        p1 = self.ss_layer(p1)
        s2, p2 = self.e2(p1)

        p2 = self.ss_layer(p2)
        s3, p3 = self.e3(p2)

        p3 = self.ss_layer(p3)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        
        """ Classifier """
        outputs = self.outputs(d4)
        outputs = self.sigmoid(outputs)
        return outputs
    
#https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    inputs = torch.randn((8, 1, 250)).to(device)
    model = build_unet(n_classes=1).to(device)
    y = model(inputs)
    print(y.shape)