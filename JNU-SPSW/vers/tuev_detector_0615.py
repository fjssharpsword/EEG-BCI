import os
import numpy as np
import math
from sklearn.model_selection import KFold
import torch
from torchvision import transforms
import torch.nn as nn
import torchvision
import random
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
#self-defined
from dsts.tuev_spsw import build_dataset
from nets.focal_ops import build_unet

def Detect_SPSW(CKPT_PATH):
    #loading dataset
    X, y = build_dataset(down_fq=250, seg_len=250) #time domain
    print('\r Sample number: {}'.format(len(y)))
    dataset = TensorDataset(torch.FloatTensor(X).unsqueeze(1), torch.LongTensor(y))
    dataloader = DataLoader(dataset=dataset, batch_size=512, shuffle=False, num_workers=1, pin_memory=True)

    #loading model
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model = build_unet(in_ch=1, n_classes=1).to(device)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained model: "+CKPT_PATH)
    model.eval()#turn to evaluation mode
    
    #starting detector
    gt_eeg = torch.FloatTensor()
    pr_prb = torch.FloatTensor()
    for eegs, _ in dataloader:
        var_out = model(eegs.to(device))
        pr_prb = torch.cat((pr_prb, var_out.data.cpu()), 0)
        gt_eeg = torch.cat((gt_eeg, eegs), 0)
    
    pr_prb = torch.where(pr_prb>0.5, 1, 0).squeeze()
    np.save('/data/pycode/EEG-BCI/JNU-SPSW/dsts/tuev_spsw_eeg.npy', gt_eeg.squeeze().numpy())
    np.save('/data/pycode/EEG-BCI/JNU-SPSW/dsts/tuev_spsw_lbl.npy', pr_prb.numpy())

def vis_prediction(PATH_TO_DST_ROOT):

    eegs, lbls = np.load(PATH_TO_DST_ROOT+'tuev_spsw_eeg.npy'), np.load(PATH_TO_DST_ROOT+'tuev_spsw_lbl.npy')
    fig, axes = plt.subplots(1,2, constrained_layout=True, figsize=(12,6))
    for i in range(2):   
        j = random.randint(0,len(lbls))
        eeg, lbl = eegs[j], lbls[j]
        x = [id for id in range(1, len(lbl)+1)]
        axes[i].plot(x, eeg, color = 'g')
        segs = np.where(np.diff(lbl != 0))[0] + 1
        axes[i].plot(segs[0], eeg[segs[0]], marker='^', color='r')
        axes[i].plot(segs[1], eeg[segs[1]], marker='v', color='r')
        axes[i].grid(b=True, ls=':')

    fig.savefig('/data/pycode/EEG-BCI/JNU-SPSW/imgs/tuev_pred.png', dpi=300, bbox_inches='tight') 

def main():
    CKPT_PATH = '/data/pycode/EEG-BCI/JNU-SPSW/ckpts/focal_ops.pkl'
    #Detect_SPSW(CKPT_PATH)
    PATH_TO_DST_ROOT = '/data/pycode/EEG-BCI/JNU-SPSW/dsts/'
    vis_prediction(PATH_TO_DST_ROOT)

if __name__ == "__main__":
    main()
    #nohup python3 -u detector.py >> /data/tmpexec/tb_log/detector.log 2>&1 &