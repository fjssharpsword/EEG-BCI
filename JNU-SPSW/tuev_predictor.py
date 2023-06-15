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
import mne
import pandas as pd
from tensorboardX import SummaryWriter
#self-defined
from dsts.tuev_spsw import build_dataset
from nets.utime import build_unet

class SPSWInstance:
    def __init__(self, id, down_fq=250):

        #parse edf and csv
        edf_path =  "/data/fjsdata/EEG/JNU-SPSW/files2/tuev_spsw/" + id + '.edf'
        ann_path =  "/data/fjsdata/EEG/JNU-SPSW/files2/tuev_spsw/" + id + '.rec'
        raw_np = mne.io.read_raw_edf(edf_path, preload=True)
        #filter
        raw_np.filter(l_freq=1, h_freq=70)

        #downsampling
        sfreq = int(raw_np.info['sfreq']) 
        self.secs = int(raw_np.n_times/sfreq) #second
        if  sfreq != down_fq:
            raw_np.resample(down_fq, npad="auto") 
        self.down_fq = down_fq

        #get SPSW instance
        ann_dict = self._parse_annotation(ann_path)
        self.lead_dict = self._parse_EEGWave(raw_np, ann_dict, down_fq)

    def _parse_EEGWave(self, raw_np, ann_dict, down_fq):

        eeg_data = raw_np.get_data() #numpy data
        ch_names = raw_np.info['ch_names'] #electrodes
        
        lead_dict = {}
        for key in ann_dict.keys(): 
            bi_ch = key.split("-", 1) #two electrodes
            F_idx, S_idx = -1, -1
            for i, ele in enumerate(ch_names):
                if bi_ch[0] in ele: F_idx = i
                if bi_ch[1] in ele: S_idx = i
            if F_idx !=-1 and S_idx !=-1:
                ch_eeg = eeg_data[F_idx,:] - eeg_data[S_idx,:] #electrode differences
                ch_eeg = (ch_eeg - np.min(ch_eeg))/(np.max(ch_eeg)-np.min(ch_eeg)) #0-1normalization
                ch_lbl = np.zeros(len(ch_eeg))
                #labeling sampling points
                values = ann_dict[key]
                for val in values:
                    st, ed = math.floor(val[0] * down_fq), math.ceil(val[1] * down_fq)
                    ch_lbl[st:ed] = 1
                
                lead_dict[key] = (ch_eeg, ch_lbl)
                
        return lead_dict

    def _parse_annotation(self, ann_path):
        montage = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ',\
                   'CZ-C4', 'C4-T4', 'T4-A2', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
        ann_dict = {}
        with open(ann_path, 'r') as ann_file:
            for line in ann_file.readlines():
                line = line.split(',')
                ch, st, ed, cl =  eval(line[0]), eval(line[1]), eval(line[2]), eval(line[3])
                ch_name = montage[ch]
                if ch_name in ann_dict.keys():
                    ann_dict[ch_name].append((st,ed))
                else:
                    ann_dict[ch_name] = [(st,ed)]

        return ann_dict

def TUEV_Prediction(CKPT_PATH):

    #loading model
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    model = build_unet(in_ch=1, n_classes=1).to(device)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained model: "+ CKPT_PATH)
    model.eval()#turn to evaluation mode

    ##loading dataset and prediction
    dir = "/data/fjsdata/EEG/JNU-SPSW/files2/tuev_spsw/"
    ids = []
    down_fq = 250 
    for _, _, files in os.walk(dir):
        for file in files:
            name = os.path.splitext(file)[0]
            if name not in ids:
                ids.append(name)
    
    for id in ids: #case
        spsw = SPSWInstance(id, down_fq)
        spsw_span = []
        for ch_name in spsw.lead_dict.keys(): #case->channel
            eeg, lbl = spsw.lead_dict[ch_name][0], spsw.lead_dict[ch_name][1]
            X, y = [], []
            for s in range(spsw.secs):
                X.append(eeg[s*down_fq:(s+1)*down_fq])
                y.append(lbl[s*down_fq:(s+1)*down_fq])
            #print('\r Sample number: {}'.format(len(y)))
            dataset = TensorDataset(torch.FloatTensor(np.array(X)).unsqueeze(1), torch.LongTensor(np.array(y)))
            dataloader = DataLoader(dataset=dataset, batch_size=512, shuffle=False, num_workers=1, pin_memory=True)

            #starting detector
            pr_lbl = torch.FloatTensor()
            for eegs, _ in dataloader:
                var_out = model(eegs.to(device))
                pr_lbl = torch.cat((pr_lbl, var_out.data.cpu()), 0)
            pr_lbl = torch.where(pr_lbl>0.5, 1, 0).squeeze()#.view(-1)
            for s in range(spsw.secs):
                #if pr_lbl[s].sum()>0 and y[s].sum()>0:
                if pr_lbl[s].sum() > 50: #a spsw contains 20+ points
                    idxs = np.where(np.diff(pr_lbl[s] != 0))[0] + 1
                    for p in range(0, len(idxs)-1, 2):
                        st, ed = s+idxs[p]/down_fq, s+idxs[p+1]/down_fq
                        spsw_span.append([ch_name, st, ed])
        #each id outputs a csv file
        spsw_span = pd.DataFrame(data=spsw_span, columns=['ch_name', 'start', 'end'])
        spsw_span.to_csv('/data/tmpexec/tb_log/tuev/pred_{}.csv'.format(id), index=False, encoding='utf-8')

def main():
    CKPT_PATH = '/data/pycode/EEG-BCI/JNU-SPSW/ckpts/utime_tf.pkl'
    TUEV_Prediction(CKPT_PATH)

if __name__ == "__main__":
    main()
    #nohup python3 -u tuev_predictor.py >> /data/tmpexec/tb_log/tuev_predictor.log 2>&1 &