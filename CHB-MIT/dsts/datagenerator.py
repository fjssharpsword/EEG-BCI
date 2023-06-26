import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from inter_patient import Patient
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pywt

"""
Dataset: CHB-MIT, https://physionet.org/content/chbmit/1.0.0/
"""

def inter_patients_CV(PATH_TO_DST_ROOT, seconds=2, neg_rate=2):

    patient_ids = [id for id in range(1, 25)]
    patient_ids.remove(12) # delete patient 12 with different channels.
    random.shuffle(patient_ids)

    #obtain channel names
    ch_com = np.array(['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', \
             'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ', 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'])
    for p_id in patient_ids:
        p = Patient(p_id)
        p_ch_name = p.get_channel_names()
        ch_com = np.intersect1d(ch_com, p_ch_name)
        p.close_files()#release files

    #datasets
    sei_data, sei_idx = [], []#positive samples
    non_sei_data, non_sei_idx = [], []#negative samples
    for p_id in patient_ids:
        win_len = int(p.get_sampling_rate() * seconds)
        p = Patient(p_id)
        data = p.get_eeg_data(ch_com)

        for sei_seg in p._seizure_intervals:
            num = int((sei_seg[1]-sei_seg[0])/win_len)
            pos = sei_seg[0]
            for i in range(num):
                pos = sei_seg[0]+i*win_len
                sei_data.append(data[pos:pos+win_len])
                sei_idx.append(1)

        for i in range(len(p._seizure_intervals)-1):
            num = int((p._seizure_intervals[i+1][0]-p._seizure_intervals[i][1])/win_len)
            for j in range(num):
                pos = p._seizure_intervals[i][1]+j*win_len
                non_sei_data.append(data[pos:pos+win_len])
                non_sei_idx.append(0)
                if len(non_sei_data) >= len(sei_data)*neg_rate: 
                    break

        p.close_files()#release files

    X = np.array(sei_data + non_sei_data)
    y = np.array(sei_idx + non_sei_idx)  
    np.save(PATH_TO_DST_ROOT+'eeg_kfold.npy', X)
    np.save(PATH_TO_DST_ROOT+'lbl_kfold.npy', y)

if __name__ == "__main__":
    #for k-fold cross validation
    PATH_TO_DST_ROOT = '/data/pycode/EEG-BCI/CHB-MIT/dsts/'
    inter_patients_CV(PATH_TO_DST_ROOT, seconds=2, neg_rate=2)
    #nohup python3 datagenerator.py > /data/tmpexec/tb_log/datagenerator.log 2>&1 &