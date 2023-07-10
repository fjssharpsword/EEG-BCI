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
import mne
import pandas as pd
"""
Dataset: CHB-MIT, https://physionet.org/content/chbmit/1.0.0/
"""

def plot_seizure_lead_MNE():

    #loadig two cases
    ch_com = ['C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8', 'FP1-F3', 'FP1-F7', \
              'FP2-F4', 'FP2-F8', 'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P8-O2', 'T7-P7', 'T8-P8']
    
    pA = Patient(id=1) #excluding 12
    clipsA = pA.get_seizure_clips(ch_com)
    sfA = pA.get_sampling_rate()
    pA.close_files()#release files

    #Creating MNE-Python data structures from scratch
    info = mne.create_info(ch_com, sfreq=sfA)#ch_types='eeg'
    eegA, lblA = np.transpose(clipsA[1][0],(1,0)), clipsA[1][1]
    raw = mne.io.RawArray(eegA, info)
    ids = np.where(np.diff(lblA != 0))[0] + 1
    raw = raw.crop(ids[0]/sfA, ids[1]/sfA)
    raw.plot(show_scrollbars=False, show_scalebars=False)
    plt.savefig('/data/pycode/EEG-BCI/CHB-MIT/imgs/spl_prob.png', dpi=300, bbox_inches='tight')


def plot_seizure_lead():

    #loadig two cases
    ch_com = np.array(['C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8', 'FP1-F3', 'FP1-F7', \
                       'FP2-F4', 'FP2-F8', 'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P8-O2', 'T7-P7', 'T8-P8'])

    pA = Patient(id=1) #initiating case
    clipsA = pA.get_seizure_clips(ch_com)
    sfA = int(pA.get_sampling_rate())
    pA.close_files()#release files
    eegA, lblA = clipsA[1][0], clipsA[1][1]
    ids = np.where(np.diff(lblA != 0))[0] + 1
    eegA = eegA[ids[0]:ids[1], :] #seisure range
    eegA = eegA[0:10*sfA, :] #0-10 seconds

    pB = Patient(id=2) #initiating case
    clipsB = pB.get_seizure_clips(ch_com)
    sfB = int(pB.get_sampling_rate())
    pB.close_files()#release files
    eegB, lblB = clipsB[1][0], clipsB[1][1]
    ids = np.where(np.diff(lblB != 0))[0] + 1
    eegB = eegB[ids[0]:ids[1], :] #seisure range
    eegB = eegB[0:10*sfB, :] #0-10 seconds
  
    #plotting eeg shape 
    fig, ax = plt.subplots(1,2, constrained_layout=True,figsize=(18,18))
    ax[0].set_title('Patient A')
    ax[0].set_xticks([0,2,4,6,8,10])
    ax[0].set_xlabel('Times(s)')
    offset = np.arange(0.5, 18, 1)
    ax[0].set_yticks(offset.tolist(), ch_com)
    ax[0].set_ylim((0, 18))
    ax[1].set_title('Patient B')
    ax[1].set_xticks([0,2,4,6,8,10])
    ax[1].set_xlabel('Times(s)')
    ax[1].set_yticks([])
    ax[1].set_ylim((0, 18))

    for i in range(len(ch_com)):
        x = np.arange(0, eegA[:,i].shape[0], 1)/sfA
        y = (eegA[:,i]-eegA[:,i].min())/(eegA[:,i].max()-eegA[:,i].min())
        ax[0].plot(x, y+i)

        x = np.arange(0, eegB[:,i].shape[0], 1)/sfB
        y = (eegB[:,i]-eegB[:,i].min())/(eegB[:,i].max()-eegB[:,i].min())
        ax[1].plot(x, y+i)

    fig.savefig('/data/pycode/EEG-BCI/CHB-MIT/imgs/spl_prob.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_seizure_lead()
