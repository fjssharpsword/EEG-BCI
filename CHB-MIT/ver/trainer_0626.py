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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score, cohen_kappa_score
from tensorboardX import SummaryWriter
#self-defined
from nets.CNN import EEG1DConvNet

def train_epoch(model, dataloader, loss_fn, optimizer, device):

    tr_loss = []
    tr_acc = 0.0
    gt_lbl = torch.FloatTensor()
    model.train()
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        optimizer.zero_grad()
        var_out = model(var_eeg)
        loss = loss_fn(var_out,var_lbl)
        loss.backward()
        optimizer.step()
        tr_loss.append(loss.item())
        _, var_prd = torch.max(var_out.data, 1)
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        tr_acc += (var_prd == var_lbl).sum().item()

    tr_loss = np.mean(tr_loss)
    tr_acc = tr_acc/len(gt_lbl)

    return tr_loss, tr_acc

def eval_epoch(model, dataloader, loss_fn, device):

    te_loss = []
    gt_lbl = torch.FloatTensor()
    pr_lbl = torch.FloatTensor()
    model.eval()
    for eegs, lbls in dataloader:
        var_eeg = eegs.to(device)
        var_lbl = lbls.to(device)
        var_out = model(var_eeg)
        loss = loss_fn(var_out,var_lbl)
        te_loss.append(loss.item())
        gt_lbl = torch.cat((gt_lbl, lbls), 0)
        pr_lbl = torch.cat((pr_lbl, var_out.data.cpu()), 0)

    te_loss = np.mean(te_loss)
    pr_prb, pr_lbl = torch.max(pr_lbl, 1) #0-1
    te_auc = roc_auc_score(gt_lbl.numpy(), pr_prb.numpy()) #probability
    te_kappa = cohen_kappa_score(gt_lbl.numpy(), pr_lbl.numpy())
    te_acc = accuracy_score(gt_lbl.numpy(), pr_lbl.numpy())
    #te_acc = (pr_lbl == gt_lbl).sum()/len(gt_lbl)
    te_f1 = f1_score(gt_lbl.numpy(), pr_lbl.numpy(), average='weighted')
    tn, fp, fn, tp = confusion_matrix(gt_lbl.numpy(), pr_lbl.numpy()).ravel()
    te_sen = tp /(tp+fn)
    te_spe = tn /(tn+fp)

    pr_dict = {'loss': te_loss, 'Acc':te_acc, 'Sen':te_sen, 'Spe':te_spe, 'AUC':te_auc, 'F1':te_f1, 'Kappa':te_kappa}
    return pr_dict

def Train_Eval(PATH_TO_DST_ROOT):

    print('********************Build model********************')
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    
    model = EEG1DConvNet(in_ch = 18, num_classes=2).to(device)  
    optimizer_model = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.CrossEntropyLoss()
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    #log_writer = SummaryWriter('/data/tmpexec/tb_log')
   
    print('********************Train and validation********************')
    X, y = np.load(PATH_TO_DST_ROOT+'eeg_kfold.npy'), np.load(PATH_TO_DST_ROOT+'lbl_kfold.npy')
    dataset = TensorDataset(torch.FloatTensor(X).permute(0,2,1), torch.LongTensor(y))
    kf_set = KFold(n_splits=10,shuffle=True).split(X, y)

    cv_pr = {'Acc':[], 'Sen':[], 'Spe':[], 'AUC':[], 'F1':[], 'Kappa':[]}
    for f_id, (tr_idx, te_idx) in enumerate(kf_set):
        print('\n Fold {} train and validation.'.format(f_id + 1))
        
        best_pr = {'Acc':0.0, 'Sen':0.0, 'Spe':0.0, 'AUC':0.0, 'F1':0.0, 'Kappa':0.0}
        for epoch in range(20):
            tr_sampler = SubsetRandomSampler(tr_idx)
            te_sampler = SubsetRandomSampler(te_idx)
            tr_dataloader = DataLoader(dataset, batch_size = 128, sampler=tr_sampler) 
            te_dataloader = DataLoader(dataset, batch_size = 128, sampler=te_sampler)
                
            tr_loss, tr_acc = train_epoch(model, tr_dataloader, criterion, optimizer_model, device)
            lr_scheduler_model.step()  #about lr and gamma
            e_pr = eval_epoch(model, te_dataloader, criterion, device)

            #log_writer.add_scalars('EEG/CHB-MIT/Loss', {'Train':tr_loss, 'Test':te_loss}, epoch+1)
            print('\n Train Epoch_{}: Loss={:.4f}, Accuracy={:.4f}'.format(epoch+1, tr_loss, tr_acc))
            print('\n Validation Epoch_{}: Acc={:.4f}, AUC={:.4f}, F1={:.4f}, Kappa={:.4f}'.format(epoch+1, e_pr['Acc'], e_pr['AUC'], e_pr['F1'], e_pr['Kappa']))

            if e_pr['Acc'] > best_pr['Acc']:
                best_pr = e_pr

        cv_pr['Acc'].append(best_pr['Acc'])
        cv_pr['Sen'].append(best_pr['Sen'])
        cv_pr['Spe'].append(best_pr['Spe'])
        cv_pr['AUC'].append(best_pr['AUC'])
        cv_pr['F1'].append(best_pr['F1'])
        cv_pr['Kappa'].append(best_pr['Kappa'])

        print('\n Fold_{}: Acc={:.2f}, AUC={:.2f}, F1={:.2f}, Kappa={:.2f}'.format(f_id + 1, \
                                                                                   best_pr['Acc']*100, best_pr['AUC']*100, best_pr['F1']*100, best_pr['Kappa']*100))

    print('\n Average performance: Accuracy={:.2f}+/-{:.2f},\
                                   Sensitivity={:.2f}+/-{:.2f},\
                                   Specificity={:.2f}+/-{:.2f}, \
                                   AUROC={:.2f}+/-{:.2f}, \
                                   F1 score={:.2f}+/-{:.2f}, \
                                   Kappa={:.2f}+/-{:.2f}'.format(\
                                   np.mean(cv_pr['Acc'])*100, np.std(cv_pr['Acc'])*100,\
                                   np.mean(cv_pr['Sen'])*100, np.std(cv_pr['Sen'])*100,\
                                   np.mean(cv_pr['Spe'])*100, np.std(cv_pr['Spe'])*100,\
                                   np.mean(cv_pr['AUC'])*100, np.std(cv_pr['AUC'])*100,\
                                   np.mean(cv_pr['F1'])*100, np.std(cv_pr['F1'])*100,\
                                   np.mean(cv_pr['Kappa'])*100, np.std(cv_pr['Kappa'])*100))

def main():
    PATH_TO_DST_ROOT = '/data/pycode/EEG-BCI/CHB-MIT/dsts/'
    Train_Eval(PATH_TO_DST_ROOT)

if __name__ == "__main__":
    main()
    #nohup python3 -u trainer.py >> /data/tmpexec/tb_log/fi_unet.log 2>&1 &