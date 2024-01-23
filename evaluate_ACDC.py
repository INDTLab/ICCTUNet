from ICCTUNet_ACDC import ICCTUNet
from medpy import metric
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random
import time
from PIL import Image
import h5py
from functools import partial
from  swin_transformer import PatchMerging,SwinTransformerBlock,window_partition,Mlp,window_reverse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import argparse
import os.path as osp
parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', type=int, default=0,help="id of cuda device,default:0")
parser.add_argument('--num_epochs', type=int, default=450,help="training epochs")
parser.add_argument('--batch_size', type=int, default=4,help="train batch_size")
parser.add_argument('--drop_path', type=float, default=0.0,help="drop path rate")
parser.add_argument('--lr', type=float, default=5e-4,help="learning rate")
config = parser.parse_args()

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

device = torch.device(f'cuda:{config.cuda_id}')



def val_data_aug(image,label,re_size=None):
    if re_size:
        image = TF.resize(image,(re_size,re_size))
        label = TF.resize(label,(re_size,re_size),interpolation=TF.InterpolationMode.NEAREST)
    return image,label

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        #hd95 = metric.binary.hd95(pred, gt)
        return dice#, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1#, 0
    else:
        return 0#, 0


class ACDC_test(Dataset):
    def __init__(self,data_list,re_size=None):
        super().__init__()
        self.data_list = data_list
        self.re_size = re_size
    def __getitem__(self,index):
        file = self.data_list[index]
        data = np.load(file)
        image, label = data['image'], data['label']

        image = torch.tensor(image).permute(2,0,1).contiguous()
        label = torch.tensor(label).permute(2,0,1).contiguous()

        if self.re_size:
            image,label = val_data_aug(image,label,self.re_size)
        sample = {'image': image, 'label': label}
        return sample
    def __len__(self):
        return len(self.data_list)

test_path = '../project_TransUNet/data/ACDC/ACDC_testset/'
test_list = os.listdir(test_path)
test_list = [os.path.join(test_path,file) for file in test_list]
test_dataset = ACDC_test(test_list,224)
test_loader = DataLoader(test_dataset,batch_size=1,shuffle=False)

ICCTUNet = ICCTUNet().to(device)
num_class=4
dice_lst = []
trans_dice_lst = []
fuse_dice_lst = []

valdice_lst = []
valtrans_dice_lst = []
valfuse_dice_lst = []

valloss_lst = []
valtrans_loss_lst= []
valfuse_loss_lst= []


def evaluate(net):

    print(f'loading best val model to evaluate.')
    ckpoint = torch.load(f'ACDC_F_bestval.pth')
    net.load_state_dict(ckpoint['net'])
    net.eval()
    test_dice_list = 0.0
    trans_dice_list = 0.0
    fuse_dice_list = 0.0
    with torch.no_grad():
        for data in test_loader:
            metric_lst = []
            trans_metric_lst = []
            fuse_metric_lst = []
            image,label = data['image'][0],data['label'][0]
            label = label.detach().numpy()
            prediction = np.zeros_like(label)
            trans_prediction = np.zeros_like(label)
            fuse_prediction = np.zeros_like(label)
            for index in range(image.shape[0]):
                slice_ = image[index,:,:]
                input_tensor = slice_.unsqueeze(0).unsqueeze(0).float().to(device)
                pred,swin_pred,fuse_pred = net(input_tensor)
                out = torch.argmax(torch.softmax(pred, dim=1), dim=1).squeeze(0)
                trans_out = torch.argmax(torch.softmax(swin_pred, dim=1), dim=1).squeeze(0)
                fuse_out = torch.argmax(torch.softmax(fuse_pred, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                trans_out = trans_out.cpu().detach().numpy()
                fuse_out = fuse_out.cpu().detach().numpy()
                prediction[index] = out
                trans_prediction[index] = trans_out
                fuse_prediction[index] = fuse_out
            for i in range(1,num_class): 
                metric_lst.append(calculate_metric_percase(prediction==i,label==i))
                trans_metric_lst.append(calculate_metric_percase(trans_prediction==i,label==i))
                fuse_metric_lst.append(calculate_metric_percase(fuse_prediction==i,label==i))

            test_dice_list += np.array(metric_lst)
            trans_dice_list += np.array(trans_metric_lst)
            fuse_dice_list += np.array(fuse_metric_lst)
    test_dice_list /= len(test_dataset)
    trans_dice_list /= len(test_dataset)
    fuse_dice_list /= len(test_dataset)
    print(f'mean dice on test set: {np.mean(test_dice_list)} trans dice{np.mean(trans_dice_list)} fuse dice{np.mean(fuse_dice_list)}')
    print(f'dice per class: {test_dice_list} trans dice{trans_dice_list} fuse dice{fuse_dice_list}')


evaluate(ICCTUNet)