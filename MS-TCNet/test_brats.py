# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import os
import torch
import numpy as np
from utils.sliding_window_inference_for_mstcnet import sliding_window_inference

from networks.mstcnet_brats import mstcnet as mstcnet_brats
from utils.data_utils_brats import get_loader_brats

from utils.utils import resample_3d
import nibabel as nib

import argparse
from functools import partial

from trainer_brat import dice
from medpy.metric import binary
from monai.transforms import AsDiscrete,Activations
from monai.data import decollate_batch
from monai.metrics import DiceMetric,HausdorffDistanceMetric
from monai.utils.enums import MetricReduction

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/raid/user_dir/ay/data/BrainTumour/UnetRdata/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--a_min', default=0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=300, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=6, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--save_image', default='No', type=str, help=' Yes:save image as nii file')

def process_label(label):
    net = label == 2
    ed = label == 1
    et = label == 3
    ET=et
    TC=net+et
    WT=net+et+ed
    return ET,TC,WT

def hd(pred,gt):
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        return  hd95
    else:
        return 0

def main():
    args = parser.parse_args()
    args.test_mode = True
    test_loader = get_loader_brats(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    output_directory=args.pretrained_dir+"test/"
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if args.saved_checkpoint == 'torchscript':
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == 'ckpt':
        model =mstcnet_brats(
            in_channels=4,
            out_channels=3,
            img_size=(128, 128, 128),
            feature_size=32,
            norm_name=args.norm_name,
            res_block=True,
            dropout_rate=args.dropout_rate)        

        print(pretrained_pth)
        model_dict = torch.load(pretrained_pth)
        model.load_state_dict({k.replace('module.', ''): v for k, v in model_dict['state_dict'].items()})
    model.eval()
    model.to(device)

    model_inferer = partial(sliding_window_inference,
                            roi_size=[128,128,128],
                            sw_batch_size=4,
                            predictor=model,
                            overlap=args.infer_overlap)

    post_sigmoid = Activations(sigmoid=True)
    post_pred =  AsDiscrete(argmax=False, threshold=0.5)# AsDiscrete(threshold=0.5)#AsDiscrete(argmax=False, logit_thresh=0.5)
    dice_acc = DiceMetric(include_background=True,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)
    metric_hd = HausdorffDistanceMetric(include_background=True, reduction='mean', percentile=95)


    with torch.no_grad():

        dice_list_original_label_case = []
        dice_list_original_label_case1 = []
        dice_list_new_three_label_case = []
        dice_list_five_label_case = []     

        hd95_list_original_label_case = []
        hd95_list_original_label_case1 = []
        hd95_list_new_three_label_case = []
        hd95_list_five_label_case = [] 

        ED_1dice_list_sub = []
        NET_2dice_list_sub = []
        ET_3dice_list_sub = []

        ED_1dice_list_sub1 = []
        NET_2dice_list_sub1 = []
        ET_3dice_list_sub1 = []

        TCdice_list_sub = []
        WTdice_list_sub = []
        ETdice_list_sub = []

        ED_1hd95_list_sub = []
        NET_2hd95_list_sub = []
        ET_3hd95_list_sub = []

        ED_1hd95_list_sub1 = []
        NET_2hd95_list_sub1 = []
        ET_3hd95_list_sub1 = []

        TChd95_list_sub = []
        WThd95_list_sub = []
        EThd95_list_sub = []
        
        for i, batch in enumerate(test_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch['label_meta_dict']['affine'][0].numpy()
            # print("val_inputs",val_inputs.shape)
            # print("val_labels",val_labels.shape)
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))

            val_outputs =  model_inferer(val_inputs)

            # print("val_outputs",val_outputs.shape)
            target=val_labels
            logits=val_outputs
            val_labels_list = decollate_batch(target)
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(post_sigmoid(val_pred_tensor))for val_pred_tensor in val_outputs_list]
            # print(val_output_convert[0].shape)

            acc = dice_acc(y_pred=val_output_convert, y=val_labels_list)
            acc_list = acc.detach().cpu().numpy()

            print("acc",acc_list[0][0]," ",acc_list[0][1]," ",acc_list[0][2])
            print("acc_average",np.mean(acc_list[0]))

            dice_list_original_label_case.append(np.mean(acc_list[0]))
            ED_1dice_list_sub.append(acc_list[0][0])
            NET_2dice_list_sub.append(acc_list[0][1])
            ET_3dice_list_sub.append(acc_list[0][2])

            acc_hd = metric_hd(y_pred=val_output_convert, y=val_labels_list)
            acc_hd_list = acc_hd.detach().cpu().numpy()
            print("acc_hd95",acc_hd_list[0][0]," ",acc_hd_list[0][1]," ",acc_hd_list[0][2])
            print("acc_hd95_average",np.mean(acc_hd_list[0]))

            hd95_list_original_label_case.append(np.mean(acc_hd_list[0]))
            ED_1hd95_list_sub.append(acc_hd_list[0][0])
            NET_2hd95_list_sub.append(acc_hd_list[0][1])
            ET_3hd95_list_sub.append(acc_hd_list[0][2])


            val_outputs = torch.sigmoid(val_outputs)[0].detach().cpu().numpy()

            val_outputs = (val_outputs > 0.5).astype(np.int8)


            # print("val_outputs",val_outputs.shape)
            seg_out = np.zeros((val_outputs.shape[1], val_outputs.shape[2], val_outputs.shape[3]))
            seg_out[val_outputs[1] == 1] = 2#WT
            seg_out[val_outputs[0] == 1] = 1#TC
            seg_out[val_outputs[2] == 1] = 3#ET 

            val_labels=val_labels[0].detach().cpu().numpy()
            seg_label = np.zeros((val_labels.shape[1], val_labels.shape[2], val_labels.shape[3]))
            seg_label[val_labels[1] == 1] = 2
            seg_label[val_labels[0] == 1] = 1
            seg_label[val_labels[2] == 1] = 3

            dice_list_case=[]
            hd95_list_case=[]
            for i in range(1, 4):
                organ_Dice = dice(seg_out == i, seg_label == i)
                print(i,'dice',np.around(organ_Dice,8))#np.around(organ_Dice,6)
                if organ_Dice!=0:
                    dice_list_case.append(organ_Dice)
                if i==1:
                    ED_1dice_list_sub1.append(organ_Dice)
                elif i==2:
                    NET_2dice_list_sub1.append(organ_Dice)
                elif i==3:
                    ET_3dice_list_sub1.append(organ_Dice)

            mean_dice = np.mean(dice_list_case)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_original_label_case1.append(mean_dice)


            for i in range(1, 4):
                organ_hd = hd(seg_out == i, seg_label == i)
                print(i,'dice',np.around(organ_hd,8))#np.around(organ_hd,6) 
                if organ_hd!=0:
                    hd95_list_case.append(organ_hd)
                if i==1:
                    ED_1hd95_list_sub1.append(organ_hd)
                elif i==2:
                    NET_2hd95_list_sub1.append(organ_hd)
                elif i==3:
                    ET_3hd95_list_sub1.append(organ_hd)
            mean_hd95 = np.mean(hd95_list_case)
            print("Mean Organ HD95: {}".format(mean_hd95))
            hd95_list_original_label_case1.append(mean_hd95)


            
            label_et,label_tc,label_wt=process_label(seg_label)
            infer_et,infer_tc,infer_wt=process_label(seg_out)
            

           
            organ_HD95_TC=hd(infer_tc,label_tc)
            organ_HD95_WT=hd(infer_wt,label_wt)
            organ_HD95_ET=hd(infer_et,label_et)

            TChd95_list_sub.append(organ_HD95_TC)
            WThd95_list_sub.append(organ_HD95_WT)
            EThd95_list_sub.append(organ_HD95_ET)
            hd95_list_new_three_label_case.append((organ_HD95_TC+organ_HD95_WT+organ_HD95_ET)/3)

            hd95_list_case.append(organ_HD95_TC)
            hd95_list_case.append(organ_HD95_WT)         

            print("hd95_list_case",hd95_list_case)        
            hd95_list_five_label_case.append(np.mean(hd95_list_case))

            organ_Dice_ET = dice(infer_et, label_et)
            # print('ET dice',np.around(organ_Dice_ET,8))#np.around(organ_Dice,6) 

            organ_Dice_TC = dice(infer_tc, label_tc)
            print('TC dice',np.around(organ_Dice_TC,8))#np.around(organ_Dice,6) 

            organ_Dice_WT = dice(infer_wt, label_wt)
            print('WT dice',np.around(organ_Dice_WT,8))#np.around(organ_Dice,6) 
            
            TCdice_list_sub.append(organ_Dice_TC)
            WTdice_list_sub.append(organ_Dice_WT)
            ETdice_list_sub.append(organ_Dice_ET)
            dice_list_new_three_label_case.append((organ_Dice_TC+organ_Dice_ET+organ_Dice_WT)/3)

            dice_list_case.append(organ_Dice_TC)
            dice_list_case.append(organ_Dice_WT)
            
            print("dice_list_case",dice_list_case)
            dice_list_five_label_case.append(np.mean(dice_list_case))

            if args.save_image=="Yes":
                nib.save(nib.Nifti1Image(seg_out.astype(np.uint8), original_affine),
                        os.path.join(output_directory, "pred_"+img_name))

                # nib.save(nib.Nifti1Image(seg_label.astype(np.uint8), original_affine),
                #         os.path.join(output_directory, "label_"+img_name)) 

        print(" ")
        print("dice_list_original_label Overall Mean Dice: {}".format(np.mean(dice_list_original_label_case)),"total:",len(dice_list_original_label_case))
        print("dice_list_original_label1 Overall Mean Dice: {}".format(np.mean(dice_list_original_label_case1)),"total:",len(dice_list_original_label_case1))
        print("dice_list_new_three_label_case Overall Mean Dice: {}".format(np.mean(dice_list_new_three_label_case)),"total:",len(dice_list_new_three_label_case))
        print("dice_list_five_label_case Overall Mean Dice: {}".format(np.mean(dice_list_five_label_case)),"total:",len(dice_list_five_label_case))
        
        print(" ")   
        print("hd95_list_original_label Overall Mean hd95: {}".format(np.mean(hd95_list_original_label_case)),"total:",len(hd95_list_original_label_case))
        print("hd95_list_original_label1 Overall Mean hd95: {}".format(np.mean(hd95_list_original_label_case1)),"total:",len(hd95_list_original_label_case1))
        print("hd95_list_new_three_label_case Overall Mean hd95: {}".format(np.mean(hd95_list_new_three_label_case)),"total:",len(hd95_list_new_three_label_case))
        print("hd95_list_five_label_case Overall Mean hd95: {}".format(np.mean(hd95_list_five_label_case)),"total:",len(hd95_list_five_label_case))

        print(" ")
        print("ED Mean Dice: {}".format(np.mean(ED_1dice_list_sub)),"total:",len(ED_1dice_list_sub))
        print("NET Mean Dice: {}".format(np.mean(NET_2dice_list_sub)),"total:",len(NET_2dice_list_sub))
        print("ET Mean Dice: {}".format(np.mean(ET_3dice_list_sub)),"total:",len(ET_3dice_list_sub))

        print("ED Mean hd95: {}".format(np.mean(ED_1hd95_list_sub)),"total:",len(ED_1hd95_list_sub))
        print("NET Mean hd95: {}".format(np.mean(NET_2hd95_list_sub)),"total:",len(NET_2hd95_list_sub))
        print("ET Mean hd95: {}".format(np.mean(ET_3hd95_list_sub)),"total:",len(ET_3hd95_list_sub))
        print(" ")
        print("ED Mean Dice1: {}".format(np.mean(ED_1dice_list_sub1)),"total:",len(ED_1dice_list_sub1))
        print("NET Mean Dice1: {}".format(np.mean(NET_2dice_list_sub1)),"total:",len(NET_2dice_list_sub1))
        print("ET Mean Dice1: {}".format(np.mean(ET_3dice_list_sub1)),"total:",len(ET_3dice_list_sub1))

        print("ED Mean hd95: {}".format(np.mean(ED_1hd95_list_sub1)),"total:",len(ED_1hd95_list_sub1))
        print("NET Mean hd95: {}".format(np.mean(NET_2hd95_list_sub1)),"total:",len(NET_2hd95_list_sub1))
        print("ET Mean hd95: {}".format(np.mean(ET_3hd95_list_sub1)),"total:",len(ET_3hd95_list_sub1))
        print(" ")
        print("TC Mean Dice: {}".format(np.mean(TCdice_list_sub)),"total:",len(TCdice_list_sub))
        print("WT Mean Dice: {}".format(np.mean(WTdice_list_sub)),"total:",len(WTdice_list_sub))
        print("ET Mean Dice: {}".format(np.mean(ETdice_list_sub)),"total:",len(ETdice_list_sub))

        print("TC Mean hd95: {}".format(np.mean(TChd95_list_sub)),"total:",len(TChd95_list_sub))
        print("WT Mean hd95: {}".format(np.mean(WThd95_list_sub)),"total:",len(WThd95_list_sub))
        print("ET Mean hd95: {}".format(np.mean(EThd95_list_sub)),"total:",len(EThd95_list_sub))

        dice_list=np.zeros([24,9],dtype=np.float64)
        for i in range(24):
            dice_list[i][0]=ED_1dice_list_sub[i]
            dice_list[i][1]=NET_2dice_list_sub[i]
            dice_list[i][2]=ET_3dice_list_sub[i]

            dice_list[i][3]=ED_1dice_list_sub1[i]
            dice_list[i][4]=NET_2dice_list_sub1[i]
            dice_list[i][5]=ET_3dice_list_sub1[i]

            dice_list[i][6]=TCdice_list_sub[i]
            dice_list[i][7]=WTdice_list_sub[i]
            dice_list[i][8]=ETdice_list_sub[i]

        hd95_list=np.zeros([24,9],dtype=np.float64)
        for i in range(24):
            hd95_list[i][0]=ED_1hd95_list_sub[i]
            hd95_list[i][1]=NET_2hd95_list_sub[i]
            hd95_list[i][2]=ET_3hd95_list_sub[i]

            hd95_list[i][3]=ED_1hd95_list_sub1[i]
            hd95_list[i][4]=NET_2hd95_list_sub1[i]
            hd95_list[i][5]=ET_3hd95_list_sub1[i]

            hd95_list[i][6]=TChd95_list_sub[i]
            hd95_list[i][7]=WThd95_list_sub[i]
            hd95_list[i][8]=EThd95_list_sub[i]


        np.savetxt(output_directory + 'dice_list.csv', dice_list, fmt='%.16f')

        np.savetxt(output_directory + 'hd_list.csv', hd95_list, fmt='%.16f')


if __name__ == '__main__':
    main()
