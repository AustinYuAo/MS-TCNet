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
from networks.mstcnet_synapse import mstcnet as mstcnet_synapse
from utils.data_utils_synapse import get_loader_synapse
from utils.utils import resample_3d
import nibabel as nib

import argparse
from functools import partial

from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from monai.metrics import DiceMetric,HausdorffDistanceMetric
from monai.utils.enums import MetricReduction


parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/dataset/dataset0/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=2.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--save_image', default='No', type=str, help=' Yes:save image as nii file')

def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader_synapse(args)
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
        model =mstcnet_synapse(
            in_channels=1,
            out_channels=9,
            img_size=(96, 96, 96),
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
                            roi_size=[96,96,96],
                            sw_batch_size=4,
                            predictor=model,
                            overlap=args.infer_overlap)
    
    post_label = AsDiscrete(to_onehot=9)
    post_pred = AsDiscrete(argmax=True,
                           to_onehot=9)
    
    dice_acc = DiceMetric(include_background=False,
                          reduction=MetricReduction.MEAN,
                          get_not_nans=True)
    metric_hd = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=95)

    all_dice_list=np.zeros([8,12],dtype=np.float64)
    all_hd95_list=np.zeros([8,12],dtype=np.float64)

    with torch.no_grad():

        dice8_list_case = []
        hd8_list_case=[]
        
        for i, batch in enumerate(val_loader):
            outNum=i
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch['label_meta_dict']['affine'][0].numpy()
            
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))

            val_outputs =  model_inferer(val_inputs)

            val_labels[val_labels==5]=0
            val_labels[val_labels==9]=0
            val_labels[val_labels==10]=0
            val_labels[val_labels==12]=0  
            val_labels[val_labels==13]=0
            val_labels[val_labels == 6] = 5  
            val_labels[val_labels == 7] = 6  
            val_labels[val_labels == 8] = 7  
            val_labels[val_labels == 11] = 8  
            
            target=val_labels
            logits=val_outputs
            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]


            acc = dice_acc(y_pred=val_output_convert, y=val_labels_convert)
            acc_list = acc.detach().cpu().numpy()
            for i in range(0,8):
                if np.isnan(acc_list[0][i])==True:
                    acc_list[0][i]=0

            # print(acc_list[0][0],acc_list[0][1],acc_list[0][2],acc_list[0][3],acc_list[0][4],acc_list[0][5],acc_list[0][6],acc_list[0][7])

            mean_dice8=(acc_list[0][0]+acc_list[0][1]+acc_list[0][2]+acc_list[0][3]+acc_list[0][4]+acc_list[0][5]+acc_list[0][6]+acc_list[0][7])/8
            
            dice8_list_case.append(mean_dice8)
                
            print(" ")
            acc_hd = metric_hd(y_pred=val_output_convert, y=val_labels_convert)
            acc_hd_list = acc_hd.detach().cpu().numpy()
            for i in range(0,8):
                if np.isnan(acc_hd_list[0][i])==True or np.isinf(acc_hd_list[0][i])==True:
                    acc_hd_list[0][i]=0

            # print(acc_hd_list[0][0],acc_hd_list[0][1],acc_hd_list[0][2],acc_hd_list[0][3],acc_hd_list[0][4],acc_hd_list[0][5],acc_hd_list[0][6],acc_hd_list[0][7])

            mean_hd8=(acc_hd_list[0][0]+acc_hd_list[0][1]+acc_hd_list[0][2]+acc_hd_list[0][3]+acc_hd_list[0][4]+acc_hd_list[0][5]+acc_hd_list[0][6]+acc_hd_list[0][7])/8

            print("Dice acc_average:",mean_dice8," HD95 acc_average:",mean_hd8)
            hd8_list_case.append(mean_hd8)
            for j in range(0,8):
                all_dice_list[j][outNum]=acc_list[0][j]
                all_hd95_list[j][outNum]=acc_hd_list[0][j]
            print(" ")
            
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_inputs = val_inputs.cpu().numpy()[0, 0, :, :, :]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            
            if args.save_image=="Yes":
                # nib.save(nib.Nifti1Image(val_inputs, original_affine),
                #         os.path.join(output_directory, "source_"+img_name))
                # nib.save(nib.Nifti1Image(val_labels.astype(np.uint8), original_affine),
                #         os.path.join(output_directory, "label_"+img_name))
                nib.save(nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                        os.path.join(output_directory, "seg_result_"+img_name))


    print("all average dice",np.mean(dice8_list_case))
    print("all average hd95",np.mean(hd8_list_case))

    # print(all_dice_list)
    # print(all_hd95_list)

    np.savetxt(output_directory + 'dice_list.csv', all_dice_list, fmt='%.16f')
    np.savetxt(output_directory + 'hd95_list.csv', all_hd95_list, fmt='%.16f')


if __name__ == '__main__':
    main()
