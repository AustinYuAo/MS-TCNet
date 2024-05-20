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
# 新写的test 加了保存图像

"""
test 方法

2024年5月3日
python test_acdc.py --data_dir=/raid/user_dir/ay/data/BTCV/UnetRdata --pretrained_dir=/raid/user_dir/ay/UnetRSSA3D3-likeSwinUNetR-FAN2-DS/MS-TCNet/runs/mstcnet_btcv/ --pretrained_model_name=model.pth --saved_checkpoint=ckpt --json_list=dataset_18_12.json

python test_acdc.py --pretrained_dir=/raid/user_dir/ay/UnetRSSA3D3-likeSwinUNetR-FAN2-DS/MS-TCNet/runs/mstcnet_acdc1/ --pretrained_model_name=model5400.pth --saved_checkpoint=ckpt
"""


import os
import torch
import numpy as np
from utils.sliding_window_inference_for_mstcnet import sliding_window_inference
from networks.mstcnet_acdc import mstcnet as mstcnet_acdc
from utils.data_utils_acdc import get_loader_acdc
from utils.utils import resample_3d
import nibabel as nib

import argparse
from functools import partial

from trainer_acdc import dice

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/raid/user_dir/ay/data/ACDC/UnetRdata01/', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_MTM.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--space_x', default=1.25, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.25, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=10.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=6, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=4, type=int, help='number of workers')
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
    test_loader = get_loader_acdc(args)
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
        model =mstcnet_acdc(
            in_channels=1,
            out_channels=4,
            img_size=(128, 128, 6),
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
                            roi_size=[128,128,6],
                            sw_batch_size=4,
                            predictor=model,
                            overlap=args.infer_overlap)
    

    with torch.no_grad():

        dice_list_case = []
        RVdice_list_sub = []
        Myodice_list_sub = []
        LVdice_list_sub = []
        
        for j, batch in enumerate(test_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch['label_meta_dict']['affine'][0].numpy()
            
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            
            val_inputs = (val_inputs - val_inputs.mean()) / val_inputs.std()
            
            val_outputs =  model_inferer(val_inputs)

            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_inputs = val_inputs.cpu().numpy()[0, 0, :, :, :]#后添加
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            dice_list_sub = []

            for i in range(1, 4):
                organ_Dice = dice(val_outputs == i, val_labels == i)
                print(i,'dice',np.around(organ_Dice,6))
                if organ_Dice!=0:
                    dice_list_sub.append(organ_Dice)
                if i==1:
                    RVdice_list_sub.append(organ_Dice)
                elif i==2:
                    Myodice_list_sub.append(organ_Dice)
                elif i==3:
                    LVdice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            
            if args.save_image=="Yes":
                # nib.save(nib.Nifti1Image(val_inputs, original_affine),
                #         os.path.join(output_directory, "source_"+img_name))
                # nib.save(nib.Nifti1Image(val_labels.astype(np.uint8), original_affine),
                #         os.path.join(output_directory, "label_"+img_name))
                nib.save(nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                        os.path.join(output_directory, "seg_result_"+img_name))


        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        print("Rv Mean Dice: {}".format(np.mean(RVdice_list_sub)))
        print("Myo Mean Dice: {}".format(np.mean(Myodice_list_sub)))
        print("Lv Mean Dice: {}".format(np.mean(LVdice_list_sub)))



if __name__ == '__main__':
    main()