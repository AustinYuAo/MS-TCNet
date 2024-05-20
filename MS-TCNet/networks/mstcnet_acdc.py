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

from typing import Tuple, Union
import torch
import torch.nn as nn
import math
from networks.SSA3D_ACDC import ShuntedTransformer as ShuntedTransformer3D
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock


from functools import partial

class Multi_scale_feature_fusion_module(nn.Module):

    def __init__(self, in_channel, b=1, gama=2):

        super(Multi_scale_feature_fusion_module, self).__init__()

        kernel_size = int(abs((math.log(in_channel, 2) + b) / gama))

        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size


        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3,
                              bias=False, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.Softmax = nn.Softmax(dim=2) 

    def forward(self, inputs):

        b, c, h, w, d = inputs.shape

        x = self.avg_pool(inputs)

        x = x.view([b, 1, c])

        x = self.conv(x)

        x = self.Softmax(x)
        x = x.view([b, c, 1, 1, 1])

        return x

class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv3d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)
        q = self.norm(q)
        return U * q  

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.Conv_Squeeze = nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv3d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)
        z = self.Conv_Squeeze(z) 
        z = self.Conv_Excitation(z) 
        z = self.norm(z)
        return U * z.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse+U_sse

class UnetrBasicBlockScSE(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(UnetrBasicBlockScSE, self).__init__()
        self.conv=UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            norm_name="instance",
            res_block=True
        )
        self.scse=scSE(out_channels)

    def forward(self,input):
        out=self.conv(input)
        out=self.scse(out)
        return out

class feature_Extraction_Block(nn.Module):
    def __init__(self,num_block, in_channelslist,out_channels):
        super(feature_Extraction_Block, self).__init__()
        
        featureSelectBlock=[] 
        for i in range(num_block):
            featureSelectBlock.append(UnetrBasicBlockScSE(in_channels=in_channelslist[i], out_channels=in_channelslist[i+1]))  
        self.featureSelect_Block = nn.Sequential(*featureSelectBlock)
        self.out= UnetOutBlock(spatial_dims=3, in_channels=32, out_channels=out_channels)  

    def forward(self,input):
        out=self.featureSelect_Block(input)
        out=self.out(out)
        return out

class mstcnet(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 32,
        norm_name: Union[Tuple, str] = "instance",
        res_block: bool = True,
        dropout_rate: float = 0.0
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.outChannel = out_channels


        self.s_transformer3d=ShuntedTransformer3D(
            img_size= img_size[0], in_chans=in_channels,
            classification= False,
            embed_dims=[64,128,256,512],
            num_heads=[4, 4, 4, 4],
            mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, qk_scale=None, drop_rate=0.,
            attn_drop_rate=0., drop_path_rate=0.,  norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            num_conv=2)
        self.residual_blocks0 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.residual_blocks1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=32,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.residual_blocks2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size*2,
            out_channels=feature_size*2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.residual_blocks3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size*4,
            out_channels=feature_size*4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.residual_blocks4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size*8,
            out_channels=feature_size*8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.residual_blocks5 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size*16,
            out_channels=feature_size*16,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.up_sampling_block5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*16,
            out_channels=feature_size*8,
            kernel_size=3,
            upsample_kernel_size=(2,2,1),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.up_sampling_block4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*8,
            out_channels=feature_size*4,
            kernel_size=3,
            upsample_kernel_size=(2,2,1),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.up_sampling_block3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*4,
            out_channels=feature_size*2,
            kernel_size=3,
            upsample_kernel_size=(2,2,1),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.up_sampling_block2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size*2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2,2,1),
            norm_name=norm_name,
            res_block=res_block,
        )
        self.up_sampling_block1 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2,2,1),
            norm_name=norm_name,
            res_block=res_block,
        )

        self.feature_extraction_block5 = feature_Extraction_Block(num_block=4, in_channelslist=[feature_size*16,feature_size*8,feature_size*4,feature_size*2,feature_size],out_channels=self.outChannel)
        self.feature_extraction_block4 = feature_Extraction_Block(num_block=3, in_channelslist=[feature_size*8,feature_size*4,feature_size*2,feature_size],out_channels=self.outChannel)
        self.feature_extraction_block3 = feature_Extraction_Block(num_block=2, in_channelslist=[feature_size*4,feature_size*2,feature_size],out_channels=self.outChannel)
        self.feature_extraction_block2 = feature_Extraction_Block(num_block=1, in_channelslist=[feature_size*2,feature_size],out_channels=self.outChannel)


        self.unpool = nn.Upsample(scale_factor=(2,2,1), mode='trilinear', align_corners=True)

        self.MSFF = Multi_scale_feature_fusion_module(in_channel=6)

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=self.outChannel) 

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=self.outChannel)  


    def forward(self, x_in):
        x, hidden_states_out = self.s_transformer3d(x_in)
        enc0 = self.residual_blocks0(x_in)

        x1 = hidden_states_out[0]

        enc1 = self.residual_blocks1(x1)

        x2 = hidden_states_out[1]
        enc2 = self.residual_blocks2(x2)

        x3 = hidden_states_out[2]
        enc3 = self.residual_blocks3(x3)

        x4 = hidden_states_out[3]
        enc4 = self.residual_blocks4(x4)

        x5 = hidden_states_out[4]
        enc5 = self.residual_blocks5(x5)

        out5= self.feature_extraction_block5(enc5)

        dec3 = self.up_sampling_block5(enc5, enc4)

        out4= self.feature_extraction_block4(dec3)

        dec2 = self.up_sampling_block4(dec3, enc3)

        out3= self.feature_extraction_block3(dec2)

        dec1 = self.up_sampling_block3(dec2, enc2)

        out2 = self.feature_extraction_block2(dec1)

        dec0 = self.up_sampling_block2(dec1, enc1)

        out0 =self.up_sampling_block1(dec0,enc0)

        feature5 = self.unpool(self.unpool(self.unpool(self.unpool(self.unpool(out5)))))
        feature4 = self.unpool(self.unpool(self.unpool(self.unpool(out4))))
        feature3 = self.unpool(self.unpool(self.unpool(out3)))
        feature2 = self.unpool(self.unpool(out2))
        feature1 = self.unpool(self.out1(dec0))
      
        feature0 = self.out(out0)
      
        B, C, H, W, L = feature0.shape
        final_out = feature0.clone()

        for i in range(1,self.outChannel):
            singleChannel6 = feature5[:, i, :, :, :].reshape(B, 1, H, W, L)  
            singleChannel5 = feature4[:, i, :, :, :].reshape(B, 1, H, W, L)
            singleChannel4 = feature3[:, i, :, :, :].reshape(B, 1, H, W, L)
            singleChannel3 = feature2[:, i, :, :, :].reshape(B, 1, H, W, L)
            singleChannel2 = feature1[:, i, :, :, :].reshape(B, 1, H, W, L)
            singleChannel1 = feature0[:, i, :, :, :].reshape(B, 1, H, W, L)

            reCatChannel = torch.cat((singleChannel1, singleChannel2, singleChannel3, singleChannel4, singleChannel5, singleChannel6), 1)

            w = self.MSFF(reCatChannel)
            
            w1 = w[:, 0, :, :, :]
            w2 = w[:, 1, :, :, :]
            w3 = w[:, 2, :, :, :]
            w4 = w[:, 3, :, :, :]
            w5 = w[:, 4, :, :, :]
            w6 = w[:, 5, :, :, :]
            

            final_out[:, i, :, :, :] = w1 * feature0[:, i, :, :, :] + w2 * feature1[:, i, :, :, :] + w3 * feature2[:, i, :, :,:] + w4 * feature3[:, i, :,:,:] + w5 * feature4[:,i,:,:,:] + w6 * feature5[:,i,:,:,:]

        return final_out,feature0,feature1,feature2,feature3,feature4,feature5
