# MS-TCNet: An effective Transformer–CNN combined network using multi-scale feature learning for 3D medical image segmentation 
 [paper](https://www.sciencedirect.com/science/article/abs/pii/S0010482524001410)

## Installation

git clone https://github.com/AustinYuAo/MS-TCNet.git

cd MS-TCNet

conda env create -f environment.yml

source activate MS-TCNet

## Dataset  
1.Download

Synapse dataset [download](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
 
ACDC dataset [download](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)

MSD BraTS dataset [download](http://medicaldecathlon.com/dataaws/)

2. your dataset folders should be organized as follows:
```
├── dataset/  
    ├── Synapse/  
        ├── imagesTr/  
        ├── imagesTs/  
        ├── labelsTr/  
        ├── labelsTs/  
        ├── dataset.json  
    ├── ACDC/  
        ├── imagesTr/  
        ├── imagesTs/  
        ├── labelsTr/  
        ├── labelsTs/  
        ├── dataset.json  
    ├── MSD BraTS/    
        ├── imagesTr/   
        ├── imagesTs/  
        ├── labelsTr/  
        ├── labelsTs/  
        ├── dataset.json  
```
You can refer to the corresponding JSON files for the data partitioning of each dataset. We have stored these files in the [data json folder](https://github.com/AustinYuAo/MS-TCNet/tree/main/MS-TCNet/dataset%20json). You can also copy these files to the corresponding dataset folder for training.
## Trainning
### Synapse dataset  
python main.py --max_epochs=8000 --batch_size=2 --logdir=mstcnet_btcv --save_checkpoint --data_dir=your_dataset_dir --json_list=dataset_Synapse.json --model_name=mstcnet_btcv --workers=6  

### ACDC dataset  
python main.py --max_epochs=8000 --batch_size=8 --sw_batch_size=4 --in_channels=1 --out_channels=4 --space_x=1.25 --space_y=1.25 --space_z=10 --roi_x=128 --roi_y=128 --roi_z=6 --logdir=mstcnet_acdc --save_checkpoint --data_dir=your_dataset_dir --json_list=dataset_ACDC.json --model_name=mstcnet_acdc --workers=6  

### MSD BraTS dataset  
python main.py --max_epochs=800 --batch_size=3 --sw_batch_size=2 --in_channels=4 --out_channels=3 --a_min=0 --a_max=300 --b_min=0 --b_max=1.0 --space_x=1 --space_y=1 --space_z=1 --roi_x=128 --roi_y=128 --roi_z=128 --logdir=mstcnet_brats --save_checkpoint --data_dir=your_dataset_dir --json_list=dataset_BraTS.json --model_name=mstcnet_brats --val_every=50 --workers=6  

## Test
### Synapse dataset  
python test_btcv.py --data_dir=your_dataset_dir --json_list=dataset_Synapse.json --pretrained_dir=your_pretrained_dir --pretrained_model_name=model.pth --saved_checkpoint=ckpt  

### ACDC dataset  
python test_acdc.py --data_dir=your_dataset_dir --json_list=dataset_ACDC.json --pretrained_dir=your_pretrained_dir --pretrained_model_name=model.pth --saved_checkpoint=ckpt  

### MSD BraTS dataset  
python test_brats.py --data_dir=your_dataset_dir --json_list=dataset_BraTS.json --pretrained_dir=your_pretrained_dir --pretrained_model_name=model.pth --saved_checkpoint=ckpt  

## References
```
@article{ao2024ms,
  title={MS-TCNet: An effective Transformer--CNN combined network using multi-scale feature learning for 3D medical image segmentation},
  author={Ao, Yu and Shi, Weili and Ji, Bai and Miao, Yu and He, Wei and Jiang, Zhengang},
  journal={Computers in Biology and Medicine},
  volume={170},
  pages={108057},
  year={2024},
  publisher={Elsevier}
}
```
## Acknowledgement
The code is implemented based on [UNETR](https://github.com/Project-MONAI/research-contributions/tree/main/UNETR/BTCV). We would like to express our sincere thanks to the contributors.
