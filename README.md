# ICCTUNet
This is the official implementation of our paper "Small Sample Image Segmentation By Coupling Convolutions and Transformers".

# Usage

### Installation:

        pip install -r requirements.txt

### Prepare Dataset
Please download datasets from these links.

Synapse, ACDC

        https://github.com/Beckschen/TransUNet

Tumor

        https://github.com/282857341/nnFormer

BUSI, ISIC

         https://github.com/jeya-maria-jose/UNeXt-pytorch

### Train
        python train_ICCTUNet_synapse.py --cuda_id 0

### Evaluate
        python evaluate_synapse.py --cuda_id 0

### Visualize
        python predict_synapse.py --cuda_id 0

Check those scripts for more details.

### Pretrained Weights

        Synapse: https://pan.baidu.com/s/1ayoRW9lCNpQplTVV-n3lsg?pwd=9me1 
        ACDC: https://pan.baidu.com/s/1IaKFbx2ef93R90C5laiLKA?pwd=fejx 

# Citation

        @article{qi2023small,
        title={Small Sample Image Segmentation By Coupling Convolutions and Transformers},
        author={Qi, Hao and Zhou, Huiyu and Dong, Junyu and Dong, Xinghui},
        journal={IEEE Transactions on Circuits and Systems for Video Technology},
        year={2023},
        publisher={IEEE}
      }
# Acknowledgments
Thanks to the authors of these repositories:
        
        https://github.com/milesial/Pytorch-UNet      
        https://github.com/Beckschen/TransUNet 
        https://github.com/microsoft/Swin-Transformer
