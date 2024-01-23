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

### Train
        python train_ICCTUNet_synapse.py --cuda_id 0

### Evaluate
        python evaluate_synapse.py --cuda_id 0

### Visualize
        python predict_synapse.py --cuda_id 0

Check those scripts for more detail.

# Citation

        @article{qi2023small,
        title={Small Sample Image Segmentation By Coupling Convolutions and Transformers},
        author={Qi, Hao and Zhou, Huiyu and Dong, Junyu and Dong, Xinghui},
        journal={IEEE Transactions on Circuits and Systems for Video Technology},
        year={2023},
        publisher={IEEE}
      }

We used these repositories to implement our code:
        https://github.com/milesial/Pytorch-UNet
        https://github.com/Beckschen/TransUNet
        https://github.com/microsoft/Swin-Transformer
