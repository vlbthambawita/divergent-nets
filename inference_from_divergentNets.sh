#!/bin/bash

python divergentNet_inference_v2.py --chk_paths /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_Deeplabv3.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_Depplabv3_plusplus.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_FPN.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_TriUnet.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_unet_plusplus.pth


