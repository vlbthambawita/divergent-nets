#!/bin/bash

python inference_from_divergentNets.py \
    --input_dir /work/vajira/data/EndoCV_2021/Kvasir_seg/Kvasir-SEG/images \
    --output_dir /home/vajira/DL/temp_data/test_save \
    --chk_paths \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_Deeplabv3.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_Depplabv3_plusplus.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_FPN.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_TriUnet.pth \
    /work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_unet_plusplus.pth


