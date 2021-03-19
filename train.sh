#!/bin/bash





python tri_unet.py train \
    --num_epochs 2 \
    --device_id 0  \
    --train_CSVs sample_CSV_files/C1.csv sample_CSV_files/C1.csv \
    --val_CSVs sample_CSV_files/C2.csv sample_CSV_files/C3.csv \
    --test_CSVs sample_CSV_files/C3.csv \
    --out_dir ../temp_data \
    --tensorboard_dir ../temp_data 

