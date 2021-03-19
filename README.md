# Endocv2021-winner
This is the winning solution of the  Endocv-2021 grand challange. 

### Dependencies 

```python
pytorch # tested with 1.7 and 1.8
torchvision 
tqdm
pandas
numpy
albumentations # for augmentations
torchsummary
segmentation_models_pytorch # for basic segmentaion models
pyra_pytorch # pyra_pytorch.PYRADatasetFromDF is used. But this can be replaced with normal pytorch dataset.

```

## Tri-Unet

### Block diagram of Tri-Unet

![TriUnet](images/EndoCV_2021_diagrams-Tri-Unet.png)

### How to train Tri-Unet and other basic models to DivergentNet?

```python

# To train Tri-unet

python tri_unet.py train \
    --num_epochs 2 \
    --device_id 0  \
    --train_CSVs sample_CSV_files/C1.csv sample_CSV_files/C1.csv \
    --val_CSVs sample_CSV_files/C2.csv sample_CSV_files/C3.csv \
    --test_CSVs sample_CSV_files/C3.csv \
    --out_dir ../temp_data \
    --tensorboard_dir ../temp_data  

# To train other models, you have to replace tri_unet.py with one of the follwings:
unet_plusplus.py
deeplabv3.py
deeplabv3_plusplus.py
```



## DivergentNet
![DivergentNet](images/EndoCV_2021_diagrams_Delphi_esemble_v2.png)


## Merging and predicting from divergent network
How?

### Sample predictions from different models used in DivergentNet and it's own output.
![predictions](images/predictions.png)


## Citation
```python
TBA
```


