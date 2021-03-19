#=========================================================
# Developer: Vajira Thambawita
# Reference: https://github.com/meetshah1995/pytorch-semseg
#=========================================================



import argparse
from datetime import datetime
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#Pytorch
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms,datasets, utils
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchsummary import summary

import segmentation_models_pytorch as smp


from data.dataset import Dataset
from data.prepare_data import prepare_data, prepare_test_data
#from data import PolypsDatasetWithGridEncoding
#from data import PolypsDatasetWithGridEncoding_TestData
import pyra_pytorch as pyra
from utils import dice_coeff, iou_pytorch, visualize

import segmentation_models_pytorch as smp

from my_models.triple_models import TripleUnet
from my_models.delphi_ensemble import DelphiEnsemble as DivergentNet


#======================================
# Get and set all input parameters
#======================================

parser = argparse.ArgumentParser()

# Hardware
#parser.add_argument("--device", default="gpu", help="Device to run the code")
parser.add_argument("--device_id", type=int, default=0, help="")

# Optional parameters to identify the experiments
parser.add_argument("--name", default="", type=str, help="A name to identify this test later")
parser.add_argument("--py_file",default=os.path.abspath(__file__)) # store current python file


parser.add_argument("--test_CSVs",
                    default=["/work/vajira/data/EndoCV_2021/CSV_file_with_paths/kvasir_seg.csv"],
                    help="CSV file list with image and mask paths")



parser.add_argument("--out_dir", 
                    default="/work/vajira/data/EndoCV_2021/2xx_checkpoints",
                    help="Main output dierectory")

parser.add_argument("--tensorboard_dir", 
                    default="/work/vajira/data/EndoCV_2021/2xx_tensorboard",
                    help="Folder to save output of tensorboard")



parser.add_argument("--test_out_dir",
                   default= "/work/vajira/DATA/mediaeval2020/test_data_predictions",
                   help="Output folder for testing data"
)    

parser.add_argument("--best_checkpoint_name", type=str, default="best_checkpoint.pth", help="A name to save bet checkpoint")


# Action handling 
parser.add_argument("--num_epochs", type=int, default=1, help="Numbe of epochs to train")
parser.add_argument("--start_epoch", type=int, default=0, help="start epoch of training")
parser.add_argument("--num_test_samples", type=int, default=5, help="Number of samples to test.")

# smp parameters
parser.add_argument("--encoder", type=str, default='se_resnext50_32x4d', help="smp encoders")
parser.add_argument("--encoder_weights", type=str, default='imagenet', help="encoder weights")
parser.add_argument("--classes", default=[0,255], help="classes per pixel")
parser.add_argument("--activation", type=str, default='softmax2d', help="last activation layers activation")

#PYRA
parser.add_argument("--pyra", type=bool, default=False, help="To enable PYRA grid encoding.")
parser.add_argument("--grid_sizes_train", type=list, default=[256], help="Grid sizes to use in training")
parser.add_argument("--grid_sizes_val", type=list, default=[256], help="Grid sizes to use in training")
parser.add_argument("--grid_sizes_test", type=list, default=[256], help="Grid sizes to use in testing")
parser.add_argument("--in_channels", type=int, default=3, help="Number of input channgels")

# Parameters
parser.add_argument("--bs", type=int, default=8, help="Mini batch size")
parser.add_argument("--val_bs", type=int, default=1, help="Batch size")
parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate for training") # reduced 0.0001 to 0.00001 from the best checkpoint
parser.add_argument("--lr_change_point", type=int, default=50, help="After this point LR will be changed.")


parser.add_argument("--num_workers", type=int, default=12, help="Number of workers in dataloader")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay of the optimizer")
parser.add_argument("--lr_sch_factor", type=float, default=0.1, help="Factor to reduce lr in the scheduler")
parser.add_argument("--lr_sch_patience", type=int, default=25, help="Num of epochs to be patience for updating lr")

parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to print from validation set")
parser.add_argument("action", type=str, help="Select an action to run", choices=[ "generate", "check",])
parser.add_argument("--checkpoint_interval", type=int, default=25, help="Interval to save checkpoint models")


parser.add_argument("--test_checkpoint", type=str, default= "/work/vajira/data/EndoCV_2021/2xx_checkpoints/800_triple_unetPlusPlus_ignore_class_0_smp_v1.py/checkpoints/backup/checkpoint_06_03_2021_7746/best_checkpoint.pth")

opt = parser.parse_args()

#=================
# Set checkpoints
#==================
opt.model_1_path = "/work/vajira/data/EndoCV_2021/2xx_checkpoints/800_triple_unetPlusPlus_ignore_class_0_smp_v1.py/checkpoints/backup/checkpoint_06_03_2021_7746/best_checkpoint.pth"
opt.model_2_path = "/work/vajira/data/EndoCV_2021/DGX_checkpoints/301_new_basic_unet_plusplus_ignore_class_0_smp_v1.py/checkpoints/best_checkpoint.pth"
opt.model_3_path = "/work/vajira/data/EndoCV_2021/DGX_checkpoints/303_new_FPN_ignore_class_0_smp_v1.py/checkpoints/best_checkpoint.pth"
opt.model_4_path = "/work/vajira/data/EndoCV_2021/DGX_checkpoints/306_new_Deeplabv3_ignore_class_0_smp_v1.py/checkpoints/best_checkpoint.pth"
opt.model_5_path = "/work/vajira/data/EndoCV_2021/DGX_checkpoints/307_new_Deeplabv3Pluse_ignore_class_0_smp_v1.py/checkpoints/best_checkpoint.pth"


#==========================================
# Device handling
#==========================================
torch.cuda.set_device(opt.device_id)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.device = DEVICE

#===========================================
# Folder handling
#===========================================

#make output folder if not exist
os.makedirs(opt.out_dir, exist_ok=True)


# make subfolder in the output folder 
py_file_name = opt.py_file.split("/")[-1] # Get python file name (soruce code name)
CHECKPOINT_DIR = os.path.join(opt.out_dir, py_file_name + "/checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# make tensorboard subdirectory for the experiment
tensorboard_exp_dir = os.path.join(opt.tensorboard_dir, py_file_name)
os.makedirs( tensorboard_exp_dir, exist_ok=True)

#==========================================
# Tensorboard
#==========================================
# Initialize summary writer
writer = SummaryWriter(tensorboard_exp_dir)



#==============================================
# Heatmap generator from tensor
#==============================================
def generate_heatmapts(img_tensor):
    print(img_tensor.shape)
    fig_list = []
    for n in range(img_tensor.shape[0]):
        img = img_tensor[n]
        img = img.squeeze(dim=0)
        img_np = img.detach().cpu().numpy()
        #img_np = np.transforms(img_np, (1,2,0))
        
        plt.imshow(img_np, cmap="hot")
        fig = plt.gcf()
        fig_list.append(fig)
        # plt.clf()
        plt.close()

    return fig_list

#===================================
# Inference from pre-trained models
#===================================

def generate_mask(opt):

    opt.record_name = "TEST"

    best_model = DivergentNet(opt)

    print("checkpoint 1=", opt.model_1_path)
    print("checkpoint 2=", opt.model_2_path)
    print("checkpoint 3=", opt.model_3_path)
    print("checkpoint 4=", opt.model_4_path)
    print("checkpoint 5=", opt.model_5_path)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder, opt.encoder_weights)
    test_dataset = prepare_test_data(opt, preprocessing_fn=None)
    test_dataset_vis = prepare_test_data(opt, preprocessing_fn=None)
    
    
    for i in range(opt.num_test_samples):
        image, mask = test_dataset[i]
        image_vis, _ = test_dataset_vis[i]

        #print(image)

        mask_tensor = torch.from_numpy(mask).to(opt.device).unsqueeze(0)

        image_tensor = torch.from_numpy(image).to(opt.device).unsqueeze(0)
        pr_mask = best_model.predict(image_tensor)

        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        fig = visualize(
            input_image_new=np.transpose(image_vis, (1,2,0)).astype(int),
            GT_mask_0=mask[0, :,:],
            Pred_mask_0 = pr_mask[0,:,:],
            GT_mask_1= mask[1,:,:],
            Pred_mask_1 = pr_mask[1, :,:]
        )

        fig.savefig(f"./test_202_{i}.png")
        writer.add_figure(f"Test_sample/sample-{i}", fig, global_step=test_epoch)


#=====================================
# Check model
#====================================
def check_val_full_score(opt):

    opt.record_name = "TEST"

    best_model = DivergentNet(opt)

    print("checkpoint 1=", opt.model_1_path)
    print("checkpoint 2=", opt.model_2_path)
    print("checkpoint 3=", opt.model_3_path)
    print("checkpoint 4=", opt.model_4_path)
    print("checkpoint 5=", opt.model_5_path)

    test_best_epoch = 0
    
    preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder, opt.encoder_weights)
    test_dataset = prepare_test_data(opt, preprocessing_fn=None)
    
    test_dataloader = DataLoader(test_dataset, num_workers=48)

    loss = smp.utils.losses.DiceLoss()
    # Testing with two class layers
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=None),
        smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=None),
        smp.utils.metrics.Accuracy(threshold=0.5, ignore_channels=None),
        smp.utils.metrics.Recall(threshold=0.5, ignore_channels=None),
        smp.utils.metrics.Precision(threshold=0.5, ignore_channels=None),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    print("logs=", str(logs))
    writer.add_text(f"{opt.py_file}-scores-->{opt.record_name}", str(logs), global_step=test_best_epoch)

    # Testing with only class layer 1 (polyps)
    loss = smp.utils.losses.DiceLoss(ignore_channels=[0])
    
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0]),
        smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[0]),
        smp.utils.metrics.Accuracy(threshold=0.5, ignore_channels=[0]),
        smp.utils.metrics.Recall(threshold=0.5, ignore_channels=[0]),
        smp.utils.metrics.Precision(threshold=0.5, ignore_channels=[0]),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    print("logs=", str(logs))
    writer.add_text(f"{opt.py_file}-val-scores-ignore-channel-0-->{opt.record_name}", str(logs), global_step=test_best_epoch)

    # Testing with only class layer 0 (BG)

    loss = smp.utils.losses.DiceLoss(ignore_channels=[1])
    metrics = [
        #smp.utils.metrics.IoU(threshold=0.5),
        smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[1]),
        smp.utils.metrics.Fscore(threshold=0.5, ignore_channels=[1]),
        smp.utils.metrics.Accuracy(threshold=0.5, ignore_channels=[1]),
        smp.utils.metrics.Recall(threshold=0.5, ignore_channels=[1]),
        smp.utils.metrics.Precision(threshold=0.5, ignore_channels=[1]),
    ]

    test_epoch = smp.utils.train.ValidEpoch(
        model=best_model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
    )

    logs = test_epoch.run(test_dataloader)
    print("logs=", str(logs))
    writer.add_text(f"{opt.py_file}-val-scores-ignore-channel-1-->{opt.record_name}", str(logs), global_step=test_best_epoch)



if __name__ == "__main__":

    #data_loaders = prepare_data()
    print(vars(opt))
    print("Test OK")

    # Train or retrain or inference
    if opt.action == "generate":
        print("Generating outputs..!")
        generate_mask(opt)
        pass

    elif opt.action == "check":
        print("Checking score..!")
        check_val_full_score(opt)
        pass

    # Finish tensorboard writer
    writer.close()

