import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import skimage.transform
from skimage import io
from  tifffile import imsave

#import albumentations as albu


import matplotlib.pyplot as plt



def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--crop_size", type=int, default=512)
    
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    
    parser.add_argument("--task_type", default="segmentation", help="task type")

    parser.add_argument("--chk_paths", default=" ",  nargs="+", help="Checkpoint paths to use in DivergentNets", required=True)  

    parser.add_argument("--input_dir", required=True, help="Input directory of images to predict mask.")  
    parser.add_argument("--output_dir", required=True, help="Output directory to save predicted mask.", default="./predicted_output")  

    return parser


def create_predFolder(task_type):
    directoryName = 'EndoCV2021'
    if not os.path.exists(directoryName):
        os.mkdir(directoryName)
        
    if not os.path.exists(os.path.join(directoryName, task_type)):
        os.mkdir(os.path.join(directoryName, task_type))
        
    return os.path.join(directoryName, task_type)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')



def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)



def mymodel(opt):
    '''
    Returns
    -------
    model : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    '''
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    models = []

    for chk_path in opt.chk_paths:
        checkpoint = torch.load(chk_path, map_location=opt.device)
        
        print("checkpoint path=", chk_path)
        print("checkpoint_best_epoch=", checkpoint["epoch"])
        
        models.append(checkpoint["model"])

    
    return models




if __name__ == '__main__':
    
    parser =get_argparser()
    opt = parser.parse_args()
    
    models = mymodel(opt)
     
        
    # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
    imgfolder=opt.input_dir
    
    # set folder to save your checkpoints here!
    saveDir = opt.output_dir
    os.makedirs(saveDir, exist_ok=True)

    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    imgfiles = detect_imgs(imgfolder, ext='.jpg')

    #data_transforms = get_preprocessing(preprocessing_fn=None)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    file = open(saveDir + '\\'+"timeElaspsed" +'.txt', mode='w')
    timeappend = []

    for imagePath in imgfiles[:]:
        """plt.imshow(img1[:,:,(2,1,0)])
        Grab the name of the file. 
        """
        filename = (imagePath.split('\\')[-1]).split('.jpg')[0]
        print('filename is printing::=====>>', filename)
        
        img1 = Image.open(imagePath).convert('RGB').resize((256,256), resample=0)
        img1 = np.array(img1)
      
        
        image = img1.transpose(2, 0, 1).astype('float32')
        images = torch.from_numpy(image).to(opt.device)
        
        
        # perform inference here:  
        img = skimage.io.imread(imagePath)
        size=img.shape
        start.record()

        preds = []

        for model in models:

            output = model.predict(images.unsqueeze(0))
            output = output.squeeze().cpu().numpy()
            pred = output[1,:,:]

            preds.append(pred)

        # Get average of all predictions
        preds = np.stack(preds)

        pred = np.mean(preds, axis=0)
        #print(pred.shape)
        

        pred = pred.round()
        pred = pred*255.0 
        pred = (pred).astype(np.uint8)

        
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        timeappend.append(start.elapsed_time(end))
       

        img_mask=skimage.transform.resize(pred, (size[0], size[1]), anti_aliasing=True) 

        # make sure all RGB channels have the same copy
        img_mask = np.dstack([img_mask] * 3)
        
        plt.imsave(saveDir +'\\'+ filename +'_mask.jpg', (img_mask*255.0).astype('uint8'))
        
        
        file.write('%s -----> %s \n' % 
            (filename, start.elapsed_time(end)))
        

        # TODO: write time in a text file
    
    file.write('%s -----> %s \n' % 
        ('average_t', np.mean(timeappend)))

    
