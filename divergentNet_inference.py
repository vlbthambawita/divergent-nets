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

    # Deeplab Options
    
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50', 'deeplabv3plus_mobilenet'], help='model name')
    
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--crop_size", type=int, default=512)
    
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    
    parser.add_argument("--task_type", default="segmentation", help="task type")
    

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



#def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
 #   _transform = []
 #   if preprocessing_fn:
 #       _transform.append(albu.Lambda(image=preprocessing_fn))
 #   _transform.append(albu.Lambda(image=to_tensor, mask=to_tensor))

    
  #  return albu.Compose(_transform)


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
 
    checkpoint_4 = torch.load('/work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_Deeplabv3.pth', map_location=opt.device)
    checkpoint_6 = torch.load('/work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_Depplabv3_plusplus.pth', map_location=opt.device)
    checkpoint_7 = torch.load('/work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_FPN.pth', map_location=opt.device)
    checkpoint_8 = torch.load('/work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_TriUnet.pth', map_location=opt.device)
    checkpoint_9 = torch.load('/work/vajira/data/EndoCV_2021/best_checkpoints/best_checkpoint_unet_plusplus.pth', map_location=opt.device)


    #print("chk_epoch=", checkpoint_1["epoch"])
   # print("chk_epoch=", checkpoint_2["epoch"])
   # print("chk_epoch=", checkpoint_3["epoch"])
    print("chk_epoch=", checkpoint_4["epoch"])
    #print("chk_epoch=", checkpoint_5["epoch"])
    print("chk_epoch=", checkpoint_6["epoch"])
    print("chk_epoch=", checkpoint_7["epoch"])
    print("chk_epoch=", checkpoint_8["epoch"])
    print("chk_epoch=", checkpoint_9["epoch"])


    #model_1 = checkpoint_1["model"]
    #model_2 = checkpoint_2["model"]
    #model_3 = checkpoint_3["model"]
    model_4 = checkpoint_4["model"]
    #model_5 = checkpoint_5["model"]

    model_6 = checkpoint_6["model"]
    model_7 = checkpoint_7["model"]
    model_8 = checkpoint_8["model"]
    model_9 = checkpoint_9["model"]

    
    return model_4,model_6, model_7, model_8, model_9 




if __name__ == '__main__':
    
    parser =get_argparser()
    opt = parser.parse_args()
    
    model_4, model_6, model_7, model_8, model_9 = mymodel(opt)
    
    #directoryName = create_predFolder(opt.task_type)
    
    
    
        
    # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
    imgfolder="/work/vajira/data/EndoCV_2021/Kvasir_seg/Kvasir-SEG/images"
    
    # set folder to save your checkpoints here!
    saveDir = "/home/vajira/DL/temp_data/test_save"
    os.makedirs(saveDir, exist_ok=True)

    if not os.path.exists(saveDir):
        os.mkdir(saveDir)

    imgfiles = detect_imgs(imgfolder, ext='.jpg')

    #data_transforms = get_preprocessing(preprocessing_fn=None)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    file = open(saveDir + '/'+"timeElaspsed" +'.txt', mode='w')
    timeappend = []

    for imagePath in imgfiles[:]:
        """plt.imshow(img1[:,:,(2,1,0)])
        Grab the name of the file. 
        """
        filename = (imagePath.split('/')[-1]).split('.jpg')[0]
        print('filename is printing::=====>>', filename)
        
        img1 = Image.open(imagePath).convert('RGB').resize((256,256), resample=0)
        #image = data_transforms(img1)
        # perform inference here:
        #images = image.to(device, dtype=torch.float32)
        img1 = np.array(img1)
        #image = data_transforms(image=img1)["image"]
        
        image = img1.transpose(2, 0, 1).astype('float32')
        images = torch.from_numpy(image).to(opt.device)
        # perform inference here:
        #images = image.to(device, dtype=torch.float32)

        #            
        img = skimage.io.imread(imagePath)
        size=img.shape
        start.record()
        #
        # outputs_1 = model_1.predict(images.unsqueeze(0))
        # outputs_2 = model_2.predict(images.unsqueeze(0))
        # outputs_3 = model_3.predict(images.unsqueeze(0))
        outputs_4 = model_4.predict(images.unsqueeze(0))
        # outputs_5 = model_5.predict(images.unsqueeze(0))

        outputs_6 = model_6.predict(images.unsqueeze(0))
        outputs_7 = model_7.predict(images.unsqueeze(0))
        outputs_8 = model_8.predict(images.unsqueeze(0))
        outputs_9 = model_9.predict(images.unsqueeze(0))

        #outputs_1 = outputs_1.squeeze().cpu().numpy()
        #outputs_2 = outputs_2.squeeze().cpu().numpy()
        #outputs_3 = outputs_3.squeeze().cpu().numpy()
        outputs_4 = outputs_4.squeeze().cpu().numpy()
        #outputs_5 = outputs_5.squeeze().cpu().numpy()

        outputs_6 = outputs_6.squeeze().cpu().numpy()
        outputs_7 = outputs_7.squeeze().cpu().numpy()
        outputs_8 = outputs_8.squeeze().cpu().numpy()
        outputs_9 = outputs_9.squeeze().cpu().numpy()

        #print(outputs)
        #break
        #
        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))
        timeappend.append(start.elapsed_time(end))
        #

        #preds = outputs[1,:,:]#outputs.detach().max(dim=1)[1].cpu().numpy()   
        
        #preds_1 = outputs_1[1,:,:]#outputs.detach().max(dim=1)[1].cpu().numpy()  
        #preds_2 = outputs_2[1,:,:]
        #preds_3 = outputs_3[1,:,:]
        preds_4 = outputs_4[1,:,:]
        #preds_5 = outputs_5[1,:,:]

        preds_6 = outputs_6[1,:,:]
        preds_7 = outputs_7[1,:,:]
        preds_8 = outputs_8[1,:,:]
        preds_9 = outputs_9[1,:,:]


        preds = (preds_4 + preds_6 + preds_7 + preds_8 + preds_9)/5
        #print(preds)
        #break
        preds = preds.round()

        pred = preds*255.0 
        pred = (pred).astype(np.uint8)

        #print(pred.shape)
        #plt.imshow(pred)

        #break


        img_mask=skimage.transform.resize(pred, (size[0], size[1]), anti_aliasing=True) 

        # make sure all RGB channels have the same copy
        img_mask = np.dstack([img_mask] * 3)
        
        plt.imsave(saveDir +'/'+ filename +'_mask.jpg', (img_mask*255.0).astype('uint8'))
        
        
        file.write('%s -----> %s \n' % 
            (filename, start.elapsed_time(end)))
        

        # TODO: write time in a text file
    
    file.write('%s -----> %s \n' % 
        ('average_t', np.mean(timeappend)))

    
