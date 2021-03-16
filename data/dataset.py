from torch.utils.data import Dataset as BaseDataset
import pyra_pytorch
import numpy as np


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    
    def __init__(
            self, 
            df, 
            classes=[0,255], 
            grid_sizes = [256],
            augmentation=None, 
            preprocessing=None,
            pyra=False
    ):
        #self.ids = os.listdir(images_dir)
        #self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        #self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.grid_sizes = grid_sizes

        self.pyra_dataset = pyra_pytorch.PYRADatasetFromDF(df, grid_sizes=grid_sizes)
        
        # convert str names to class values on masks
        self.class_values = classes
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.pyra = pyra
    
    def __getitem__(self, i):
        
        # read data
        data = self.pyra_dataset[i]
        #image = cv2.imread(self.images_fps[i])
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks_fps[i], 0)
        image = data["img"]
        grid = np.expand_dims(data["grid_encode"], axis=2)
        mask = data["mask"]
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        if self.pyra:
            image = np.concatenate([image, grid], axis=2)

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.pyra_dataset)