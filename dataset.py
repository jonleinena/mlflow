from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import albumentations as albu

class Dataset(BaseDataset):
    """Dataset for image segmentation. Reads images and their corresponding masks, applies augmentation and preprocessing transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transformation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. normalization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['background','patches', 'inclusion', 'scratches']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = [f.split('.')[0] for f in os.listdir(images_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes] 
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # read image
        
        image_path = os.path.join(self.images_dir, self.ids[i] + '.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #opencv loads as BGR and we need RGB
        # read mask
        mask_path = os.path.join(self.masks_dir, self.ids[i] + '.png')
        mask = cv2.imread(mask_path, 0)
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == (v)) for v in self.class_values] 
        
        mask = np.stack(masks, axis=-1).astype('float') 
        

        # apply augmentations
        if self.augmentation:
            # print(image.shape)
            # print(mask.shape)
            sample = self.augmentation(image=image, mask=mask)
            
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask, image_path

    def __len__(self):
        return len(self.ids)



def get_training_augmentation(img_height, img_width):
    
    train_transform = [

        albu.Resize(height=img_height, width=img_width, always_apply=True), #to ensure that the images are all the same size
        albu.PadIfNeeded(pad_height_divisor = 32, pad_width_divisor=32, min_width=None, min_height=None, always_apply=True, border_mode = 0, value=0), #expectedly not gonna need it because the img will already be resized to height and width divisible by 32
        albu.HorizontalFlip(p=0.5),
        
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        
    ]

    
    return albu.Compose(train_transform, is_check_shapes=False)


def get_validation_augmentation(img_height, img_width):
    """Add paddings to make validation image shape divisible by 32"""

    
    # Define the transform
    valid_transform = [
        albu.Resize(height=img_height, width=img_width, always_apply=True), #to ensure that the images are all the same size
        albu.PadIfNeeded(pad_height_divisor=32,pad_width_divisor=32, min_width=None, min_height=None,always_apply=True, border_mode=0, value=0), #expectedly not gonna need it because the img will already be resized to height and width divisible by 32
        
        #albu.RandomCrop(height=224, width=224, always_apply=True),
    ]
    
    return albu.Compose(valid_transform)


def to_tensor(x, **kwargs):
    
    return x.transpose(2, 0, 1).astype('float32')  # PyTorch works with CHW order, we read images in HWC [height, width, channels], don`t forget to transpose image.


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callabale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform, is_check_shapes=False)