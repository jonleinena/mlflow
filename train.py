#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 train.py
 Maintainer: Jon Leiñena
 Fecha: 2024
 Created By: Vicomtech 

"""

# ---------------------------------------------------------------------------
# Imports 
# ---------------------------------------------------------------------------

import os


import albumentations as albu

from datetime import datetime

import argparse

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import torch.optim.lr_scheduler as schedulers

import mlflow
import mlflow.pytorch
from dataset import Dataset, get_training_augmentation, get_validation_augmentation, get_preprocessing

import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils
#from segmentation_models_pytorch import metrics
# SSL: CERTIFICATE_VERIFY_FAILED solution
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
#

"""
Training script
"""

def my_collate(batch):
    batch = list(filter (lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

# Initialization functions

def parse_arguments():
    """
        This will allow to parametrize all the training with command line args. This way we can just launch a .sl script on the HPC with the different parameters to train models with diff configurations.        
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch", type=int, default=1)
    # ALWAYS THE VALUES OF HEIGHT AND WIDTH MUST BE MULTIPLE OF 32!!
    parser.add_argument("--img_height", type=int, default=224, help="size of image height")
    parser.add_argument("--img_width", type=int, default=224, help="size of image width")
    parser.add_argument("--encoder", type=str, default='efficientnet-b0')
    parser.add_argument("--encoder_weights", type=str, default='imagenet')    
    parser.add_argument("--activation", type=str, default='sigmoid')
    parser.add_argument("--classes", type=str,  nargs='+', default=['patches', 'inclusion', 'scratches'])         
    parser.add_argument("--network", type=str, default='unet', help = "unet, deeplabv3, fpn, pan, pspnet, unetplusplus")
    parser.add_argument("--epochs", type=int, default=1)   
    parser.add_argument("--data_dir", type=str, default='./data', help="source dir of dataset")
    parser.add_argument("--base_dir", type=str, default="./neu_output", help="dir to save models")
    parser.add_argument('--loss', type=str, default="dice")
    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Epochs to wait by scheduler for plateau before reducing learning rate")
    parser.add_argument('--optimizer', type=str, default="Adam")
    parser.add_argument('--scheduler', type=str, default='reduceOnPlateau', help="The learning rate scheduler to be used")    
    parser.add_argument('--experiment_name', type=str, default='exp_'+datetime.now().strftime("%d_%m_%Y"))
    opt = parser.parse_args()

 
    return opt

#os.path.join(*os.path.realpath(__file__).split("/")[:len(os.path.realpath(__file__).split("/"))-1])


################# MAIN FUNCTION #######################

def main():

    
    # Load arguments
    # Loads the user defined config into the program, enabling it to start operations with the specified settings.
    config = parse_arguments()

    DATA_DIR = config.data_dir
    BASE_DIR =  config.base_dir #el base dir donde guardaré los modelos

    dataset_name = DATA_DIR.split("/")[len(DATA_DIR.split("/")) -2]
    os.environ['MLFLOW_TRACKING_URI']  # Adjust the URI as needed
    mlflow.set_experiment(config.experiment_name)
    # Sets the output directory path based on the current date and time. Creates the new directory if it doesn't already exist.
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y")
    

    # LOAD DATA 
    # load repo with data if it is not exists
    if not os.path.exists(DATA_DIR):
        print('No data dir found')

    x_train_dir = os.path.join(DATA_DIR, 'train', 'imgs')
    y_train_dir = os.path.join(DATA_DIR, 'train', 'masks')



    x_valid_dir = os.path.join(DATA_DIR, 'validation', 'imgs')
    y_valid_dir = os.path.join(DATA_DIR, 'validation', 'masks')

    # CREATE MODEL AND TRAIN
    # Creates the segmentation model using a pretrained encoder (the type of encoder as well as its weights are specified by the user in config).
    ENCODER = config.encoder
    ENCODER_WEIGHTS = config.encoder_weights
    CLASSES = config.classes
    ACTIVATION = config.activation # could be None for logits or 'softmax2d' for multicalss segmentation
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    if config.network == 'unet':
        model = smp.Unet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )    
    elif config.network == 'deeplabv3':
        # create segmentation model with pretrained encoder
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )

    elif config.network == 'fpn':
        model = smp.FPN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif config.network == 'pan':
        model = smp.PAN(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )   
    elif config.network == 'pspnet':
        model = smp.PSPNet(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    elif config.network == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER, 
            encoder_weights=ENCODER_WEIGHTS, 
            classes=len(CLASSES), 
            activation=ACTIVATION,
        )
    else:
        return False
    

    # Gets the preprocessing function for the segmentation network according to the specified encoder name and weights.
    # This preprocessing might include resizing, scaling, etc.
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    # Loads the images and masks (for both training and validation datasets) in to the program. 
    #  Augmentation like cropping, flipping, random scaling, etc. does also occur here.
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir, 
        classes=CLASSES,
        augmentation=get_training_augmentation(config.img_height, config.img_width), 
        preprocessing=get_preprocessing(preprocessing_fn),
       
    )

    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir, 
        classes=CLASSES,
        augmentation=get_validation_augmentation(config.img_height, config.img_width), 
        preprocessing=get_preprocessing(preprocessing_fn),
        
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=True, num_workers=1, collate_fn=my_collate , drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    print("Training dataset: ", len(train_dataset), " images loaded. -> ", len(train_loader), "number of batches" )
    print("Validation dataset: ", len(valid_dataset), " images loaded.")

    # Define metrics:
    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

    
    if config.loss == "jaccard":
        loss = utils.losses.JaccardLoss()
    elif config.loss == "dice":
        loss= utils.losses.DiceLoss()
    elif config.loss == 'cross_entropy':
        loss = utils.losses.CrossEntropyLoss()
    elif config.loss == 'bce':
        loss = utils.losses.BCELoss()
    elif config.loss == 'bce_dice':
        loss = utils.losses.BCEDiceLoss()
    else:
        return False


    metrics = [
        utils.metrics.IoU(threshold=0.5),
        utils.metrics.Accuracy(),
        utils.metrics.Precision(),
        utils.metrics.Recall()
    ]



    optim_params = [ 
        dict(params=model.parameters(), lr=config.lr),
    ]

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(optim_params)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(optim_params, momentum=0.9, weight_decay=1e-6, dampening=0, nesterov=True)
    elif config.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(optim_params)
    elif config.optimizer == 'ASGD':
        optimizer = torch.optim.ASGD(optim_params)
    else:
        return False

    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.patience, threshold = 0.05,  verbose=True)

    if config.scheduler == 'reduceOnPlateau':
        scheduler = schedulers.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config.patience, threshold = 0.05,  verbose=True)
                
    elif config.scheduler == 'onecyclelr':
        scheduler = schedulers.OneCycleLR(optimizer, max_lr = 0.001, epochs = 50, steps_per_epoch = len(train_loader))
        
    elif config.scheduler == 'cosine':
        scheduler = schedulers.CosineAnnealingLR(optimizer, T_max = 50, eta_min=config.lr)
        

    # create epoch runners 
    # it is a simple loop of iterating over dataloader`s samples
    train_epoch = utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )


    valid_epoch = utils.train.ValidEpoch(
        model, 
        loss=loss, 
        metrics=metrics,    
        device=DEVICE,
        verbose=True,
    )

    
    
    max_score = 0
    prev_loss = 1
    num_lr_updates = 0
    # Loop over the 0th and 1st element in range
    best_epoch = 0
    for i in range(0, config.epochs):
        # Write into the file
          
        # Print epoch number 
        print('\nEpoch: {}'.format(i))
        # Train the model on train loader
        train_logs = train_epoch.run(train_loader)
        # Validate the model on valid loader
        valid_logs = valid_epoch.run(valid_loader)
        lr = optimizer.param_groups[0]['lr']
        if lr > 1e-7 and not (config.scheduler == 'reduceOnPlateau' or config.scheduler == 'onecyclelr'):   #we do not reduce past 1e-7 cause it's counter productive
            scheduler.step()  # Adjust learning rate depending on loss function to avoid rushing 
            num_lr_updates +=1 
        elif lr > 1e-7 and config.scheduler == 'reduceOnPlateau':
            scheduler.step(valid_logs['iou_score'])
            num_lr_updates += 1
        if optimizer.param_groups[0]['lr'] < lr:
            print(f"Learning rate updated to {optimizer.param_groups[0]['lr']}")

        #loss & accuracy
        # Check if current score is higher than maximum score otherwise save model
        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            #torch.save(model, './output/' + config.network + '_' + config.encoder + '_' + config.loss + '_bsz' + str(config.batch) + '_size' + str(config.img_height) + '_' + str(config.scheduler)+  '_p' + str(config.patience)+ '_best_model.pth')
            model_name = 'model_' + config.network + '_' + config.encoder + '_' + config.loss + '_bsz' + str(config.batch) + '_p' + str(config.patience) + dt_string
            mlflow.pytorch.log_model(model, model_name, registered_model_name = 'model_' + config.network + '_' + config.encoder + '_' + config.loss + '_bsz' + str(config.batch))
            best_epoch = i            
        
        
        #experiment metrics tracking
        mlflow.log_metric("train_loss", train_logs[f'{config.loss}_loss'], step=i)
        mlflow.log_metric("val_loss", valid_logs[f'{config.loss}_loss'], step=i)
        mlflow.log_metric("train_iou", train_logs['iou_score'], step=i)
        mlflow.log_metric("val_iou", valid_logs['iou_score'], step=i)
        mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step = i)

    



    

if __name__ == "__main__":
    with mlflow.start_run():
        main()
    

        