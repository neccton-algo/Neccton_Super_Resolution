import os
from Data_generators.data_generator3V201 import Data_generator, load_standardization_data
from models.attention_res_net3V201_noAtt import Att_Res_UNet
import sys
import tensorflow as tf
import time
import shutil
import numpy as np
import pandas as pd
import argparse
import abfile
import interpolation

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

def load_model(field_target, layer_target):

## Create a list of strings of the predictors variables and their layers to create the predictors and targets list used by the network (just below)
    if field_target == 'dp':
        var_pred_list = ['dp', 'temp', 'saln']
    elif field_target == 'u':
        var_pred_list = ['u', 'v']
    elif field_target == 'v':
        var_pred_list = ['v', 'u']
    else:
         var_pred_list = [field_target]

    if layer_target == 1:
        layer_pred_list = [1,2]
    elif layer_target == 50:
        layer_pred_list = [49,50]
    else:
        if field_target == 'dp':
            layer_pred_list = [layer_target]
        else:
            layer_pred_list = [layer_target - 1, layer_target, layer_target + 1]
## Create the predictors and targets list used by the network
    list_predictors = [
        f"{var}-layer-{layer_pred}" for var in var_pred_list for layer_pred in layer_pred_list]
    list_targets = [
        f"{field_target}-layer-{layer_target}"]

    list_predictors += ['tp5_mask']
    paths = {}
    root_data_dir = '/cluster/work/users/antber/from_Elio/Files/'
    paths["standard"] = root_data_dir
    paths["forcings"] = root_data_dir
    paths["data_res"] = os.path.join(root_data_dir,"LR_upsampled")
    paths["data_LR_upsampled"] = os.path.join(root_data_dir,"LR_upsampled")
    paths["weights"] = os.path.join(root_data_dir,"Weights")

    file_standardization_LR_upsampled = os.path.join(paths["standard"],"standard_LR_upsampled.h5")
    standard_LR_upsampled = load_standardization_data(file_standardization_LR_upsampled)
    standard_res = load_standardization_data(file_standardization_LR_upsampled) ## dummy standard
    standard_bathy = load_standardization_data(file_standardization_LR_upsampled) ## dummy standard
    standard_forcings = load_standardization_data(file_standardization_LR_upsampled) ## dummy     standard

    model_params = {"list_predictors": list_predictors,
                    "list_targets": list_targets,
                    "dim": (760, 800), # (jdm,idm)
                    "cropped_dim": (768, 800), #
                    "batch_size": 1,
                    "n_filters": [32*(i+1) for i in range(6)], #Ref Cyril: 32
                    "activation": "relu",
                    "kernel_initializer": "he_normal",
                    "batch_norm": True,
                    "pooling_type": "Average",
                    "dropout": 0,
                   }

    unet_model = Att_Res_UNet(**model_params).make_unet_model()

    return unet_model, standard_LR_upsampled, standard_res, standard_bathy, standard_forcings

def super_resolution(date, field_target, layer_target, mem, model_data):
    unet_model, standard_LR_upsampled, standard_res, standard_bathy, standard_forcings = model_data

## Create a list of strings of the predictors variables and their layers to create the predictors and targets list used by the network (just below)
    if field_target == 'dp':
         var_pred_list = ['dp', 'temp', 'saln']
    elif field_target == 'u':
        var_pred_list = ['u', 'v']
    elif field_target == 'v':
        var_pred_list = ['v', 'u']       
    else:
         var_pred_list = [field_target]
    
    if layer_target == 1:
        layer_pred_list = [1,2]
    elif layer_target == 50:
        layer_pred_list = [49,50]
    else:
        if field_target == 'dp':
            layer_pred_list = [layer_target]        
        else:
            layer_pred_list = [layer_target - 1, layer_target, layer_target + 1]

## Create the predictors and targets list used by the network
    list_predictors = [
        f"{var}-layer-{layer_pred}" for var in var_pred_list for layer_pred in layer_pred_list]
    list_targets = [
        f"{field_target}-layer-{layer_target}"]

    list_predictors += ['tp5_mask']

    dates_test = [str(date)]
    
    print('predictors : ', list_predictors)
    print('targets : ', list_targets)
    print('date : ', dates_test)
    paths = {}
    root_data_dir = '/cluster/work/users/antber/from_Elio/Files/'
    paths["standard"] = root_data_dir
    paths["forcings"] = root_data_dir
    paths["data_res"] = os.path.join(root_data_dir,"LR_upsampled")
    paths["data_LR_upsampled"] = os.path.join(root_data_dir,"LR_upsampled")
    paths["weights"] = os.path.join(root_data_dir,"Weights")

    params_test_dt1 = {"list_predictors": list_predictors,
                    "list_labels": list_targets,
                    "list_dates": dates_test,
                    "standard_forcings": standard_forcings,
                    "standard_res": standard_res,
                    "standard_LR_upsampled": standard_LR_upsampled,
                    "standard_bathy": standard_bathy,
                    "batch_size": 1,
                    "path_data_res": paths["data_res"],
                    "path_forcings": paths["forcings"],
                    "path_data_LR_upsampled": paths["data_LR_upsampled"],
                    "dim": (760, 800),
                    "cropped_dim": (768, 800),
                    "shuffle": False,
                    "res_normalization": 0,                    
                    "dtime": 1,
                    "mem": mem 
                    }
    test_generator1 = Data_generator(**params_test_dt1)
  
    predictors_part = "_".join(map(str, layer_pred_list)) ## From [1,2,3] constructs '1_2_3'
    checkpoint_file_name = paths["weights"] + f"/Checkpoints_3V201_{field_target}_Pred-{predictors_part}_Target-{layer_target}.h5"
    print('Weights file used: ', checkpoint_file_name)
    unet_model.load_weights(checkpoint_file_name)
    predict_batch1 = unet_model.predict(test_generator1[0][0])

    predict_dt1 = predict_batch1[0,:,:,0]
    predict_dt1 = predict_dt1[0:760,:]

    # write a new restart file with the Super resolved field downsampled for TP2
    Initial_restart_file = '/cluster/work/users/antber/TP2a0.10/expt_02.5/LR_upsampled/restart.' + str(dates_test[0]) + '_00_0000_mem' + mem + f'.a'    
    Initial_restart = abfile.ABFileRestart(Initial_restart_file, 'r', idm=800, jdm=760)

    New_restart_file = '/cluster/work/users/antber/TP2a0.10/expt_02.5/SR_fields/restart.' + str(date) + '_00_0000_mem' + mem + f'_part_{field_target}_{layer_target}.a'
    New_restart = abfile.ABFileRestart(New_restart_file,"w",idm=800,jdm=760)
    New_restart.write_header(25, Initial_restart._iversn, Initial_restart._yrflag, Initial_restart._sigver, Initial_restart._nstep, Initial_restart._dtime, Initial_restart._thbase)
        
    New_restart.write_field(predict_dt1, True, field_target, layer_target, 1)
    #New_restart.write_field(predict_dt2, True, field_target, layer_target, 2)
   
    New_restart.close()
    Initial_restart.close() 

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Apply the super resolution operator to a LR upsampled field.")
    parser.add_argument('date', type=str, help='The date in format yyyy_ddd, for instance 2019_273.')
    parser.add_argument('var', type=str, help='The variable to super resolve, eg temp')
    parser.add_argument('layer', type=int, help='The number of the layer, between 1 and 50')
    # Parse the command-line arguments
    args = parser.parse_args()
    
    model_data = load_model(args.var, args.layer)

    for mem in range(1,101):
        formatted_mem = f"{mem:03}"
        super_resolution(args.date, args.var, args.layer, formatted_mem, model_data)
