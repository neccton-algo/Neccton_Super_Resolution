#!/usr/bin/env python
# coding: utf-8

import os
from data_generator3V201 import Data_generator, generate_dates, convert_date_format, \
                            load_standardization_data
from models.attention_res_net3V201_noAtt import Att_Res_UNet
from net_util import save_model_parameters
import tensorflow as tf
import time
import numpy as np
import argparse

# In[3]:


tf.keras.utils.set_random_seed(420)
print("GPUs available: ", tf.config.list_physical_devices('GPU'))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def parse_var_arg(value):
    """
    Parses var_arg to allow either a single string or a comma-separated list of strings.
    """
    if ',' in value:
        # Split the comma-separated string into a list of strings
        return value.split(',')
    return value

def parse_layer_pred(value):
    """
    Custom parser for `layer_pred_arg`. Accepts a single integer or a comma-separated list of integers.
    """
    try:
        # Attempt to parse as a single integer
        return int(value)
    except ValueError:
        # Parse as a list of integers
        try:
            return [int(x) for x in value.split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid value for layer_pred_arg: {value}. Must be an integer or a comma-separated list of integers."
            )

parser = argparse.ArgumentParser(description="Arguments for training")
parser.add_argument('var_arg', type=parse_var_arg, help='Name of the prediction and target layer')
parser.add_argument('layer_pred_arg', type=parse_layer_pred, help='Number of the prediction layer')
parser.add_argument('layer_target_arg', type=parse_layer_pred, help='Number of the target layer')
args = parser.parse_args()

if isinstance(args.var_arg, str):  # Single string
    var_list = [args.var_arg]
else:  # List of strings
    var_list = args.var_arg

if isinstance(args.layer_pred_arg, int):  # Single integer
    layer_pred_list = [args.layer_pred_arg]
else:  # List of integers
    layer_pred_list = args.layer_pred_arg


t0 = time.time()
experiment_name = "Attention_Res_UNet"

date_min_train = '1994_012'
date_max_train = '2017_361'

date_min_valid = '2018_003'
date_max_valid = '2018_360'

days_range = 7 # number of days between two restart
dates_learning = generate_dates(date_min_train, date_max_train, days_range)
#print('date list of learning days:')
#print(dates_learning)
# print('date list of restart converted in cice format:')
# converted_dates = [convert_date_format(date) for date in dates_list]
# print(converted_dates)
#print(f'number of training days: {len(dates_learning)}')
# print(f'To notice: the last date generated in the list is: {dates_list[-1]}')
dates_valid = generate_dates(date_min_valid, date_max_valid, days_range)
#print('date list of validation days:')
#print(dates_valid)
print(f'number of validation days: {len(dates_valid)}')

#
paths = {}
root_data_dir = 'data/NEW_data/'
root_output_dir = 'outputs'

paths["data_residuals"] = os.path.join(root_data_dir,"HR")
paths["data_LR_upsampled"] = os.path.join(root_data_dir,"LR_upsampled")

paths["forcings"] = root_data_dir
paths["standard"] = root_data_dir
paths["outputs"] = root_output_dir
paths["model_weights"] = os.path.join(root_output_dir,"Model_weights",experiment_name)
paths["checkpoints"] = os.path.join(root_output_dir,"Model_weights",experiment_name,"Checkpoints")
#
for var in paths:
    if os.path.isdir(paths[var]) == False:
        os.system("mkdir -p " + paths[var])
#
file_standardization_LR_upsampled = os.path.join(paths["standard"],"standard_LR_upsampled.h5")
file_standardization_res = os.path.join(paths["standard"],"standard_HR.h5")
file_standardization_bathy = os.path.join(paths["standard"],"standard_bathy_HR.h5")
file_standardization_ssh_LR_upsampled = os.path.join(paths["standard"],"standard_ssh_LR_upsampled.h5")

if isinstance(args.layer_pred_arg, int):
    predictors_part = f"{args.layer_pred_arg}"
else:
    predictors_part = "_".join(map(str, args.layer_pred_arg))

checkpoint_file_name = f"Checkpoints_3V201_{var_list[0]}_Pred-{predictors_part}_Target-{args.layer_target_arg}.h5"

# Example: Suppose paths dictionary is defined somewhere
file_checkpoints = os.path.join(paths["checkpoints"], checkpoint_file_name)

#
if os.path.isfile(file_checkpoints) == True:
    os.system("rm " + file_checkpoints)


# In[5]:


## The predictor must be of the form 'varname-layer-xx' or 'varname-cat-xx' if they 
## have an ice category (varname is the same as in the restart and iced files
## layers is between 1 and 50 and cat is between 1 and 5
## exceptions : "tp5_mask", "tp5_bathy", "tp5_lat", "iceumask", "aicenSumMask", "aicenSumMask015", "ssh_upsampled"

#if isinstance(args.layer_pred_arg, int):
#    list_predictors = [f"{args.var_arg}-layer-{args.layer_pred_arg}"]
#else:  # Handle if layer_pred_arg is a list of integers
#    list_predictors = [f"{args.var_arg}-layer-{x}" for x in args.layer_pred_arg]

#if isinstance(args.layer_target_arg, int):
#    list_targets = [f"{args.var_arg}-layer-{args.layer_target_arg}"]
#else:  # Handle if layer_pred_arg is a list of integers
#    list_targets = [f"{args.var_arg}-layer-{x}" for x in args.layer_target_arg]

# Generate list_predictors with all combinations of var_arg and layer_pred_arg
list_predictors = [
    f"{var}-layer-{layer_pred}" for var in var_list for layer_pred in layer_pred_list
]

if isinstance(args.layer_target_arg, int):  # Single integer
    layer_target_list = [args.layer_target_arg]
else:  # List of integers
    layer_target_list = args.layer_target_arg

# Generate list_targets using only the first element of var_arg
list_targets = [
    f"{var_list[0]}-layer-{layer_target}" for layer_target in layer_target_list
]

list_predictors += ['tp5_mask']

training_1_1 = 0

if training_1_1 : #If we're doing a 1_1 training, we here define how many layers we want to train on and wich one will be used for the test/plots
    n_layers_training = [1, 5, 10, 15, 25]
else :
    n_layers_training = None

print('list predictors : ', list_predictors)
print('NB: if the target is an ice variable then the ocean predictors layer is always set to 1 in the data generator') 
print('list targets : ', list_targets)

#
model_params = {"list_predictors": list_predictors,
                "list_targets": list_targets, 
                "dim": (760, 800), # (jdm,idm)
                "cropped_dim": (768, 800), #
                "batch_size": 4,
                "n_filters": [32*(i+1) for i in range(6)], #Ref Cyril: 32
                "activation": "relu",
                "kernel_initializer": "he_normal",
                "batch_norm": True,
                "pooling_type": "Average", # Average
                "dropout": 0.1,
               }
#

standard_LR_upsampled = load_standardization_data(file_standardization_LR_upsampled)
standard_res = load_standardization_data(file_standardization_res)
standard_bathy = load_standardization_data(file_standardization_res)
standard_ssh_LR_upsampled = load_standardization_data(file_standardization_res)
standard_forcings = load_standardization_data(file_standardization_LR_upsampled) #DUMMY FILE


params_train = {"list_predictors": model_params["list_predictors"],
                "list_labels": model_params["list_targets"],
                "list_dates": dates_learning,
                "standard_forcings": standard_forcings,
                "standard_res": standard_res,
                "standard_LR_upsampled": standard_LR_upsampled,
                "standard_bathy": standard_bathy,
                "standard_ssh_LR_upsampled": standard_ssh_LR_upsampled,
                "batch_size": model_params["batch_size"],
                "path_forcings": paths["forcings"],
                "path_data_res": paths["data_residuals"],
                "path_data_LR_upsampled": paths["data_LR_upsampled"],
                "dim": model_params["dim"],
                "cropped_dim": model_params["cropped_dim"],
                "shuffle": True,
                "res_normalization":0,
                "n_layers": n_layers_training
                }
#
params_valid = {"list_predictors": model_params["list_predictors"],
                "list_labels": model_params["list_targets"],
                "list_dates": dates_valid,
                "standard_forcings": standard_forcings,
                "standard_res": standard_res,
                "standard_LR_upsampled": standard_LR_upsampled,
                "standard_bathy": standard_bathy,
                "standard_ssh_LR_upsampled": standard_ssh_LR_upsampled,
                "batch_size": model_params["batch_size"],
                "path_forcings": paths["forcings"],
                "path_data_res": paths["data_residuals"],
                "path_data_LR_upsampled": paths["data_LR_upsampled"],
                "dim": model_params["dim"],
                "cropped_dim": model_params["cropped_dim"],
                "shuffle": True,
                "res_normalization":0,
                "n_layers": n_layers_training
                }
#
train_generator = Data_generator(**params_train)
valid_generator = Data_generator(**params_valid)


# In[7]:
class WarmUpAndDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, warmup_steps, decay_schedule_fn):
        super(WarmUpAndDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.warmup_steps = warmup_steps
        self.decay_schedule_fn = decay_schedule_fn

    def __call__(self, step):
        # Warmup phase
        def warmup_lr_fn():
            return self.initial_learning_rate + (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)) * (
                compile_params["max_learning_rate"] - self.initial_learning_rate)
        
        # Decay phase
        def decay_lr_fn():
            return self.decay_schedule_fn(step - self.warmup_steps)

        return tf.cond(step < self.warmup_steps, warmup_lr_fn, decay_lr_fn)

n_epochs = 100
scheduler = 'warmup_decay'

if scheduler == 'warmup_decay':
    compile_params = {
        "initial_learning_rate": 0.0, 
        "max_learning_rate": 0.01, # 0.01
        "warmup_steps": 1000, 
        "decay_steps": 10000, 
        "decay_rate": 0.96, 
        "staircase": False, 
        "n_epochs": n_epochs,
    }

    # Exponential Decay Schedule (used after warmup)
    decay_lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=compile_params["max_learning_rate"],
        decay_steps=compile_params["decay_steps"],
        decay_rate=compile_params["decay_rate"],
        staircase=compile_params["staircase"])

    # Combined Warmup and Decay Schedule
    lr_schedule = WarmUpAndDecay(
        initial_learning_rate=compile_params["initial_learning_rate"],
        warmup_steps=compile_params["warmup_steps"],
        decay_schedule_fn=decay_lr)

    # Use this combined lr_schedule in the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

def masked_Huber(delta, weight):
    def Huber_loss(y_true, y_pred):
        # Cropping the true and predicted tensors to exclude certain rows (like in RMSE)
        cropped_y_true = y_true[:, 0:-8, :, :]
        cropped_y_pred = y_pred[:, 0:-8, :, :]
        weight_extended = tf.expand_dims(weight, axis=-1)
        # Applying the mask to both true and predicted values
        weighted_diff = (cropped_y_true - cropped_y_pred) * weight_extended
        # Huber loss implementation
        abs_diff = tf.abs(weighted_diff)
        quadratic = 0.5 * tf.square(abs_diff)
        linear = delta * (abs_diff - 0.5 * delta)

        # If |difference| < delta, use quadratic, otherwise use linear
        huber_loss = tf.where(abs_diff <= delta, quadratic, linear)
        # Return the mean Huber loss
        return tf.reduce_mean(huber_loss)
    
    return Huber_loss
delta = 1.0

def masked_RMSE(weight):
    def RMSE_loss(y_true, y_pred):
        cropped_y_true = y_true[:, 0:-8, :, :]
        cropped_y_pred = y_pred[:, 0:-8, :, :]
        weighted_diff = (cropped_y_true - cropped_y_pred) * tf.expand_dims(weight, axis=-1)
        return tf.sqrt(tf.reduce_mean(tf.square(weighted_diff)))
    return RMSE_loss

tp5_mask = np.load( os.path.join(root_data_dir,'tp5mask.npy') )
only_atl = 0
if only_atl:
    print('using Atlantic mask')
    tp5_mask_atl = np.copy(tp5_mask)
    tp5_mask_atl[:] = 1
    tp5_mask_atl[0:340, 300:] = tp5_mask[0:340, 300:]
    tp5_mask = 1 - tp5_mask_atl
    tp5_mask = tf.convert_to_tensor(tp5_mask, dtype=tf.float32)
    tp5_mask = tf.expand_dims(tp5_mask, axis=0)
else:
    tp5_mask = 1 - tp5_mask
    tp5_mask = tf.convert_to_tensor(tp5_mask, dtype=tf.float32)
    tp5_mask = tf.expand_dims(tp5_mask, axis=0)

import matplotlib.pyplot as plt
# Plot the array
plt.figure(figsize=(6, 5))
plt.imshow(np.squeeze(tp5_mask), cmap='viridis', origin='lower')
plt.colorbar(label='tp5_mask values')
plt.title('tp5_mask')
plt.tight_layout()

# Save as PNG (or any supported format like .pdf, .jpg)
plt.savefig('tp5_mask.png', dpi=300)
plt.close()

tp5_rmse_loss = masked_RMSE(tp5_mask)
tp5_huber_loss = masked_Huber(delta, tp5_mask)

from optimizer_AdamW import AdamWOptimizer
opt = AdamWOptimizer(weight_decay=1e-4, learning_rate = lr_schedule)
#
unet_model = Att_Res_UNet(**model_params).make_unet_model()
unet_model.compile(loss=tp5_huber_loss, metrics = [tp5_rmse_loss], optimizer = opt)

print("Model compiled")
#
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = file_checkpoints, save_weights_only = True,
                                                monitor = 'val_loss', mode = 'min', verbose = 2,
                                                save_best_only = True)


model_history = unet_model.fit(train_generator, validation_data = valid_generator, 
                               epochs = compile_params["n_epochs"], verbose = 2, 
                               callbacks = [checkpoint])
print("Model fitted")


# In[8]:

import pickle
filename_model = f'UNet.h5'
unet_model.save_weights(os.path.join(root_output_dir,filename_model))
#

training_history_file_name = f"Training_history3V201_{args.var_arg[0]}_Pred-{predictors_part}_Target-{args.layer_target_arg}.pkl"

file_model_training_history = os.path.join(paths["outputs"], training_history_file_name)
pickle.dump(model_history.history, open(file_model_training_history, "wb"))
#
t1 = time.time()
dt = t1 - t0
print("Computing time: " + str(dt) + " seconds")
