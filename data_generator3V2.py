import os
from datetime import datetime, timedelta
import tensorflow as tf
import numpy as np
import abfile
from netCDF4 import Dataset
import re
tf.keras.utils.set_random_seed(1234)

# Data_generator
#     
#     list_predictors: list of predictors (list format)
#     list_labels: list of labels (list format)
#     list_dates: list of dates (list format)   
#     lead_time: lead time (starting at 0) in integer format
#     standard: dictionary containing the standardization statistics (mean, standard deviation, min, max)
#     batch_size: batch size (integer)
#     path_data: path where the data are located
#     dim: tuple of two dimensions indicating the dimensions of the input dataindicating the dimensions of the input data (y_dim, x_dim) 



class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, list_predictors, list_labels, list_dates, standard_res, standard_LR_upsampled, batch_size, path_data_res, path_data_LR_upsampled, dim, cropped_dim, shuffle, res_normalization):
        self.list_predictors = list_predictors
        self.list_labels = list_labels
        self.list_dates = list_dates
        self.standard_res = standard_res
        self.standard_LR_upsampled = standard_LR_upsampled
        self.batch_size = batch_size
        self.path_data_res = path_data_res
        self.path_data_LR_upsampled = path_data_LR_upsampled
        self.HR_dim = dim
        self.LR_dim = tuple([d // 2 for d in dim])
        self.HR_cropped_dim = cropped_dim
        self.LR_cropped_dim = tuple([d // 2 for d in cropped_dim])
        self.shuffle = shuffle
        self.res_normalization = res_normalization
        self.list_IDs = np.arange(len(list_dates))
        self.n_predictors = len(list_predictors)
        self.n_labels = len(list_labels)
        self.on_epoch_end()
        self.forcings = ["airtmp"]
        self.ice_variables = ["uvel", "vvel", "scale_factor", "swvdr", "swvdf", "swidr", "swidf", "strocnxT", "strocnyT",
    "stressp_1", "stressp_2", "stressp_3", "stressp_4", "stressm_1", "stressm_2", "stressm_3", "stressm_4",
    "stress12_1", "stress12_2", "stress12_3", "stress12_4", "frz_onset", "fsnow"]
        self.ocean_variables = ["u", "v", "dp", "temp", "saln", "ubavg", "vbavg", "pbavg", "pbot", "psikk", "thkk", "dpmixl"]
        self.ice_category_variables = [
    "aicen", "vicen", "vsnon", "Tsfcn", "iage", "FY", "alvl", "vlvl", "apnd", "hpnd", "ipnd", "dhs", "ffrac",
    "sice001", "qice001", "sice002", "qice002", "sice003", "qice003", "sice004", "qice004", "sice005", "qice005",
    "sice006", "qice006", "sice007", "qice007", "qsno001"]
        self.BGC_variables = ["flac","diac","cclc"]
        self.filenames = [self.get_filename_from_ID(ID) for ID in self.list_IDs]

    def get_filenames(self):
        """Return the list of filenames corresponding to the current order of samples."""
        return [self.get_filename_from_ID(ID) for ID in self.list_IDs]

    def get_filename_from_ID(self, ID):
        """Given an ID, return the corresponding filename."""
        return self.list_dates[ID]

    #
    def __len__(self): # Number of batches per epoch
        return int(np.ceil(len(self.list_IDs)) / self.batch_size)
    #
    def __getitem__(self, index): # Generate one batch of data
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Find list of IDs
        list_IDs_batch = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_batch)
        return(X, y)
        
    def get_denormalized(self,index): # Generate one bach of denormalized data
        X, y = self.__getitem__(index)
        X_denorm = np.zeros_like(X)
        y_denorm = np.zeros_like(y)
        for i, varname in enumerate(self.list_predictors):
            X_denorm[...,i] = X[...,i] * (self.standard[var][layer]['max'] - self.standard[var][layer]['min']) + self.standard[var][layer]['min']
        for i, varname in enumerate(self.list_labels):
            y_denorm[...,i] = y[...,i] * (self.standard[var][layer]['max'] - self.standard[var][layer]['min']) + self.standard[var][layer]['min']
        return (X_denorm, y_denorm)
            
    #
    def on_epoch_end(self): 
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(self.indexes)
    
    #
    def standardize(self, var_name, layer, var_data, resolution):
        if resolution == 'res':
            stand_data = (var_data - self.standard_res[var_name][layer]["mean"]) / self.standard_res[var_name][layer]["std"]
        elif resolution == 'LR':
            stand_data = (var_data - self.standard_LR_upsampled[var_name][layer]["mean"]) / self.standard_LR_upsampled[var_name][layer]["std"]
        return(stand_data)
    #
    def normalize(self, var_name, layer, var_data, resolution):
        if resolution == 'res':
            if (self.standard_res[var_name][layer]["max"] - self.standard_res[var_name][layer]["min"]) == 0:
                norm_data = var_data
            else:
                norm_data = (var_data - self.standard_res[var_name][layer]["min"]) / (self.standard_res[var_name][layer]["max"] - self.standard_res[var_name][layer]["min"])
        elif resolution == 'LR':
            if (self.standard_LR_upsampled[var_name][layer]["max"] - self.standard_LR_upsampled[var_name][layer]["min"]) == 0:
                norm_data = var_data
            else:
                norm_data = (var_data - self.standard_LR_upsampled[var_name][layer]["min"]) / (self.standard_LR_upsampled[var_name][layer]["max"] - self.standard_LR_upsampled[var_name][layer]["min"])
    
    def denormalize(self, var_name, layer, var_data,clip=False,vmin=0,vmax=100):
        if resolution == 'res':
            denorm_data = var_data * (self.standard_res[var_name][layer]["max"] - self.standard_res[var_name][layer]["min"]) + self.standard_res[var_name][layer]["min"]
        elif resolution == 'LR':
            denorm_data = var_data * (self.standard_LR_upsampled[var_name][layer]["max"] - self.standard_LR_upsampled[var_name][layer]["min"]) + self.standard_LR_upsampled[var_name][layer]["min"]
        if clip:
            denorm_data = np.clip(denorm_data,vmin, vmax)
        return denorm_data
    #
    def __data_generation(self, list_IDs_batch): # Generates data containing batch_size samples
        #
        # Initialization
        X = np.full((self.batch_size, *self.HR_cropped_dim, self.n_predictors), np.nan)
        y = np.full((self.batch_size, *self.HR_cropped_dim, self.n_labels), np.nan)
        
        # Create the column of zeros to add to the field to have the right dimensions for the Unet
        zeropadHR = np.zeros((self.HR_cropped_dim[0] - self.HR_dim[0], self.HR_dim[1]))
        
        tp5_mask = np.load( os.path.join(self.path_data_res,'tp5mask.npy') )
        mask_indices_tp5 = np.where(tp5_mask==1)
        
        # Generate data
        for i, ID in enumerate(list_IDs_batch):
            date_ID = self.list_dates[ID]
            
            # Generate Batch of predictors
            for v, var in enumerate(self.list_predictors):
                parts = var.split('-')
                var_name = parts[0]
                ## Search for digits at the end of the predictor, and find the layer or ice cat number
                match = re.search(r'(\d+)$', parts[-1])
                if match:
                    layer_number = int(match.group())
                    cat_match = re.search(r'cat-(\d+)$', var)
                    cat_number = int(cat_match.group(1))-1 if cat_match else None
                else:
                    layer_number = None
                    cat_number = None
                if var_name in self.ocean_variables:
                    # take field.data (with fill values of 1e30), crop it, gets the indices of the mask
                    # normalize and finally put the mask at 0
                    file_ID = os.path.join(self.path_data_LR_upsampled,f"restart.{date_ID}_00_0000.a")
                    ab_file = abfile.ABFileRestart(file_ID,"r",idm=self.HR_dim[1],jdm=self.HR_dim[0])
                    var_data = ab_file.read_field(var_name,layer_number,1).data
                    ab_file.close()
                    var_data = self.normalize(var_name, layer_number, var_data,'LR')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    X[i,:,:,v] = var_data
                elif var_name in self.forcings:
                    file_ID = os.path.join(self.path_forcings, var_name + ".a")
                    ab_file = abfile.ABFileForcing(file_ID,"r")
                    var_data = ab_file.read_field(var_name,forcings_date(date_ID)).data
                    ab_file.close()
                    var_data = self.normalize(var_name, 0, var_data,'forcings')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    X[i,:,:,v] = var_data
                elif var_name in self.ice_variables:
                    # same as for the ocean variable, but already 0 instead of fill in values for the mask
                    nc_file = Dataset(os.path.join(self.path_data_LR_upsampled,f"iced.{date_ID}_00_0000.nc"), 'r')
                    var_data =  nc_file.variables[var_name][:].data
                    nc_file.close()
                    var_data = self.normalize(var_name, cat_number, var_data,'LR')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    X[i,:,:,v] = var_data
                elif var_name in self.ice_category_variables: # variable with ice category
                    nc_file = Dataset(os.path.join(self.path_data_LR_upsampled,f"iced.{date_ID}_00_0000.nc"), 'r')
                    var_data =  nc_file.variables[var_name][cat_number,:].data
                    nc_file.close()
                    var_data = self.normalize(var_name, cat_number, var_data,'LR')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    X[i,:,:,v] = var_data
                elif var_name == 'tp5_mask': # We put a 0 on the land and a 1 on the ocean
                    X[i,:,:,v] = np.vstack((1-tp5_mask, zeropadHR)) 
                elif var_name in self.BGC_variables:
                    file_ID = os.path.join(self.path_data_LR_upsampled,f"restart.{date_ID}_00_0000.a")
                    ab_file = abfile.ABFileRestart(file_ID,"r",idm=self.HR_dim[1],jdm=self.HR_dim[0])
                    var_data = ab_file.read_field('ECO_'+var_name,layer_number,1).data
                    ab_file.close()
                    #var_data = self.normalize(var_name, layer_number, var_data,'LR')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    X[i,:,:,v] = var_data
                elif var_name == 'BGC':
                    file_ID = os.path.join(self.path_data_LR_upsampled,f"restart.{date_ID}_00_0000.a")
                    ab_file = abfile.ABFileRestart(file_ID,"r",idm=self.HR_dim[1],jdm=self.HR_dim[0])
                    var_data1 = ab_file.read_field('ECO_flac',layer_number,1).data
                    var_data2 = ab_file.read_field('ECO_diac',layer_number,1).data
                    var_data3 = ab_file.read_field('ECO_cclc',layer_number,1).data
                    var_data = var_data1 + var_data2 + var_data3
                    ab_file.close()
                    #var_data = self.normalize(var_name, layer_number, var_data,'LR')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    X[i,:,:,v] = var_data
                else:
                    raise Exception(f"Invalid predictor variable: {var_name}")
                    
            # Generate Batch of labels
            for v, var in enumerate(self.list_labels):
                parts = var.split('-')
                var_name = parts[0]
                ## Search for digits at the end of the predictor, and find the layer or ice cat number
                match = re.search(r'(\d+)$', parts[-1])
                if match:
                    layer_number = int(match.group())
                    cat_match = re.search(r'cat-(\d+)$', var)
                    cat_number = int(cat_match.group(1))-1 if cat_match else None
                else:
                    layer_number = None
                    cat_number = None
                if var_name in self.ocean_variables:
                    file_ID = os.path.join(self.path_data_res,f"restart.{date_ID}_00_0000.a")
                    ab_file = abfile.ABFileRestart(file_ID,"r",idm=self.HR_dim[1],jdm=self.HR_dim[0])
                    var_data = ab_file.read_field(var_name,layer_number,1).data
                    ab_file.close()
                    #var_data = var_data[0:self.HR_cropped_dim[0],0:self.HR_cropped_dim[1]]
                    if self.res_normalization == 1:
                        var_data = self.normalize(var_name, layer_number, var_data,'res')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    y[i,:,:,v] = var_data
                elif var_name in self.ice_variables:
                    nc_file = Dataset(os.path.join(self.path_data_res,f"iced.{date_ID}_00_0000.nc"), 'r')
                    var_data =  nc_file.variables[var_name][:].data
                    nc_file.close()
                    #var_data = var_data[0:self.HR_cropped_dim[0],0:self.HR_cropped_dim[1]]
                    if  self.res_normalization == 1:
                        var_data = self.normalize(var_name, cat_number, var_data,'res')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    y[i,:,:,v] = var_data
                elif var_name in self.ice_category_variables: # variable with ice category
                    nc_file = Dataset(os.path.join(self.path_data_res,f"iced.{date_ID}_00_0000.nc"), 'r')
                    var_data =  nc_file.variables[var_name][cat_number,:].data
                    nc_file.close()
                    #var_data = var_data[0:self.HR_cropped_dim[0],0:self.HR_cropped_dim[1]]
                    if  self.res_normalization == 1:
                        var_data = self.normalize(var_name, cat_number, var_data,'res')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    y[i,:,:,v] = var_data
                elif var_name in self.BGC_variables:
                    file_ID = os.path.join(self.path_data_res,f"restart.{date_ID}_00_0000.a")
                    ab_file = abfile.ABFileRestart(file_ID,"r",idm=self.HR_dim[1],jdm=self.HR_dim[0])
                    var_data = ab_file.read_field('ECO_'+var_name,layer_number,1).data
                    ab_file.close()
                    #if self.res_normalization == 1:
                    #    var_data = self.normalize(var_name, layer_number, var_data,'res')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    y[i,:,:,v] = var_data
                elif var_name == 'BGC':
                    file_ID = os.path.join(self.path_data_res,f"restart.{date_ID}_00_0000.a")
                    ab_file = abfile.ABFileRestart(file_ID,"r",idm=self.HR_dim[1],jdm=self.HR_dim[0])
                    var_data1 = ab_file.read_field('ECO_flac',layer_number,1).data
                    var_data2 = ab_file.read_field('ECO_diac',layer_number,1).data
                    var_data3 = ab_file.read_field('ECO_cclc',layer_number,1).data
                    var_data = var_data1 + var_data2 + var_data3
                    ab_file.close()
                    #if self.res_normalization == 1:
                    #    var_data = self.normalize(var_name, layer_number, var_data,'res')
                    var_data[mask_indices_tp5] = 0 # Put the mask at 0
                    var_data = np.vstack((var_data, zeropadHR))
                    y[i,:,:,v] = var_data
                else:
                    raise Exception(f"Invalid target variable: {var_name}")
                    
            #
        return(X, y)


# Generate a list of days under the format of the hycom restart file 'yyyy_ddd' between
# start_date and end_date with a spacing of days_range between each date
def generate_dates(start_date, end_date, days_range,shift = 0):
    date_format = '%Y_%j'
    start_datetime = datetime.strptime(start_date, date_format)
    end_datetime = datetime.strptime(end_date, date_format)
    dates_list = []

    current_date = start_datetime
    if shift != 0:
        current_date += timedelta(days=shift)
        end_datetime += timedelta(days=days_range)
    while current_date <= end_datetime:
        dates_list.append(current_date.strftime(date_format))
        current_date += timedelta(days=days_range)

    return dates_list

# Convert a day from the Hycom format 'yyyy_ddd' to the cice format 'yyyy_mm_dd' 
def convert_date_format(input_date):
    date_format_input = '%Y_%j'
    date_format_output = '%Y-%m-%d'
    
    # Parse the input date in the '2000_364' format
    input_datetime = datetime.strptime(input_date, date_format_input)

    # Convert the date to the desired format 'YYYY-mm-day'
    output_date = input_datetime.strftime(date_format_output)

    return output_date

def forcings_date(date_str):
    # Get the year and the day number from the string format 'year_day'
    year, day_of_year = map(int, date_str.split('_'))
    
    # Calculate the number of days since the beginning of the year
    days_since_start = day_of_year - 1
    
    # Calculate the number of days since January 1 1995
    days_since_1995 = (datetime(year, 1, 1) - datetime(1995, 1, 1)).days
    
    # in Hycom January 1 1995 corresponds to 34334
    total_days = 34334 + days_since_start + days_since_1995
    
    return total_days


import h5py
def load_standardization_data(file_standardization):    
    # Initialize an empty dictionary to store the loaded data
    loaded_statistics_dict = {}
    
    # Define a list of fields to not load 'M' and 'n'
    excluded_fields = ['M', 'n']
    
    # Open the HDF5 file for reading
    with h5py.File(file_standardization, 'r') as hdf5_file:
        # Access the group containing your data
        group = hdf5_file['my_dict_group']
    
        # Iterate through field groups
        for fieldname in group:
            field_group = group[fieldname]
            loaded_statistics_dict[fieldname] = {}
            
            # Iterate through k subgroups
            for k in field_group:
                k_group = field_group[k]
                loaded_statistics_dict[fieldname][int(k)] = {}
                
                # Iterate through statistics datasets
                for stat_name, stat_dataset in k_group.items():
                    if stat_name not in excluded_fields:
                        loaded_statistics_dict[fieldname][int(k)][stat_name] = stat_dataset[()]
    return loaded_statistics_dict
