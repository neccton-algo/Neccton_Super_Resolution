import os
import sys
import numpy as np
import numpy.ma as ma
path = os.path.expanduser('~/testhycom/NERSC-HYCOM-CICE/pythonlibs/abfile/')
sys.path.append(path)
import abfile
from netCDF4 import Dataset
from scipy.spatial import cKDTree
import interpolation
import argparse
from datetime import datetime, timedelta

def convert_date_format(input_date):
    date_format_input = '%Y_%j'
    date_format_output = '%Y-%m-%d'
    
    # Parse the input date in the '2000_364' format
    input_datetime = datetime.strptime(input_date, date_format_input)

    # Convert the date to the desired format 'YYYY-mm-day'
    output_date = input_datetime.strftime(date_format_output)

    return output_date

def get_number_from_date(date_str):
    # Convert the date string to a datetime object
    target_date = datetime.strptime(date_str, '%Y-%m-%d')

    # Define the reference date
    reference_date = datetime(1950, 1, 1)

    # Calculate the difference in days between the target date and the reference date
    delta_days = (target_date - reference_date).days

    return delta_days

def downsample_restarts(source_path, destination_path, date, start_mem, end_mem):
    """
    Upsamples restart files from a source directory and saves them to a destination directory.

    Parameters:
        source_path (str): The path to the source directory containing the restart files.
        destination_path (str): The path to the destination directory where upsampled files will be saved.
        start_mem (int): The starting memory number.
        end_mem (int): The ending memory number.
    """
    date = str(date)
    date_cice = convert_date_format(str(args.date))
#    julday = str(get_number_from_date(date_cice))

    for mem in range(start_mem, end_mem + 1):
        mem_str = f"{mem:03}"
        file = os.path.join(source_path, f'restart.{date}_00_0000_mem{mem_str}.a')
        destination_file = os.path.join(destination_path, f'restart.{date}_00_0000_mem{mem_str}.a')
 #       TP2_forecast_file = '/cluster/work/users/antber/TP2_Reanalysis/TOBACKUP/' + julday + f'/FORECAST/restart.{date}_00_0000_mem{mem_str}.a'
        interpolation.downsample_restart_V2(file, destination_file)

def downsample_icefiles(source_path, destination_path, date, start_mem, end_mem):
    """ 
    Upsamples restart files from a source directory and saves them to a destination directory.

    Parameters:
        source_path (str): The path to the source directory containing the restart files.
        destination_path (str): The path to the destination directory where upsampled files will be saved.        start_mem (int): The starting memory number.
        end_mem (int): The ending memory number.
    """
    date = str(date)
    date_cice = convert_date_format(date)
    for mem in range(start_mem, end_mem + 1):
        mem_str = f"{mem:03}"
        file = os.path.join(source_path, f'iced.{date_cice}-00000_mem{mem_str}.nc')
        destination_file = os.path.join(destination_path, f'iced.{date_cice}-00000_mem{mem_str}.nc')        
        interpolation.downsample_ncfile(file, destination_file)

if __name__ == "__main__":
    # Define the argument parser
    parser = argparse.ArgumentParser(description="Upsample restart files from a range of members indices.")
    parser.add_argument('date', type=str, help='The date in format yyyy_ddd, for instance 2019_273.')
    parser.add_argument('start_mem', type=int, help='The starting member number (inclusive).')
    parser.add_argument('end_mem', type=int, help='The ending member number (inclusive).')

    # Parse the command-line arguments
    args = parser.parse_args()
    date_cice = convert_date_format(str(args.date))
    julday = str(get_number_from_date(date_cice))

    # Set source and destination paths
    source_path = '/cluster/work/users/antber/from_Elio/Files/data/'
    #source_path = '/cluster/work/users/antber/TP5a0.06/expt_01.4/data'
    destination_path = '/cluster/work/users/antber/TP2a0.10/expt_02.5/downsampled_assimilated_SR_fields/'

    # Call the function with command-line arguments
    downsample_restarts(source_path, destination_path, args.date, args.start_mem, args.end_mem)

    source_path_ice = '/cluster/work/users/antber/from_Elio/Files/data/cice/'    
    #source_path_ice = '/cluster/work/users/antber/TP5a0.06/expt_01.4/data/cice/'
    destination_path_ice = '/cluster/work/users/antber/TP2a0.10/expt_02.5/downsampled_assimilated_SR_fields/cice/'
    downsample_icefiles(source_path_ice, destination_path_ice, args.date, args.start_mem, args.end_mem)
