## Imports, plot function and masks fields

import numpy as np
import numpy.ma as ma
                            
import sys
import os
path = os.path.expanduser('~/testhycom/NERSC-HYCOM-CICE/pythonlibs/abfile/')
sys.path.append(path)
import abfile
from netCDF4 import Dataset

import shutil
from scipy.spatial import cKDTree

from scipy.interpolate import RegularGridInterpolator

tp2mask = np.load('/cluster/work/users/antber/TP5a0.06/grid/tp2mask.npy')
Tp5mask = np.load('/cluster/work/users/antber/TP5a0.06/grid/tp5mask.npy')

def upsample_array_V2(original_array, method, target_shape=(760,800)):
    n_rows, n_cols = original_array.shape
    x = np.linspace(0, 359, n_rows)
    y = np.linspace(0, 399, n_cols)
    interp = RegularGridInterpolator((x, y), original_array, method=method, bounds_error=False, fill_value=None)
    x_new = np.linspace(0, 359, target_shape[0])
    y_new = np.linspace(0, 399, target_shape[1])
    xg_new, yg_new = np.meshgrid(x_new, y_new, indexing='ij')
    return interp((xg_new, yg_new))

def downsample_array_V2(original_array, method, target_shape=(380,400)):
    n_rows, n_cols = original_array.shape
    x = np.linspace(0, 759, n_rows)
    y = np.linspace(0, 799, n_cols)
    interp = RegularGridInterpolator((x, y), original_array, method, bounds_error=False, fill_value=None)
    x_new = np.linspace(0, 759, target_shape[0])
    y_new = np.linspace(0, 799, target_shape[1])
    xg_new, yg_new = np.meshgrid(x_new, y_new, indexing='ij')
    return interp((xg_new, yg_new))

def upsample_field_V2(field, fieldname, k, method):
## For the dp: no multiplicative factor but extend the first non 0 dp / iteratively closes dp to reach bathymetry
## (see downsample_restart_V2)
    if fieldname == 'dp':
        if k < 10: ## these layers are fixed: we copy them from another safe TP2 restart file
            tp5_template_dp = abfile.ABFileRestart('/cluster/work/users/antber/dummy/restartTP5/restart.2019_267_00_0000.a',"r",idm=800,jdm=760)
            field_template_dp = tp5_template_dp.read_field(fieldname,k,1) ## nb: tlevel doesn't matter here
            fieldHR = field_template_dp.data
            fieldHR[np.where(Tp5mask==1)] = 0
            tp5_template_dp.close()
        else:
            if isinstance(field, np.ma.MaskedArray):
                   fieldData = field.data
            else:
                   fieldData = field
            fieldData[np.where(tp2mask==1)] = np.nan
            fieldHR = upsample_array_V2(fieldData, method)
    
            fieldHR[np.where(Tp5mask==1)] = 0

            # Find the indices where NaN values occur
            nan_indices = np.isnan(fieldHR.data)
            
            # Coordinates of the non nan values inside the field
            coordinates = np.column_stack(np.where(np.logical_and(~Tp5mask.astype(bool), ~nan_indices)))
            tree = cKDTree(coordinates)
            # nearest indices for each nan
            nearest_indices = tree.query(np.column_stack(np.where(nan_indices)))[1]
            # according values
            nearest_values = fieldHR[coordinates[nearest_indices][:, 0], coordinates[nearest_indices][:, 1]]
    
            fieldHR[nan_indices] = nearest_values
    else:
        if isinstance(field, np.ma.MaskedArray):
               fieldData = field.data
        else:
               fieldData = field
        fieldData[np.where(tp2mask==1)] = np.nan
        fieldHR = upsample_array_V2(fieldData, method)
    
        fieldHR[np.where(Tp5mask==1)] = 0
    
        # Find the indices where NaN values occur
        nan_indices = np.isnan(fieldHR.data)
    
        # Coordinates of the non nan fill in values
        coordinates = np.column_stack(np.where(np.logical_and(~Tp5mask.astype(bool), ~nan_indices)))
        tree = cKDTree(coordinates)
        # nearest indices for each nan
        nearest_indices = tree.query(np.column_stack(np.where(nan_indices)))[1]
        # according values
        nearest_values = fieldHR[coordinates[nearest_indices][:, 0], coordinates[nearest_indices][:, 1]]
    
        fieldHR[nan_indices] = nearest_values
    return fieldHR

def upsample_restart_V2(file_path,destination_path):
    # Reading LR restart file
    fileLR = file_path
    LRtmp = abfile.ABFileRestart(fileLR,"r",idm=400,jdm=380)

    # Create new LR ab file for residuals
    newfile = destination_path
    new_abfile = abfile.ABFileRestart(newfile,"w",idm=800,jdm=760)
    new_abfile.write_header(22, LRtmp._iversn, LRtmp._yrflag, LRtmp._sigver, LRtmp._nstep, LRtmp._dtime, LRtmp._thbase)

    # A temp abfile to store the dp until final post process
    newfiledp = destination_path[:-2] + 'tmp.a'
    new_abfiledp = abfile.ABFileRestart(newfiledp,"w",idm=800,jdm=760)
    new_abfiledp.write_header(22, LRtmp._iversn, LRtmp._yrflag, LRtmp._sigver, LRtmp._nstep, LRtmp._dtime, LRtmp._thbase)

    tp5_template_dp = abfile.ABFileRestart('/cluster/work/users/antber/dummy/restartTP5/restart.2019_267_00_0000.a',"r",idm=800,jdm=760)

    # Just to count the number of tlevel
    unique_tlevels = set()
    # First downsampling for the dp in new_abfiledp
    # Also copy the pbot variable from the same template tp5 file
    for keys in sorted( LRtmp.fields.keys() ):
        fieldname = LRtmp.fields[keys]["field"]
        k         = LRtmp.fields[keys]["k"]
        t         = LRtmp.fields[keys]["tlevel"]
        unique_tlevels.add(t)
        if fieldname == 'dp':
            field     = LRtmp.read_field(fieldname,k,t)
            new_abfiledp.write_field(upsample_field_V2(field, fieldname, k, 'linear'), True, fieldname, k, t)

    new_abfiledp.close()
    new_abfiledp = abfile.ABFileRestart(newfiledp,"r",idm=800,jdm=760)

    # Then post process the dp to reach bathymetry
    dp3d = np.zeros((760,800,50))
    for keys in sorted( LRtmp.fields.keys() ):
        fieldname = LRtmp.fields[keys]["field"]
        k         = LRtmp.fields[keys]["k"]
        t         = LRtmp.fields[keys]["tlevel"]
        field     = LRtmp.read_field(fieldname,k,t)
        if fieldname == 'dp':
            dp3d[:,:,k-1] = new_abfiledp.read_field(fieldname,k,1)/9806 ## dp3d in meters

    ## Post process for the dp
    ## The sum of the dp is not equal everywhere to the TP2 bathymetry because of this:
    ## After the 9 first dp, If the dp is 0 in HR but not in LR the adjustement multiplication could not have happened
    ## So here we find these points and open the 10th dp with the missing depth

    ## Find the points where sum(dp) > bathy
    # in this case : decrease iteratively the dp from bottom to top until reaching correct bathymetry
    bathyHR = '/cluster/work/users/antber/TP5a0.06/topo/depth_TP5a0.06_05.a'
    bathyfileHR = abfile.ABFileBathy(bathyHR,"r",idm=800,jdm=760)
    bathyfieldHR = bathyfileHR.read_field("depth")
    dp_sum = np.sum(dp3d, axis=2)
    diff = dp_sum - bathyfieldHR

    positive_diff_indices = np.where(diff > 1e-7)
    for i, j in zip(positive_diff_indices[0], positive_diff_indices[1]):
        total_diff = diff[i, j]
        # Loop over layers 50 to 10
        for layer in range(49, 0, -1): ## Nb: no need to do range(49, 0, -1) because we know the first layers will
                                        # reach the correct bathymetry
            if total_diff <= 0:
                break
            # Determine how much we can decrease this layer
            decrease_amount = min(dp3d[i, j, layer], total_diff)
            dp3d[i, j, layer] -= decrease_amount
            total_diff -= decrease_amount

    ## Find the points where sum(dp) < bathy
    # in this case : increase the first closed dp to reach bottom
    negative_diff_indices = np.where(diff < -1e-7)
    for i, j in zip(negative_diff_indices[0], negative_diff_indices[1]):
        total_diff = -diff[i, j]
        zeros_layers = np.where(dp3d[i, j, :] == 0)
        # Find the first zero layer from the top
        if len(zeros_layers[0]) == 0:
            zero_layer_indice = 49
        else:
            zero_layer_indice = np.where(dp3d[i, j, :] == 0)[0][0]
        dp3d[i, j, zero_layer_indice] += total_diff

    # Now write all the downsampled fields
    for keys in sorted( LRtmp.fields.keys() ):
            fieldname = LRtmp.fields[keys]["field"]
            k         = LRtmp.fields[keys]["k"]
            t         = LRtmp.fields[keys]["tlevel"]
            field     = LRtmp.read_field(fieldname,k,t)
            print('processing', fieldname, k, t)
            if fieldname == 'dp':
                new_abfile.write_field(dp3d[:, :, k-1]*9806,True,fieldname,k,t)
            elif fieldname == 'pbot':
                field     = tp5_template_dp.read_field(fieldname,k,t)
                new_abfile.write_field(field, True, fieldname, k, t)
            else:
                new_abfile.write_field(upsample_field_V2(field, fieldname, k, 'linear'), True, fieldname, k, t)
    
    tp5_template_dp.close()
    new_abfiledp.close()
    #print('removing', newfiledp, newfiledp[:-2] + '.b')
    os.remove(newfiledp)
    os.remove(newfiledp[:-2] + '.b')

    new_abfile.close()
    LRtmp.close()
    
def downsample_field_V2(field, fieldname, k, method):
## For the dp: no multiplicative factor but extend the first non 0 dp / iteratively closes dp to reach bathymetry
## (see downsample_restart_V2)
    if fieldname == 'dp':
        if k < 10: ## these layers are fixed: we copy them from another safe TP2 restart file
            tp2_template_dp = abfile.ABFileRestart('/cluster/work/users/antber/dummy/restartTP2/restart.2001_002_00_0000.a',"r",idm=400,jdm=380)
            field_template_dp = tp2_template_dp.read_field(fieldname,k,1) ## nb: tlevel doesn't matter here
            fieldLR = field_template_dp.data
            fieldLR[np.where(tp2mask==1)] = 0
            tp2_template_dp.close()
        else:
            if isinstance(field, np.ma.MaskedArray):
                   fieldData = field.data
            else:
                   fieldData = field
            fieldData[np.where(Tp5mask==1)] = np.nan
            fieldLR = downsample_array_V2(fieldData, method)
    
            fieldLR[np.where(tp2mask==1)] = 0
    
            # Find the indices where NaN values occur
            nan_indices = np.isnan(fieldLR.data)
            
            # Coordinates of the non nan values inside the field
            coordinates = np.column_stack(np.where(np.logical_and(~tp2mask.astype(bool), ~nan_indices)))
            tree = cKDTree(coordinates)
            # nearest indices for each nan
            nearest_indices = tree.query(np.column_stack(np.where(nan_indices)))[1]
            # according values
            nearest_values = fieldLR[coordinates[nearest_indices][:, 0], coordinates[nearest_indices][:, 1]]
    
            fieldLR[nan_indices] = nearest_values
    else:
        if isinstance(field, np.ma.MaskedArray):
               fieldData = field.data
        else:
               fieldData = field
        fieldData[np.where(Tp5mask==1)] = np.nan
        fieldLR = downsample_array_V2(fieldData, method)
    
        fieldLR[np.where(tp2mask==1)] = 0
    
        # Find the indices where NaN values occur
        nan_indices = np.isnan(fieldLR.data)
    
        # Coordinates of the non nan fill in values
        coordinates = np.column_stack(np.where(np.logical_and(~tp2mask.astype(bool), ~nan_indices)))
        tree = cKDTree(coordinates)
        # nearest indices for each nan
        nearest_indices = tree.query(np.column_stack(np.where(nan_indices)))[1]
        # according values
        nearest_values = fieldLR[coordinates[nearest_indices][:, 0], coordinates[nearest_indices][:, 1]]
    
        fieldLR[nan_indices] = nearest_values
    return fieldLR

def downsample_restart(file_path,destination_path):
    # Reading HR restart file
    fileHR = file_path
    HRtmp = abfile.ABFileRestart(fileHR,"r",idm=800,jdm=760)
    
    # Create new HR ab file for residual
    newfile = destination_path
    new_abfile = abfile.ABFileRestart(newfile,"w",idm=400,jdm=380)
    new_abfile.write_header(22, HRtmp._iversn, HRtmp._yrflag, HRtmp._sigver, HRtmp._nstep, HRtmp._dtime, HRtmp._thbase)

    for keys in sorted( HRtmp.fields.keys() ) :
        fieldname = HRtmp.fields[keys]["field"]
        k         = HRtmp.fields[keys]["k"]
        t         = HRtmp.fields[keys]["tlevel"]
        field     = HRtmp.read_field(fieldname,k,t)
        print('processing', fieldname)
        new_abfile.write_field(downsample_field(field,fieldname),True,fieldname,k,t)

    new_abfile.close()
    HRtmp.close()

def downsample_restart_V2(file_path,destination_path):
    # Reading HR restart file
    fileHR = file_path
    HRtmp = abfile.ABFileRestart(fileHR,"r",idm=800,jdm=760)

    # Create new HR ab file for residuals
    newfile = destination_path
    new_abfile = abfile.ABFileRestart(newfile,"w",idm=400,jdm=380)
    new_abfile.write_header(22, HRtmp._iversn, HRtmp._yrflag, HRtmp._sigver, HRtmp._nstep, HRtmp._dtime, HRtmp._thbase)

    # A temp abfile to store the dp until final post process
    newfiledp = destination_path[:-2] + 'tmp.a'
    new_abfiledp = abfile.ABFileRestart(newfiledp,"w",idm=400,jdm=380)
    new_abfiledp.write_header(22, HRtmp._iversn, HRtmp._yrflag, HRtmp._sigver, HRtmp._nstep, HRtmp._dtime, HRtmp._thbase)

    tp2_template_dp = abfile.ABFileRestart('/cluster/work/users/antber/dummy/restartTP2/restart.2001_002_00_0000.a',"r",idm=400,jdm=380)

    # Just to count the number of tlevel
    unique_tlevels = set()
    # First downsampling for the dp in new_abfiledp
    # Also copy the pbot variable from the same template tp2 file
    for keys in sorted( HRtmp.fields.keys() ):
        fieldname = HRtmp.fields[keys]["field"]
        k         = HRtmp.fields[keys]["k"]
        t         = HRtmp.fields[keys]["tlevel"]
        unique_tlevels.add(t)
        if fieldname == 'dp':
            field     = HRtmp.read_field(fieldname,k,t)
            new_abfiledp.write_field(downsample_field_V2(field, fieldname, k, 'linear'), True, fieldname, k, t)

    new_abfiledp.close()
    new_abfiledp = abfile.ABFileRestart(newfiledp,"r",idm=400,jdm=380)

    # Then post process the dp to reach bathymetry
    dp3d = np.zeros((380,400,50))
    for keys in sorted( HRtmp.fields.keys() ):
        fieldname = HRtmp.fields[keys]["field"]
        k         = HRtmp.fields[keys]["k"]
        t         = HRtmp.fields[keys]["tlevel"]
        field     = HRtmp.read_field(fieldname,k,t)
        if fieldname == 'dp':
            dp3d[:,:,k-1] = new_abfiledp.read_field(fieldname,k,1)/9806 ## dp3d in meters

    ## Post process for the dp
    ## The sum of the dp is not equal everywhere to the TP2 bathymetry because of this:
    ## After the 9 first dp, If the dp is 0 in HR but not in LR the adjustement multiplication could not have happened
    ## So here we find these points and open the 10th dp with the missing depth

    ## Find the points where sum(dp) > bathy
    # in this case : decrease iteratively the dp from bottom to top until reaching correct bathymetry
    bathyLR = '/cluster/work/users/antber/TP2a0.10/topo/depth_TP2a0.10_01.a'
    bathyfileLR = abfile.ABFileBathy(bathyLR,"r",idm=400,jdm=380)
    bathyfieldLR = bathyfileLR.read_field("depth")
    dp_sum = np.sum(dp3d, axis=2)
    diff = dp_sum - bathyfieldLR

    positive_diff_indices = np.where(diff > 1e-7)
    for i, j in zip(positive_diff_indices[0], positive_diff_indices[1]):
        total_diff = diff[i, j]
        # Loop over layers 50 to 10
        for layer in range(49, 0, -1): ## Nb: no need to do range(49, 0, -1) because we know the first layers will
                                        # reach the correct bathymetry
            if total_diff <= 0:
                break
            # Determine how much we can decrease this layer
            decrease_amount = min(dp3d[i, j, layer], total_diff)
            dp3d[i, j, layer] -= decrease_amount
            total_diff -= decrease_amount

    ## Find the points where sum(dp) < bathy
    # in this case : increase the first closed dp to reach bottom
    negative_diff_indices = np.where(diff < -1e-7)
    for i, j in zip(negative_diff_indices[0], negative_diff_indices[1]):
        total_diff = -diff[i, j]
        zeros_layers = np.where(dp3d[i, j, :] == 0)
        # Find the first zero layer from the top
        if len(zeros_layers[0]) == 0:
            zero_layer_indice = 49
        else:
            zero_layer_indice = np.where(dp3d[i, j, :] == 0)[0][0]
        dp3d[i, j, zero_layer_indice] += total_diff

    # Now write all the downsampled fields
    for keys in sorted( HRtmp.fields.keys() ):
        fieldname = HRtmp.fields[keys]["field"]
        k         = HRtmp.fields[keys]["k"]
        t         = HRtmp.fields[keys]["tlevel"]
        field     = HRtmp.read_field(fieldname,k,t)
        print('processing', fieldname, k, t)
        if fieldname == 'dp':
            new_abfile.write_field(dp3d[:, :, k-1]*9806,True,fieldname,k,t)
        elif fieldname == 'pbot':
            field     = tp2_template_dp.read_field(fieldname,k,t)
            new_abfile.write_field(field, True, fieldname, k, t)
        else:
            new_abfile.write_field(downsample_field_V2(field, fieldname, k, 'linear'), True, fieldname, k, t)
    
    tp2_template_dp.close()
    new_abfiledp.close()
    #print('removing', newfiledp, newfiledp[:-2] + '.b')
    os.remove(newfiledp)
    os.remove(newfiledp[:-2] + '.b')

    new_abfile.close()
    HRtmp.close()

def upsample_ncfile(file_path,destination_path):
    # Reading LR restart file
    fileLR = file_path
    nc_fileLR = Dataset(fileLR, 'r')

    # Create the HR new ice file
    new_nc_file = Dataset(destination_path, 'w', format='NETCDF3_CLASSIC')
    new_nc_file.setncatts(nc_fileLR.__dict__) # It should have the same attributes (time, time_forc ...)
    new_nc_file.createDimension('ni', nc_fileLR.dimensions['ni'].size*2) 
    new_nc_file.createDimension('nj', nc_fileLR.dimensions['nj'].size*2)
    new_nc_file.createDimension('ncat', nc_fileLR.dimensions['ncat'].size)
    
    for var_name in nc_fileLR.variables:
        print('processing ' + var_name)
        var = nc_fileLR.variables[var_name]
        if var.ndim == 2:
            # Create the variable in the new .nc file
            new_var = new_nc_file.createVariable(var_name, var.dtype, ('nj', 'ni',))
            if isinstance(var[:], np.ma.MaskedArray):
                   fieldLR = var[:].data
            else:
                   fieldLR = var[:]
            fieldHR = upsample_field(fieldLR,var_name)
            new_var[:] = fieldHR
            
        elif var.ndim == 3:
            # Create the variable in the new .nc file
            new_var = new_nc_file.createVariable(var_name, var.dtype, ('ncat', 'nj', 'ni',))
            for ncat in range(5):
                if isinstance(var[ncat,:], np.ma.MaskedArray):
                       fieldLR = var[ncat,:].data
                else:
                       fieldLR = var[ncat,:]                
                fieldHR = upsample_field(fieldLR,var_name)
                new_var[ncat,:] = fieldHR

    new_nc_file.close()
    nc_fileLR.close()

def downsample_ncfile(file_path,destination_path):
    # Reading HR restart file
    fileHR = file_path
    nc_fileHR = Dataset(fileHR, 'r')

    # Create the LR new ice file
    new_nc_file = Dataset(destination_path, 'w', format='NETCDF3_CLASSIC')
    new_nc_file.setncatts(nc_fileHR.__dict__) # It should have the same attributes (time, time_forc ...)
    new_nc_file.createDimension('ni', nc_fileHR.dimensions['ni'].size/2) 
    new_nc_file.createDimension('nj', nc_fileHR.dimensions['nj'].size/2)
    new_nc_file.createDimension('ncat', nc_fileHR.dimensions['ncat'].size)
    
    for var_name in nc_fileHR.variables:
        # It is faster to open and close at each iteration than keeping the 2 files open,
        # maybe because memory issues ...
        new_nc_file.close()
        nc_fileHR.close()
        nc_fileHR = Dataset(fileHR, 'r')
        new_nc_file = Dataset(destination_path, 'a', format='NETCDF3_CLASSIC')
        print('processing ' + var_name)
        var = nc_fileHR.variables[var_name]
        if var.ndim == 2:
            # Create the variable in the new .nc file
            new_var = new_nc_file.createVariable(var_name, var.dtype, ('nj', 'ni',))
            if isinstance(var[:], np.ma.MaskedArray):
                   fieldHR = var[:].data
            else:
                   fieldHR = var[:]
            fieldLR = downsample_field(fieldHR,var_name)
            new_var[:] = fieldLR
            
        elif var.ndim == 3:
            # Create the variable in the new .nc file
            new_var = new_nc_file.createVariable(var_name, var.dtype, ('ncat', 'nj', 'ni',))
            for ncat in range(5):
                if isinstance(var[ncat,:], np.ma.MaskedArray):
                       fieldHR = var[ncat,:].data
                else:
                       fieldHR = var[ncat,:]                
                fieldLR = downsample_field(fieldHR,var_name)
                new_var[ncat,:] = fieldLR

    #new_nc_file.close()
    #nc_fileLR.close()
