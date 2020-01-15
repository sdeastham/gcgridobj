#!/usr/bin/env python3
import cubedsphere
from gcgridobj import gc_horizontal, gc_vertical
import numpy as np
import xesmf
import xarray
import warnings
import os

def reshape_cs_arb(cs_data):
    # Go from [...,6N,N] to [...,6,N,N]
    in_shape = cs_data.shape
    if in_shape[-2] == 6*in_shape[-1]:
       in_data = cs_data.copy()
       # Data is non-GMAO
       n_cs = in_shape[-1]
       out_shape = np.zeros(len(in_shape)+1,int)
       out_shape[:-3] = in_shape[:-2]
       out_shape[-3:] = [6,n_cs,n_cs]
       if in_shape == 2:
          # Data is 2-D
          out_data = np.reshape(in_data,out_shape)
       else:
          # Ugh
          n_other = int(np.product(in_shape[:-2]))
          in_reshape = np.reshape(in_data,[-1,n_cs*6,n_cs])
          out_reshape = np.zeros((in_reshape.shape[0],6,n_cs,n_cs))
          for i_other in range(n_other):
             out_reshape[i_other,:,:,:] = np.reshape(in_reshape[i_other,:,:],[1,6,n_cs,n_cs])
          out_data = np.reshape(out_reshape,out_shape)
       return out_data
    else:
       return cs_data

def reshape_cs(cs_data):
    return reshape_cs_arb(cs_data)
    ## Go from [6NxN] to [6xNxN]
    #if cs_data.shape[-2] == 6*cs_data.shape[-1]:
    #   full_data = cs_data.copy()
    #   # Data is non-GMAO
    #   n_cs = full_data.shape[-1]
    #   new_shape = [6,n_cs,n_cs]
    #   if len(full_data.shape) == 2:
    #      # Data is 2-D
    #      full_data = np.reshape(full_data,new_shape)
    #   else:
    #      # Ugh
    #      n_layers = full_data.shape[0]
    #      old_data = full_data
    #      full_data = np.zeros((n_layers,6,n_cs,n_cs))
    #      for i_layer in range(n_layers):
    #         full_data[i_layer,:,:,:] = np.reshape(old_data[i_layer,:,:],new_shape)
    #   return full_data
    #else:
    #   return cs_data

def unshape_cs_arb(cs_data):
    # Go from [...,6,N,N] to [...,6N,N]
    in_shape = cs_data.shape
    if in_shape[-2] == in_shape[-1]:
       # Data is GMAO
       in_data = cs_data.copy()
       n_cs = in_shape[-1]
       out_shape = np.zeros(len(in_shape)-1,int)
       out_shape[:-2] = in_shape[:-3]
       out_shape[-2:] = [6*n_cs,n_cs]
       if in_shape == 2:
          # Data is 2-D
          out_data = np.reshape(in_data,out_shape)
       else:
          # Ugh
          n_other = int(np.product(in_shape[:-3]))
          in_reshape = np.reshape(in_data,[-1,6,n_cs,n_cs])
          out_reshape = np.zeros((in_reshape.shape[0],6*n_cs,n_cs))
          for i_other in range(n_other):
             out_reshape[i_other,...] = np.reshape(in_reshape[i_other,...],[1,6*n_cs,n_cs])
          out_data = np.reshape(out_reshape,out_shape)
       return out_data
    else:
       return cs_data

def unshape_cs(cs_data):
    return unshape_cs_arb(cs_data)
    ## Go from [6xNxN] to [6NxN]
    #if cs_data.shape[-2] == cs_data.shape[-1]:
    #   full_data = np.squeeze(cs_data.copy())
    #   # Data is non-GMAO
    #   n_cs = full_data.shape[-1]
    #   new_shape = [6*n_cs,n_cs]
    #   if len(full_data.shape) == 3:
    #      # Data is 2-D
    #      full_data = np.reshape(full_data,new_shape)
    #   else:
    #      # Ugh
    #      n_layers = full_data.shape[0]
    #      old_data = full_data
    #      full_data = np.zeros((n_layers,6*n_cs,n_cs))
    #      for i_layer in range(n_layers):
    #         full_data[i_layer,:,:] = np.reshape(old_data[i_layer,:,:,:],new_shape)
    #   return full_data
    #else:
    #   return cs_data

def l2c(ll_data,cs_grid=None,ll_grid=None,regridder_list=None):
    '''
    # regrid lat-lon data to cubed sphere
    '''
    single_layer = len(ll_data.shape) == 2
    if single_layer:
       full_data = np.zeros((1,ll_data.shape[0],ll_data.shape[1]))
       full_data[0,:,:] = ll_data.copy()
    else:
       full_data = ll_data.copy()

    full_shape = full_data.shape
    n_lev = full_shape[0]

    if regridder_list is None:
       warnings.warn('Regridder list will become a required argument in a coming version of gcgridobj',FutureWarning)
       regridder_list = gen_l2c_regridder(cs_grid=cs_grid,ll_grid=ll_grid)

    # Get cs grid size from regridder_list
    out_shape = regridder_list[0]._grid_out.coords[0][0].shape
    n_cs = out_shape[0]

    cs_data = np.zeros((n_lev,6,n_cs,n_cs))
    for i_lev in range(n_lev):
       for i_face in range(6):
          cs_data[i_lev,i_face,:,:] = regridder_list[i_face](full_data[i_lev,:,:])

    if single_layer:
       cs_data = np.squeeze(cs_data) 

    return cs_data 

def c2c(cs_data,regridder_list=None):
    '''
    # regrid cubed sphere data to different cs resolution
    '''
    full_data = cs_data.copy()

    # Assume the CS data is 3D
    single_layer = len(full_data.shape) == 3
    if single_layer:
       layer_shape = list(full_data.shape)
       full_shape = layer_shape.copy()
       full_shape.insert(0,1)
       full_data = np.reshape(full_data,full_shape)
    else:
       layer_shape = full_data.shape[1:]

    full_shape = full_data.shape
    n_lev = full_shape[0]

    # Get all data from regridders
    out_shape = regridder_list[0]._grid_out.coords[0][0].shape
    n_cs_out = out_shape[-1]

    out_data = np.zeros((n_lev,6,n_cs_out,n_cs_out))
    for i_lev in range(n_lev):
       for i_face in range(6):
          out_data[i_lev,i_face,:,:] += regridder_list[i_face](full_data[i_lev,i_face,:,:])

    if single_layer:
       out_data = np.squeeze(out_data) 

    return out_data 

def c2l(cs_data,ll_grid=None,cs_grid=None,regridder_list=None):
    '''
    # regrid cubed sphere data to lat-lon
    '''
    full_data = cs_data.copy()

    # Assume the CS data is 3D
    single_layer = len(full_data.shape) == 3
    if single_layer:
       layer_shape = list(full_data.shape)
       full_shape = layer_shape.copy()
       full_shape.insert(0,1)
       full_data = np.reshape(full_data,full_shape)
    else:
       layer_shape = full_data.shape[1:]

    full_shape = full_data.shape
    n_lev = full_shape[0]

    if regridder_list is None:
       warnings.warn('Regridder list will become a required argument in a coming version of gcgridobj',FutureWarning)
       regridder_list = gen_c2l_regridder(cs_grid=cs_grid,ll_grid=ll_grid)

    # Get all data from regridders
    out_shape = regridder_list[0]._grid_out.coords[0][0].shape
    n_lon = out_shape[0]
    n_lat = out_shape[1]

    ll_data = np.zeros((n_lev,n_lat,n_lon))
    for i_lev in range(n_lev):
       for i_face in range(6):
          ll_data[i_lev,:,:] += regridder_list[i_face](full_data[i_lev,i_face,:,:])

    if single_layer:
       ll_data = np.squeeze(ll_data) 

    return ll_data 

def gen_regridder(grid_in,grid_out,method='conservative',grid_dir='.'):
    # What kind of grids are these?
    cs_in  = len(grid_in['lat'])  == 6
    cs_out = len(grid_out['lat']) == 6
    if cs_in and cs_out:
       # CS -> CS
       n_in  = grid_in['lat'][0].shape[0]
       n_out = grid_out['lat'][0].shape[0]
       if n_in == n_out: 
          # Grids are identical
          regrid_obj = None
       else:
          regridder_obj=[]
          with warnings.catch_warnings():
              warnings.filterwarnings("ignore", message="Input array is not F_CONTIGUOUS. Will affect performance.")
              # Assume the faces align
              for i_face in range(6):
                 sub_grid_in  = {'lat':   grid_in['lat'][i_face], 
                                 'lon':   grid_in['lon'][i_face],
                                 'lat_b': grid_in['lat_b'][i_face], 
                                 'lon_b': grid_in['lon_b'][i_face]}
                 sub_grid_out = {'lat':   grid_out['lat'][i_face], 
                                 'lon':   grid_out['lon'][i_face],
                                 'lat_b': grid_out['lat_b'][i_face], 
                                 'lon_b': grid_out['lon_b'][i_face]}
                 fname = os.path.join(grid_dir,'{:s}_c{:d}f{:d}_c{:d}f{:d}'.format(method,cs_in,i_face,cs_out,i_face))
                 regridder_obj.append(xesmf.Regridder(grid_in,grid_out,method=method,reuse_weights=True,filename=fname))
    elif cs_in:
       # CS -> LL
       regrid_obj = gen_c2l_regridder(cs_grid=grid_in,ll_grid=grid_out,method=method,grid_dir=grid_dir)
    elif cs_out:
       # LL -> CS
       regrid_obj = gen_l2c_regridder(cs_grid=grid_out,ll_grid=grid_in,method=method,grid_dir=grid_dir)
    else:
       # LL -> LL
       n_lon_in  = grid_in['lon'].size
       n_lat_in  = grid_in['lat'].size
       n_lon_out = grid_out['lon'].size
       n_lat_out = grid_out['lat'].size
       fname = os.path.join(grid_dir,'{:s}_{:d}x{:d}_{:d}x{:d}'.format(
                              method,n_lat_in,n_lon_in,n_lat_out,n_lon_out))
       regrid_obj = xesmf.Regridder(grid_in,grid_out,method=method,reuse_weights=True,
                                    filename=fname)
    return regrid_obj

def gen_l2c_regridder(cs_grid,ll_grid,method='conservative',grid_dir='.'):
    regridder_list=[]
    n_lon = ll_grid['lon'].size
    n_lat = ll_grid['lat'].size
    n_cs = cs_grid['lat'][0].shape[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input array is not F_CONTIGUOUS. Will affect performance.")
        for i_face in range(6):
           sub_grid = {'lat':   cs_grid['lat'][i_face], 
                       'lon':   cs_grid['lon'][i_face],
                       'lat_b': cs_grid['lat_b'][i_face], 
                       'lon_b': cs_grid['lon_b'][i_face]}
           fname = os.path.join(grid_dir,'{:s}_{:d}x{:d}_c{:d}f{:d}'.format(method,n_lat,n_lon,n_cs,i_face))
           regridder_list.append(xesmf.Regridder(ll_grid,sub_grid,method=method,reuse_weights=True,filename=fname))
    return regridder_list

def gen_c2l_regridder(cs_grid,ll_grid,method='conservative',grid_dir='.'):
    regridder_list=[]
    n_lon = ll_grid['lon'].size
    n_lat = ll_grid['lat'].size
    n_cs = cs_grid['lat'][0].shape[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input array is not F_CONTIGUOUS. Will affect performance.")
        for i_face in range(6):
           sub_grid = {'lat':   cs_grid['lat'][i_face], 
                       'lon':   cs_grid['lon'][i_face],
                       'lat_b': cs_grid['lat_b'][i_face], 
                       'lon_b': cs_grid['lon_b'][i_face]}
           fname = os.path.join(grid_dir,'{:s}_c{:d}f{:d}_{:d}x{:d}'.format(method,n_cs,i_face,n_lat,n_lon))
           regridder_list.append(xesmf.Regridder(sub_grid,ll_grid,method=method,reuse_weights=True,filename=fname))
    return regridder_list

def guess_ll_grid(ll_data_shape,is_nested=None,first_call=True):
    # Try to match a grid based only on its size [lat, lon]
    # Target not yet found
    is_target = False
    out_grid = None
    # First try global
    if is_nested is None or (not is_nested):
        for grid in gc_horizontal.global_grid_inventory:
            is_target = grid.lon.size == ll_data_shape[1] and grid.lat.size == ll_data_shape[0]
            if is_target:
                out_grid = grid
                break
    if not is_target and (is_nested is None or is_nested):
        for grid in gc_horizontal.nested_grid_inventory:
            is_target = grid.lon.size == ll_data_shape[1] and grid.lat.size == ll_data_shape[0]
            if is_target:
                out_grid = grid
                break
    if not is_target and first_call:
        # Try transposing but prevent recursion
        out_grid = guess_ll_grid([ll_data_shape[1],ll_data_shape[0]],is_nested,False)

    if out_grid is None and first_call:
       warnings.warn('Could not identify grid with size {:d}x{:d}'.format(ll_data_shape[0],ll_data_shape[1]))

    # Return result
    return out_grid

def guess_n_cs(cs_data_shape):
    # Is the data GMAO-style (6xNxN) or flat (6NxN)?
    is_gmao = len(cs_data_shape) == 3
    if is_gmao:
       assert cs_data_shape[1] == cs_data_shape[2], '3D CS data not square'
       assert cs_data_shape[0] == 6, '3D CS data must have 6 faces'
       n_cs = cs_data_shape[1]
    else:
       assert len(cs_data_shape) == 2, 'CS data must be 2D or 3D'
       assert cs_data_shape[0] == 6*cs_data_shape[1], '2D CS data must be 6NxN'
       n_cs = cs_data_shape[1]

    if not is_gmao:
       warnings.warn('Data is old format (shape is [6NxN]). Suggest using reshape_cs')
  
    return n_cs, is_gmao
    
def guess_cs_grid(cs_data_shape):
    # Made this consistent with guess_ll_grid
    n_cs, is_gmao = guess_n_cs(cs_data_shape)
    return cubedsphere.csgrid_GMAO(n_cs)
