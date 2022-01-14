#!/usr/bin/env python3
from gcgridobj import gc_horizontal, gc_vertical, cstools
import numpy as np
import xarray
import warnings
import os
import scipy.sparse

regrid_archive='regridders'

class vrt_regridder:
    def __init__(self,xmat):
        self.xmat = xmat
    def __call__(self,data):
        return regrid_vertical(data,self.xmat)

class regridder:
    def __init__(self,xe_regridder):
       self.xe_regridder = xe_regridder
    def __call__(self,data):
       return regrid(data,self.xe_regridder)

def regrid(in_data,regridder_obj):
    if regridder_obj is None:
       return in_data
    # Get the shapes
    elif isinstance(regridder_obj,list):
       # Input, output, or both are cubed sphere
       shape_in = regridder_obj[0].shape_in
       shape_out = regridder_obj[0].shape_out
       # ASSUMPTION: Square = CS; rectangular = LL
       cs_in  = shape_in[0] == shape_in[1]
       cs_out = shape_out[0] == shape_out[1]
       if cs_in and cs_out:
          return c2c_arb(in_data,regridder_obj)
       elif cs_in and (not cs_out):
          return c2l_arb(in_data,regridder_obj)
       elif (not cs_in) and cs_out:
          return l2c_arb(in_data,regridder_obj)
       else:
          raise ValueError('Cannot automatically determine appropriate regridding routine')
    else:
       # Lat-lon
       return l2l_arb(in_data,regridder_obj)

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

def l2c_arb(ll_data,regridder_list):
    '''
    # regrid lat-lon data to cubed sphere
    # Allows for arbitrary leading dimensions
    '''
    single_layer = len(ll_data.shape) == 2
    if single_layer:
       in_reshape = np.reshape(ll_data,[1]  + list(ll_data.shape))
    else:
       in_reshape = np.reshape(ll_data,[-1] + list(ll_data.shape[-2:]))

    # How many slices do we have?
    n_samples = in_reshape.shape[0]

    # Get all data from regridders
    try:
        n_cs_out = regridder_list[0]._grid_out.coords[0][0].shape[-1]
    except:
        n_cs_out = regridder_list[0].grid_out.coords[0][0].shape[-1]
    
    out_reshape = np.zeros((n_samples,6,n_cs_out,n_cs_out))
    for i_sample in range(n_samples):
       for i_face in range(6):
          out_reshape[i_sample,i_face,...] = regridder_list[i_face](in_reshape[i_sample,...])

    if single_layer:
       cs_data = out_reshape[0,...]
    else:
       cs_data = np.reshape(out_reshape,list(ll_data.shape[:-2]) + [6,n_cs_out,n_cs_out])

    return cs_data 

def l2c(ll_data,cs_grid=None,ll_grid=None,regridder_list=None):
    '''
    # regrid lat-lon data to cubed sphere
    '''
    #single_layer = len(ll_data.shape) == 2
    #if single_layer:
    #   full_data = np.zeros((1,ll_data.shape[0],ll_data.shape[1]))
    #   full_data[0,:,:] = ll_data.copy()
    #else:
    #   full_data = ll_data.copy()

    #full_shape = full_data.shape
    #n_lev = full_shape[0]

    if regridder_list is None:
       warnings.warn('Regridder list will become a required argument in a coming version of gcgridobj',FutureWarning)
       regridder_list = gen_l2c_regridder(cs_grid=cs_grid,ll_grid=ll_grid)

    ## Get cs grid size from regridder_list
    #out_shape = regridder_list[0]._grid_out.coords[0][0].shape
    #n_cs = out_shape[0]

    #cs_data = np.zeros((n_lev,6,n_cs,n_cs))
    #for i_lev in range(n_lev):
    #   for i_face in range(6):
    #      cs_data[i_lev,i_face,:,:] = regridder_list[i_face](full_data[i_lev,:,:])

    #if single_layer:
    #   cs_data = np.squeeze(cs_data) 

    #return cs_data 
    return l2c_arb(ll_data,regridder_list)

def c2c_arb(cs_data,regridder_list):
    '''
    Regrid cubed sphere data to different cs resolution
    Assumes data is [...,6,N,N] in shape
    '''
    full_data = cs_data.copy()

    single_layer = len(full_data.shape) == 3
    if single_layer:
       in_reshape = np.reshape(full_data,[1]  + list(full_data.shape))
    else:
       in_reshape = np.reshape(full_data,[-1] + list(full_data.shape[-3:]))

    # How many CS slices do we have?
    n_samples = in_reshape.shape[0]

    # Get all data from regridders
    try:
        n_cs_out = regridder_list[0]._grid_out.coords[0][0].shape[-1]
    except:
        n_cs_out = regridder_list[0].grid_out.coords[0][0].shape[-1]
    
    out_reshape = np.zeros((n_samples,6,n_cs_out,n_cs_out))
    for i_sample in range(n_samples):
       for i_face in range(6):
          out_reshape[i_sample,i_face,:,:] += regridder_list[i_face](
                                              in_reshape[i_sample,i_face,:,:])

    if single_layer:
       out_data = out_reshape[0,...]
    else:
       out_data = np.reshape(out_reshape,list(full_data.shape[:-3]) + 
                                         [6,n_cs_out,n_cs_out])

    return out_data 

def c2c(cs_data,regridder_list=None):
    '''
    # regrid cubed sphere data to different cs resolution
    '''
    return c2c_arb(cs_data,regridder_list)
    #full_data = cs_data.copy()

    ## Assume the CS data is 3D
    #single_layer = len(full_data.shape) == 3
    #if single_layer:
    #   layer_shape = list(full_data.shape)
    #   full_shape = layer_shape.copy()
    #   full_shape.insert(0,1)
    #   full_data = np.reshape(full_data,full_shape)
    #else:
    #   layer_shape = full_data.shape[1:]

    #full_shape = full_data.shape
    #n_lev = full_shape[0]

    ## Get all data from regridders
    #out_shape = regridder_list[0]._grid_out.coords[0][0].shape
    #n_cs_out = out_shape[-1]

    #out_data = np.zeros((n_lev,6,n_cs_out,n_cs_out))
    #for i_lev in range(n_lev):
    #   for i_face in range(6):
    #      out_data[i_lev,i_face,:,:] += regridder_list[i_face](full_data[i_lev,i_face,:,:])

    #if single_layer:
    #   out_data = np.squeeze(out_data) 

    #return out_data 

def c2l_arb(cs_data,regridder_list):
    '''
    # regrid cubed-sphere data to lat-lon
    # Allows for arbitrary leading dimensions
    '''
    single_layer = len(cs_data.shape) == 3
    if single_layer:
       in_reshape = np.reshape(cs_data,[1]  + list(cs_data.shape))
    else:
       in_reshape = np.reshape(cs_data,[-1] + list(cs_data.shape[-3:]))

    # How many slices do we have?
    n_samples = in_reshape.shape[0]

    # Get all data from regridders
    try:
        out_shape = regridder_list[0]._grid_out.coords[0][0].shape
    except:
        out_shape = regridder_list[0].grid_out.coords[0][0].shape
    # Note unusual ordering - coords are [lon x lat] for some reason
    n_lon = out_shape[0]
    n_lat = out_shape[1]
    
    out_reshape = np.zeros((n_samples,n_lat,n_lon))
    for i_sample in range(n_samples):
       for i_face in range(6):
          out_reshape[i_sample,...] += regridder_list[i_face](in_reshape[i_sample,i_face,...])

    if single_layer:
       ll_data = out_reshape[0,...]
    else:
       ll_data = np.reshape(out_reshape,list(cs_data.shape[:-3]) + [n_lat,n_lon])

    return ll_data 

def c2l(cs_data,ll_grid=None,cs_grid=None,regridder_list=None):
    '''
    # regrid cubed sphere data to lat-lon
    '''
    #full_data = cs_data.copy()

    ## Assume the CS data is 3D
    #single_layer = len(full_data.shape) == 3
    #if single_layer:
    #   layer_shape = list(full_data.shape)
    #   full_shape = layer_shape.copy()
    #   full_shape.insert(0,1)
    #   full_data = np.reshape(full_data,full_shape)
    #else:
    #   layer_shape = full_data.shape[1:]

    #full_shape = full_data.shape
    #n_lev = full_shape[0]

    if regridder_list is None:
       warnings.warn('Regridder list will become a required argument in a coming version of gcgridobj',FutureWarning)
       regridder_list = gen_c2l_regridder(cs_grid=cs_grid,ll_grid=ll_grid)

    ## Get all data from regridders
    #out_shape = regridder_list[0]._grid_out.coords[0][0].shape
    #n_lon = out_shape[0]
    #n_lat = out_shape[1]

    #ll_data = np.zeros((n_lev,n_lat,n_lon))
    #for i_lev in range(n_lev):
    #   for i_face in range(6):
    #      ll_data[i_lev,:,:] += regridder_list[i_face](full_data[i_lev,i_face,:,:])

    #if single_layer:
    #   ll_data = np.squeeze(ll_data) 

    #return ll_data 
    return c2l_arb(cs_data,regridder_list)

def l2l(in_data,regridder_obj):
    single_layer = len(in_data.shape) == 2
    if single_layer:
       in_reshape = np.reshape(in_data,[1]  + list(in_data.shape))
    else:
       in_reshape = np.reshape(in_data,[-1] + list(in_data.shape[-2:]))

    # How many slices do we have?
    n_samples = in_reshape.shape[0]

    try:
        out_shape = regridder_obj._grid_out.coords[0][0].shape
    except:
        out_shape = regridder_obj.grid_out.coords[0][0].shape
    # Note unusual ordering - coords are [lon x lat] for some reason
    n_lon = out_shape[0]
    n_lat = out_shape[1]
    
    out_reshape = np.zeros((n_samples,n_lat,n_lon))
    for i_sample in range(n_samples):
       out_reshape[i_sample,...] = regridder_obj(in_reshape[i_sample,...])

    if single_layer:
       out_data = out_reshape[0,...]
    else:
       out_data = np.reshape(out_reshape,list(in_data.shape[:-2]) + [n_lat,n_lon])

    return out_data

l2l_arb = l2l

def gen_regridder(grid_in,grid_out,method='conservative',grid_dir=None,make_obj=True):
    import xesmf
    if grid_dir is None:
       grid_dir = regrid_archive
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
          regrid_obj=[]
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
                 fname = os.path.join(grid_dir,'{:s}_c{:d}f{:d}_c{:d}f{:d}.nc'.format(method,n_in,i_face,n_out,i_face))
                 try:
                    regrid_obj.append(xesmf.Regridder(sub_grid_in,sub_grid_out,method=method,reuse_weights=os.path.isfile(fname),filename=fname))
                 except KeyError as e:
                    print('Regridder object generation failed. If you received a permission error, please check that the output directory {} exists'.format(grid_dir))
                    raise
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
       if ((n_lon_in == n_lon_out) and (n_lat_in == n_lat_out) and 
           (np.allclose(grid_in['lon'].values,grid_out['lon'].values)) and
           (np.allclose(grid_in['lat'].values,grid_out['lat'].values))):
           # In/out are identical
           regrid_obj = None
       else: 
           fname = os.path.join(grid_dir,'{:s}_{:s}_{:s}.nc'.format(
                                  method,gen_ll_name(grid_in),gen_ll_name(grid_out)))
           try:
              regrid_obj = xesmf.Regridder(grid_in,grid_out,method=method,reuse_weights=os.path.isfile(fname),
                                           filename=fname)
           except KeyError as e:
              print('Regridder object generation failed. If you received a permission error, please check that the output directory {} exists'.format(grid_dir))
              raise

    if make_obj:
       # Make it a little fancier...
       return regridder(regrid_obj)
    else:
       return regrid_obj

# Aliases
gen_l2l_regridder = gen_regridder
gen_c2c_regridder = gen_regridder

def gen_ll_name(grid):
    lat = np.asarray(grid['lat'])
    lon = np.asarray(grid['lon'])
    dlat = lat[1:] - lat[:-1]
    dlon = lon[1:] - lon[:-1]
    n_lon = lon.size
    n_lat = lat.size
    # Check regional
    if (np.median(dlat) * (n_lat+1)) < 179.9:
        #print(np.median(dlat),n_lat+1,np.median(dlat)*(n_lat+1),'PX')
        Pstr = 'PX'
    else:
        #print(lat[0],lat[-1],np.min(dlat),np.median(dlat))
        if (lat[0] < (-90.0 + 1.0e-10) or lat[-1] > (90 - 1.0e-10) or
                 (np.min(dlat) < 0.9 * np.median(dlat))):
            Pstr = 'PC'
        else:
            Pstr = 'PE'
    if np.median(dlon) * (n_lon+1) < 359.9:
        #print(np.median(dlon),n_lon,np.median(dlon)*n_lon+1)
        Dstr = 'DX'
    else:
        if np.min(np.abs(np.mod(lon,360.0)-180.0)) > (0.1*np.median(dlon)):
            #print(np.min(np.abs(np.mod(lon,360.0)-180.0)),'DE')
            Dstr = 'DE'
        else:
            Dstr = 'DC'
    return '{:d}x{:d}{:s}'.format(n_lat,n_lon,Pstr + Dstr)

def gen_l2c_regridder(cs_grid,ll_grid,method='conservative',grid_dir=None):
    import xesmf
    if grid_dir is None:
       grid_dir = regrid_archive
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
           fname = os.path.join(grid_dir,'{:s}_{:s}_c{:d}f{:d}.nc'.format(method,gen_ll_name(ll_grid),n_cs,i_face))
           try:
              regridder_list.append(xesmf.Regridder(ll_grid,sub_grid,method=method,reuse_weights=os.path.isfile(fname),filename=fname))
           except KeyError as e:
              print('Regridder object generation failed. If you received a permission error, please check that the output directory {} exists'.format(grid_dir))
              raise
    return regridder_list

def gen_c2l_regridder(cs_grid,ll_grid,method='conservative',grid_dir=None):
    import xesmf
    if grid_dir is None:
       grid_dir = regrid_archive
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
           fname = os.path.join(grid_dir,'{:s}_c{:d}f{:d}_{:s}.nc'.format(method,n_cs,i_face,gen_ll_name(ll_grid)))
           try:
              regridder_list.append(xesmf.Regridder(sub_grid,ll_grid,method=method,reuse_weights=os.path.isfile(fname),filename=fname))
           except KeyError as e:
              print('Regridder object generation failed. If you received a permission error, please check that the output directory {} exists'.format(grid_dir))
              raise
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
    return cstools.gen_grid(n_cs)

def gen_vrt_regridder(grid_in,grid_out,make_obj=True):
    xmat = gen_xmat(grid_in.p_edge(),grid_out.p_edge())
    if make_obj:
        return vrt_regridder(xmat)
    else:
        return xmat

def regrid_vertical(src_data_3D, xmat_regrid):
    # Performs vertical regridding using a sparse regridding matrix
    # Assumes that the FIRST dimension of the input data is vertical
    nlev_in = src_data_3D.shape[0]
    if xmat_regrid.shape[1] == nlev_in:
        # Current regridding matrix is for the reverse regrid
        # Rescale matrix to get the contributions right
        # Warning: this assumes that the same vertical range is covered
        warnings.warn('Using inverted regridding matrix. This may cause incorrect extrapolation')
        xmat_renorm = xmat_regrid.transpose().toarray()
        for ilev in range(xmat_renorm.shape[1]):
            norm_fac = np.sum(xmat_renorm[:,ilev])
            if np.abs(norm_fac) < 1.0e-20:
                norm_fac = 1.0
            xmat_renorm[:,ilev] /= norm_fac
        
        xmat_renorm = scipy.sparse.coo_matrix(xmat_renorm)
    elif xmat_regrid.shape[0] == nlev_in:
        # Matrix correctly dimensioned
        xmat_renorm = xmat_regrid.copy()
    else:
        raise ValueError('Regridding matrix not correctly sized')

    nlev_out = xmat_renorm.shape[1]
    out_shape = [nlev_out] + list(src_data_3D.shape[1:])
    n_other = np.product(src_data_3D.shape[1:])
    temp_data = np.zeros((nlev_out,n_other))
    #in_data = np.array(src_data_3D)
    in_data = np.reshape(np.array(src_data_3D),(nlev_in,n_other))
    for ix in range(n_other):
        in_data_vec = np.matrix(in_data[:,ix])
        temp_data[:,ix] = in_data_vec * xmat_renorm
    out_data = np.reshape(temp_data,out_shape)
    #for ix in range(in_data.shape[2]):
    #    for iy in range(in_data.shape[1]):
    #        in_data_vec = np.matrix(in_data[:,iy,ix])
    #        out_data[:,iy,ix] = in_data_vec * xmat_renorm
    return out_data

def gen_xmat(p_edge_from,p_edge_to):
    n_from = len(p_edge_from) - 1
    n_to   = len(p_edge_to) - 1
    
    # Guess - max number of entries?
    n_max = max(n_to,n_from)*5
    
    # Index being mapped from
    xmat_i = np.zeros(n_max)
    # Index being mapped to
    xmat_j = np.zeros(n_max)
    # Weights
    xmat_s = np.zeros(n_max)
    
    # Find the first output box which has any commonality with the input box
    first_from = 0
    i_to = 0
    if p_edge_from[0] > p_edge_to[0]:
        # "From" grid starts at lower altitude (higher pressure)
        while p_edge_to[0] < p_edge_from[first_from+1]:
            first_from += 1
    else:
        # "To" grid starts at lower altitude (higher pressure)
        while p_edge_to[i_to+1] > p_edge_from[0]:
            i_to += 1
    
    p_base_to = p_edge_to[i_to]
    p_top_to  = p_edge_to[i_to+1]
    frac_to_total = 0.0
    
    i_weight = 0
    for i_from in range(first_from,n_from):
        p_base_from = p_edge_from[i_from]
        p_top_from = p_edge_from[i_from+1]
        
        # Climb the "to" pressures until you intersect with this box
        while i_to < n_to and p_base_from <= p_edge_to[i_to+1]:
            i_to += 1
            frac_to_total = 0.0

        # Now, loop over output layers as long as there is any overlap,
        # i.e. as long as the base of the "to" layer is below the
        # top of the "from" layer
        last_box = False
        
        while p_edge_to[i_to] >= p_top_from and not last_box and not i_to >= n_to:
            p_base_common = min(p_base_from,p_edge_to[i_to])
            p_top_common = max(p_top_from,p_edge_to[i_to+1])
            # Fraction of source box
            frac_from = (p_base_common - p_top_common)/(p_base_from-p_top_from)
            # Fraction of target box
            frac_to   = (p_base_common - p_top_common)/(p_edge_to[i_to]-p_edge_to[i_to+1])
            #print(frac_to)
            
            xmat_i[i_weight] = i_from
            xmat_j[i_weight] = i_to
            xmat_s[i_weight] = frac_to
            
            i_weight += 1
            last_box = p_edge_to[i_to+1] <= p_top_from
            if not last_box:
                i_to += 1
            
    return scipy.sparse.coo_matrix((xmat_s[:i_weight],(xmat_i[:i_weight],xmat_j[:i_weight])),shape=(n_from,n_to))
