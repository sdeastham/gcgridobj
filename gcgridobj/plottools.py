#!/usr/bin/env python3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cubedsphere
from gcgridobj import gc_horizontal, gc_vertical
import numpy as np
import xesmf
import xarray

__all__ = ['regrid_cs','plot_zonal','plot_layer',
           'plot_cs','plot_latlon','guess_cs_grid',
           'reshape_cs']
 

crs_plot_standard = ccrs.PlateCarree()
crs_data_standard = ccrs.PlateCarree() 

def reshape_cs(cs_data):
    if cs_data.shape[-2] == 6*cs_data.shape[-1]:
       full_data = cs_data.copy()
       # Data is non-GMAO
       n_cs = full_data.shape[-1]
       new_shape = [6,n_cs,n_cs]
       if len(full_data.shape) == 2:
          # Data is 2-D
          full_data = np.reshape(full_data,new_shape)
       else:
          # Ugh
          n_layers = full_data.shape[0]
          old_data = full_data
          full_data = np.zeros((n_layers,6,n_cs,n_cs))
          for i_layer in range(n_layers):
             full_data[i_layer,:,:,:] = np.reshape(old_data[i_layer,:,:],new_shape)
       return full_data
    else:
       return cs_data

def regrid_cs(cs_data,new_grid,cs_grid=None,regridder_list=None):
    '''
    # regrid cs to ll
    ds_dev_cmp = np.zeros([nlev, cmpgrid['lat'].size, cmpgrid['lon'].size])
    for i in range(6):
        regridder = devregridder_list[i]
        ds_dev_cmp += regridder(ds_dev_reshaped[i])
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

    # Figure out the grid size
    n_cs, is_gmao = guess_cs_grid(layer_shape)

    if cs_grid is None:
       # Try to figure out the grid from the layer data
       cs_grid = cubedsphere.csgrid_GMAO(n_cs)
  
    n_lon = len(new_grid['lon'])
    n_lat = len(new_grid['lat'])
    if regridder_list is None:
       regridder_list = gen_cs_regridder(cs_grid,new_grid)

    ll_data = np.zeros((n_lev,n_lat,n_lon))
    for i_lev in range(n_lev):
       for i_face in range(6):
          ll_data[i_lev,:,:] += regridder_list[i_face](full_data[i_lev,i_face,:,:])

    if single_layer:
       ll_data = np.squeeze(ll_data) 

    return ll_data 

def gen_cs_regridder(cs_grid,ll_grid,method='conservative',grid_dir='.'):
    regridder_list=[]
    n_lon = ll_grid['lon'].size
    n_lat = ll_grid['lat'].size
    n_cs = cs_grid['lat'][0].shape[0]
    for i_face in range(6):
       sub_grid = {'lat':   cs_grid['lat'][i_face], 
                   'lon':   cs_grid['lon'][i_face],
                   'lat_b': cs_grid['lat_b'][i_face], 
                   'lon_b': cs_grid['lon_b'][i_face]}
       regridder_list.append(xesmf.Regridder(sub_grid,ll_grid,method='conservative',reuse_weights=True,filename='conservative_c{:d}f{:d}_{:d}x{:d}'.format(n_cs,i_face,n_lat,n_lon)))
    return regridder_list
 
def guess_cs_grid(cs_data_shape):
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

    return n_cs, is_gmao    

def plot_zonal(zonal_data,hrz_grid,vrt_grid,ax=None,is_pressure=True,show_colorbar=True):
    '''Plot 2D data as a zonal profile
    '''

    lat_b = hrz_grid['lat_b']
    if is_pressure:
       alt_b = vrt_grid.p_edge()
    else:
       raise ValueError('Altitude plotting not yet enabled')
       alt_b = 0.0

    assert len(zonal_data.shape) == 2, 'Zonal data must be 2-D'
    assert len(alt_b) == zonal_data.shape[0]+1, 'Zonal data incorrectly shaped (altitude)'
    assert len(lat_b) == zonal_data.shape[1]+1, 'Zonal data incorrectly shaped (latitude)'

    if ax is None:
       f, ax = plt.subplots(1,1,figsize=(8,5))
    else:
       f = ax.figure

    im = ax.pcolormesh(lat_b,alt_b,zonal_data)

    if is_pressure:
       ax.invert_yaxis()
       ax.set_yscale('log')

    if show_colorbar:
       cb = f.colorbar(im, ax=ax, shrink=0.6, orientation='vertical', pad=0.04)
    else:
       cb = None

    return im, cb

def plot_layer(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True):

    if crs_data is None:
       crs_data = crs_data_standard

    if crs_plot is None:
       crs_plot = crs_plot_standard

    if ax is None:
       f, ax = plt.subplots(1,1,figsize=(8,5),subplot_kw={'projection':crs_plot})
    else:
       f = ax.figure

    # Test the data; if it looks cubed-sphere, throw it to the CS routines. Otherwise assume lat-lon
    ld_shape = layer_data.shape
    if len(ld_shape) < 2 or len(ld_shape) > 3:
       raise ValueError('Layer data shape invalid (bad dimension count)')
    elif len(ld_shape) == 3:
       # Could this be a valid CS dataset?
       assert ld_shape[0] == 6 and (ld_shape[1] == ld_shape[2]), 'Layer data shape invalid (3D and not CS)'
       im_obj = plot_cs(layer_data,hrz_grid=hrz_grid,ax=ax,crs_data=crs_data,crs_plot=crs_plot)
    elif ld_shape[0] == 6*ld_shape[1]:
       # Assume cubed sphere
       is_cs = True
       im_obj = plot_cs(layer_data,hrz_grid=hrz_grid,ax=ax,crs_data=crs_data,crs_plot=crs_plot)
    else:
       # Assume lat-lon
       im_obj = plot_latlon(layer_data,hrz_grid=hrz_grid,ax=ax,crs_data=crs_data,crs_plot=crs_plot)

    # If cubed-sphere, use the first image
    is_cs = isinstance(im_obj, list)
    if is_cs:
       im = im_obj[0]
    else:
       im = im_obj

    if show_colorbar:
       cb = f.colorbar(im, ax=ax, shrink=0.6, orientation='vertical', pad=0.04)
    else:
       cb = None

    ax.coastlines()

    return im_obj, cb

def plot_latlon(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True):
    '''Plot 2D lat-lon data
    '''

    assert len(layer_data.shape) == 2, 'Layer data must be 2-D'

    if hrz_grid is None:
       hrz_grid = gc_horizontal.get_grid(layer_data.shape)
       assert hrz_grid is not None, 'Could not auto-identify grid'
   
    lon_b = hrz_grid['lon_b']
    lat_b = hrz_grid['lat_b']

    assert len(lon_b) == layer_data.shape[1]+1, 'Layer data incorrectly shaped (longitude)'
    assert len(lat_b) == layer_data.shape[0]+1, 'Layer data incorrectly shaped (latitude)'

    im = ax.pcolormesh(lon_b,lat_b,layer_data,transform=crs_data)

    return im

def plot_cs(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True):

    n_cs, is_gmao = guess_cs_grid(layer_data.shape)    

    if is_gmao:
       # Use data as-is
       layer_data_3D = layer_data
    else:
       # Reshape data to be "GMAO-style" (3D)
       layer_data_3D = np.reshape(layer_data,(6,n_cs,n_cs))

    if hrz_grid is None:
       # Try to figure out the grid from the layer data
       hrz_grid = cubedsphere.csgrid_GMAO(n_cs)
    
    cs_threshold = 5.0
    masked_data = np.ma.masked_where(np.abs(hrz_grid['lon'] - 180.0) < cs_threshold, layer_data_3D)
 
    im_vec = []
    for i_face in range(6):
       im = ax.pcolormesh(hrz_grid['lon_b'][i_face,:,:],hrz_grid['lat_b'][i_face,:,:],masked_data[i_face,:,:],transform=crs_data)
       im_vec.append(im)

    c_lim = [np.min(layer_data),np.max(layer_data)]
    if (c_lim[0] == c_lim[1]):
       c_lim = [-1.0,1.0]

    for im in im_vec:
       im.set_clim(c_lim)

    return im_vec
