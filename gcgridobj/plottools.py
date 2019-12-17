#!/usr/bin/env python3
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from gcgridobj import regrid
import numpy as np
import warnings
import cartopy.io.shapereader as shpreader

__all__ = ['plot_zonal','plot_layer',
           'plot_cs','plot_latlon',
           'plot_state','plot_shape',
           'plot_country']

crs_plot_standard = ccrs.PlateCarree()
crs_data_standard = ccrs.PlateCarree() 

def reshape_cs(cs_data):
    warnings.warn('plottools.reshape_cs is deprecated. Please use regrid.reshape_cs instead',FutureWarning)
    return regrid.reshape_cs(cs_data)

def regrid_ll_to_cs(ll_data,cs_grid=None,ll_grid=None,regridder_list=None):
    warnings.warn('plottools.regrid_ll_to_cs is deprecated. Please use regrid.l2c instead',FutureWarning)
    return regrid.l2c(ll_data,cs_grid=cs_grid,ll_grid=ll_grid,regridder_list=regridder_list)

def regrid_cs(cs_data,new_grid=None,cs_grid=None,regridder_list=None):
    warnings.warn('plottools.regrid_cs is deprecated. Please use regrid.c2l instead',FutureWarning)
    return regrid.c2l(cs_data,cs_grid=cs_grid,ll_grid=new_grid,regridder_list=regridder_list)

def gen_l2c_regridder(cs_grid,ll_grid,method='conservative',grid_dir='.'):
    warnings.warn('plottools.gen_l2c_regridder is deprecated. Please use regrid.gen_regridder instead',FutureWarning)
    return regrid.gen_regridder(ll_grid,cs_grid,method,grid_dir)

def gen_cs_regridder(cs_grid,ll_grid,method='conservative',grid_dir='.'):
    warnings.warn('plottools.gen_cs_regridder is deprecated. Please use regrid.gen_regridder instead',FutureWarning)
    return regrid.gen_regridder(cs_grid,ll_grid,method,grid_dir)
 
def guess_cs_grid(cs_data_shape):
    # This used to return face side length and is_gmao, which was inconsistent
    warnings.warn('plottools.guess_cs_grid is deprecated. Please use regrid.guess_cs_grid or regrid.guess_n_cs instead',FutureWarning)
    return regrid.guess_n_cs(cs_data_shape)

def plot_zonal(zonal_data,hrz_grid,vrt_grid,ax=None,is_pressure=None,show_colorbar=True,z_edge=None,vert_coord=None):
    '''Plot 2D data as a zonal profile
    '''

    if is_pressure is not None:
       warnings.warn("is_pressure option is deprecated. Use vert_coord instead", FutureWarning)
       # Assert compatible options
       if vert_coord is None:
          if is_pressure:
             vert_coord = 'pressure'
          else:
             vert_coord = 'altitude'
       else:
          raise ValueError("Cannot specify vert_coord and deprecated is_pressure option together")

    # Default option. This will be moved to the argument list once is_pressure is removed
    if vert_coord is None:
       vert_coord = 'altitude'

    lat_b = hrz_grid['lat_b']
    if vert_coord == 'pressure':
       alt_b = vrt_grid.p_edge()
    else:
       # Use vertical grid description if available, otherwise
       # get explicit z_edge from the user
       if vrt_grid is None:
          assert z_edge is not None, 'Need altitude edges in km'
       elif z_edge is None:
          z_edge = vrt_grid.z_edge_ISA() / 1000.0
       alt_b = z_edge
    
    assert len(zonal_data.shape) == 2, 'Zonal data must be 2-D'
    assert len(alt_b) == zonal_data.shape[0]+1, 'Zonal data incorrectly shaped (altitude)'
    assert len(lat_b) == zonal_data.shape[1]+1, 'Zonal data incorrectly shaped (latitude)'

    if ax is None:
       f, ax = plt.subplots(1,1,figsize=(8,5))
    else:
       f = ax.figure

    im = ax.pcolormesh(lat_b,alt_b,zonal_data)

    if vert_coord == 'pressure':
       ax.invert_yaxis()
       ax.set_yscale('log')
       ax.set_ylabel('Pressure, hPa')
    elif vert_coord == 'altitude':
       ax.set_ylabel('Altitude, km')
    else:
       raise ValueError('Vertical coordinate {:s} not recognized'.format(vert_coord))

    if show_colorbar:
       cb = f.colorbar(im, ax=ax, shrink=0.6, orientation='vertical', pad=0.04)
    else:
       cb = None

    return im, cb

def plot_layer(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True,coastlines=True):

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

    if coastlines:
        # If user wants a different resolution, they can disable set coastlines=False
        # and run this command after calling plot_layer
        ax.coastlines('50m')

    return im_obj, cb

def plot_latlon(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True):
    '''Plot 2D lat-lon data
    '''

    assert len(layer_data.shape) == 2, 'Layer data must be 2-D'

    if hrz_grid is None:
       #hrz_grid = gc_horizontal.get_grid(layer_data.shape)
       hrz_grid = regrid.guess_ll_grid(layer_data.shape)
       assert hrz_grid is not None, 'Could not auto-identify grid'
   
    lon_b = hrz_grid['lon_b']
    lat_b = hrz_grid['lat_b']

    assert len(lon_b) == layer_data.shape[1]+1, 'Layer data incorrectly shaped (longitude)'
    assert len(lat_b) == layer_data.shape[0]+1, 'Layer data incorrectly shaped (latitude)'

    im = ax.pcolormesh(lon_b,lat_b,layer_data,transform=crs_data)

    return im

def update_cs(layer_data,im_vec,hrz_grid=None,cs_threshold=5):
    # WARNING: layer_data must be [6 x N x N]
    if hrz_grid is None:
       # Try to figure out the grid from the layer data
       #n_cs, is_gmao = guess_cs_grid(layer_data.shape)    
       #hrz_grid = cubedsphere.csgrid_GMAO(n_cs)
       cs_grid = regrid.guess_cs_grid(layer_data.shape)    
    masked_data = np.ma.masked_where(np.abs(hrz_grid['lon'] - 180.0) < cs_threshold, layer_data)
    for i_face in range(6):
       im_vec[i_face].set_array(masked_data[i_face,:,:].ravel())

def plot_cs(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True,cs_threshold=5.0):

    # 2019-12-17: dropped support for non-GMAO grids
    #n_cs, is_gmao = regrid.guess_n_cs(layer_data.shape)    

    #if is_gmao:
    #   # Use data as-is
    #   layer_data_3D = layer_data
    #else:
    #   # Reshape data to be "GMAO-style" (3D)
    #   layer_data_3D = np.reshape(layer_data,(6,n_cs,n_cs))

    if hrz_grid is None:
       # Try to figure out the grid from the layer data
       #hrz_grid = cubedsphere.csgrid_GMAO(n_cs)
       hrz_grid = regrid.guess_cs_grid(layer_data.shape)
    
    masked_data = np.ma.masked_where(np.abs(hrz_grid['lon'] - 180.0) < cs_threshold, layer_data)
 
    im_vec = []
    for i_face in range(6):
       im = ax.pcolormesh(hrz_grid['lon_b'][i_face,:,:],hrz_grid['lat_b'][i_face,:,:],masked_data[i_face,:,:],transform=crs_data)
       im_vec.append(im)

    c_lim = [np.min(layer_data),np.max(layer_data)]
    if (c_lim[0] == c_lim[1]):
       c_lim = [c_lim[0] - 0.5,c_lim[1] + 0.5]

    for im in im_vec:
       im.set_clim(c_lim)

    return im_vec

def plot_shape(state_name,state_val,shape_data_archive,classifier,
               edgecolor='black',cmap=None,c_lim=(0.0,1.0),ax=None,nofail=True):
    '''Plot an shape onto a set of axes'''
    
    if ax is None:
        f, ax = plt.subplots(1,1,figsize=(10,8),subplot_kw={'projection': ccrs.PlateCarree()})
    
    if state_val is None:
        facecolor = 'none'
    else:
        if cmap is None:
            temp_cm = plt.get_cmap('viridis',50)
        elif isinstance(cmap, str):
            temp_cm = plt.get_cmap(cmap)
        else:
            temp_cm = cmap
        
        cmap_val = (state_val - c_lim[0]) / (c_lim[1] - c_lim[0])
        facecolor = temp_cm(cmap_val)
    
    im_shp = None
    
    for astate in shpreader.Reader(shape_data_archive).records():
        if state_name == astate.attributes[classifier]:
            im_shp = ax.add_geometries([astate.geometry], ccrs.PlateCarree(),
                                      facecolor=facecolor,edgecolor=edgecolor)
            break
    
    if im_shp is None:
        state_msg = 'Shape ''{:s}'' not found'.format(state_name)
        if nofail:
            warnings.warn(state_msg)
        else:
            raise ValueError(state_msg)
    return im_shp, ax

def plot_state(state_name,state_val,resolution='110m',**kwargs):
    '''Plot a US state, Australian territory, or Brazilian state onto a set of axes'''
    shape_data_archive = shpreader.natural_earth(resolution=resolution,
                                         category='cultural', name='admin_1_states_provinces_lakes_shp')
    classifier = 'name'
    im_shp, ax = plot_shape(state_name,state_val,classifier='name',shape_data_archive=shape_data_archive,**kwargs)
    return im_shp, ax

def plot_country(country_name,country_val,resolution='110m',**kwargs):
    '''Plot a country onto a set of axes'''
    shape_data_archive = shpreader.natural_earth(resolution=resolution, category='cultural', name='admin_0_countries')
    im_shp, ax = plot_shape(country_name,country_val,classifier='NAME',
                            shape_data_archive=shape_data_archive,**kwargs)
    return im_shp, ax

def get_clim(im_obj):
    if isinstance(im_obj,list):
       c_lim = [+np.inf,-np.inf]
       for im in im_obj:
          c_lim_temp = get_clim(im)
          #print(c_lim,c_lim_temp,'ccc')
          c_lim[0] = min(c_lim_temp[0],c_lim[0])
          c_lim[1] = max(c_lim_temp[1],c_lim[1])
    else:
       c_lim = im_obj.get_clim()
    return c_lim

def set_clim(im_obj,c_lim=None,cmap=None):
    if isinstance(im_obj,list):
       for im in im_obj:
          set_clim(im,c_lim,cmap)
    else:
       if np.isscalar(c_lim):
          # Assume max
          c_lim = [-c_lim,c_lim]
       im_obj.set_clim(c_lim)
       if cmap is not None:
          im_obj.set_cmap(cmap)
    return None
