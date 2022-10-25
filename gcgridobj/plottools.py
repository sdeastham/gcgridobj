#!/usr/bin/env python3

# Prevent failure if importing gcgridobj without an X server
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from . import regrid, gc_vertical
import numpy as np
import warnings
import cartopy.io.shapereader as shpreader
from . import atmos_isa_mini

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

def plot_zonal(zonal_data,hrz_grid,vrt_grid,ax=None,show_colorbar=True,z_edge=None,vert_coord='altitude',
               sec_axis=False,sec_minor=False,sec_ticklabels=True,sec_axlabel=True,**kwargs):
    '''Plot 2D data as a zonal profile
    

    Keyword arguments:
    ax            -- axes to use for plotting (default None, results in new axes)
    show_colorbar -- show colorbar on right hand side of plot (default True)
    vert_coord    -- choice of vertical coordinate (default altitude, can be pressure or altitude)
    z_edge        -- altitude edges in km; alternative to vrt_grid (default None)
    sec_axis      -- show the other vertical coordinate as a secondary axis (default False)
    sec_minor     -- show minor tick labels on secondary axis; useful for small alt ranges (default False)
    '''

    # Check whether the user is providing old-style (p_mid returned through a function call) or
    # new-style (p_mid is a property) vertical grid definitions
    new_vrt_grid = isinstance(vrt_grid,gc_vertical.vert_grid_nd)
    lat_b = hrz_grid['lat_b']
    if vert_coord == 'pressure':
       if new_vrt_grid:
           alt_b = vrt_grid.p_edge
       else:
           alt_b = vrt_grid.p_edge().copy()
    else:
       # Use vertical grid description if available, otherwise
       # get explicit z_edge from the user
       if vrt_grid is None:
          assert z_edge is not None, 'Need altitude edges in km'
       elif z_edge is None:
          if new_vrt_grid:
             z_edge = vrt_grid.z_edge_ISA / 1000.0
          else:
             z_edge = vrt_grid.z_edge_ISA() / 1000.0
       alt_b = z_edge
    
    assert len(zonal_data.shape) == 2, 'Zonal data must be 2-D'
    assert len(alt_b) == zonal_data.shape[0]+1, 'Zonal data incorrectly shaped (altitude)'
    assert len(lat_b) == zonal_data.shape[1]+1, 'Zonal data incorrectly shaped (latitude)'

    if ax is None:
       f, ax = plt.subplots(1,1,figsize=(8,5))
    else:
       f = ax.figure

    im = ax.pcolormesh(lat_b,alt_b,zonal_data,**kwargs)

    if vert_coord == 'pressure':
       ax.invert_yaxis()
       ax.set_yscale('log')
       ax.set_ylabel('Pressure, hPa')
    elif vert_coord == 'altitude':
       ax.set_ylabel('Altitude, km')
    else:
       raise ValueError('Vertical coordinate {:s} not recognized'.format(vert_coord))

    cb_pad = 0.04
    if sec_axis:
       # Need colorbar to appear further out, if used
       cb_pad = 0.14
       # EXPERIMENTAL: Shows a secondary axis
       import matplotlib.ticker as mticker
       from decimal import Decimal
       def z_to_p(z):
          z_copy = np.where(z > 1000,1000,z).flatten()
          p = atmos_isa_mini.altitude_to_pressure(z_copy*1000)*0.01
          #print('zp',z_copy,p)
          return p.reshape(z_copy.shape)
       def p_to_z(p):
          p_copy = np.where(p<0.01,0.01,p).flatten()
          z = atmos_isa_mini.pressure_to_altitude(p_copy*100) * 1.0e-3
          #print('pz',p_copy,z)
          return z.reshape(p.shape)
       if vert_coord == 'pressure':
          sa_fwd = p_to_z
          sa_inv = z_to_p
          sec_name = 'Altitude, km'
       elif vert_coord == 'altitude':
          sa_fwd = z_to_p
          sa_inv = p_to_z
          sec_name = 'Pressure, hPa'

       # === IF SECONDARY_YAXIS WORKS ===
       # This SHOULD accomplish the goal - but doesn't seem to work yet
       # May work with matplotlib 3.2+, but for now use custom function
#       ax2=ax.secondary_yaxis('right',functions=(sa_fwd,sa_inv))
       # === ELSE ===
       # Custom approach using linked axes
       # The secondary axis is actually identical to the primary one, but
       # we mark tick points on it based on our transformed vertical coord.
       # The axes are also linked so that a change in the y-coord of the
       # primary axis will modify the secondary one. However, changing the
       # y-scale of the primary axis will not cause an appropriate change 
       # in the secondary axis. Equally, although the current ticks will
       # always turn up in the right place when y-limits are changed, new
       # ticks will not be produced if (for example) a very small altitude
       # range is desired. 
       def update_ax2(ax1):
          y1,y2 = ax1.get_ylim()
          #ax2.set_ylim(sa_fwd(y1),sa_fwd(y2))
          ax2.set_ylim(y1,y2)
          ax2.figure.canvas.draw()
      
       ax2 = ax.twinx()

       # log-p and z do not exactly line up - need to identify "by hand"
       # locations of each tick mark on the new axis
       if vert_coord == 'altitude':
          alt_ticks = np.logspace(-2,3,6)
          alt_minor = []
          for alt in alt_ticks[:-1]:
             for sub_alt in np.linspace(2,10,9)[:-1]:
                alt_minor.append(alt*sub_alt)
          alt_minor.append(alt_ticks[-1])
          # Generate the labels
          def tick_label_gen(ticks):
             tick_labels = []
             for tick in ticks:
                if tick >= 1.0:
                   tick_label = '{:.0f}'.format(tick)
                else:
                   d_tick = Decimal(tick)
                   (sign, digits, exponent) = d_tick.as_tuple()
                   tick_exp = len(digits) + exponent - 1
                   #tick_mag = d_tick.scaleb(-tick_exp).normalize()
                   #tick_label = '{:.0f}$\\times$10$^{:.0f}$'.format(tick_mag,tick_exp)
                   tick_label = '{:.1g}'.format(tick)
                tick_labels.append(tick_label)
             return tick_labels
       else:
          alt_ticks = np.linspace(0,150,16)
          if sec_minor:
             alt_minor = [x for x in np.linspace(0,150,151) if x not in alt_ticks]
          else:
             alt_minor = []
          ax2.invert_yaxis()
          ax2.set_yscale('log')
          # Generate the labels
          tick_label_gen = lambda ticks : ['{:.0f}'.format(x) for x in ticks]

       # Now, figure out where on the vertical axes the ticks should actually be
       alt_ticks_adj = sa_inv(np.array(alt_ticks))
       alt_minor_adj = sa_inv(np.array(alt_minor))

       # Hard-code the y-ticks and tick labels
       ax2.set_yticks(alt_ticks_adj)
       ax2.set_yticklabels(tick_label_gen(alt_ticks))
       ax2.set_yticks(alt_minor_adj,minor=True)
       # Force minor ticks to also be shown (dangerous!)
       if sec_minor and sec_ticklabels:
          ax2.set_yticklabels(tick_label_gen(alt_minor),minor=True)
           
       # Initialize the limits
       update_ax2(ax)

       # Automatically update ylim of ax2 when ylim of ax1 changes
       ax.callbacks.connect("ylim_changed", update_ax2)
       # === END IF ===

       if sec_axlabel:
          ax2.set_ylabel(sec_name)
       else:
          ax2.set_ylabel('')

       if not sec_ticklabels:
          ax2.set_yticklabels([])

    if show_colorbar:
       cb = f.colorbar(im, ax=ax, shrink=0.6, orientation='vertical', pad=cb_pad)
    else:
       cb = None

    return im, cb

def plot_layer(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True,coastlines=True,**kwargs):

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
       im_obj = plot_cs(layer_data,hrz_grid=hrz_grid,ax=ax,crs_data=crs_data,crs_plot=crs_plot,**kwargs)
    elif ld_shape[0] == 6*ld_shape[1]:
       # Assume cubed sphere
       is_cs = True
       im_obj = plot_cs(layer_data,hrz_grid=hrz_grid,ax=ax,crs_data=crs_data,crs_plot=crs_plot,**kwargs)
    else:
       # Assume lat-lon
       im_obj = plot_latlon(layer_data,hrz_grid=hrz_grid,ax=ax,crs_data=crs_data,crs_plot=crs_plot,**kwargs)

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

def plot_latlon(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True,**kwargs):
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

    im = ax.pcolormesh(lon_b,lat_b,layer_data,transform=crs_data,**kwargs)

    return im

def update_cs(layer_data,im_vec,hrz_grid=None,cs_threshold=None):
    # WARNING: layer_data must be [6 x N x N]
    if cs_threshold is not None:
        if hrz_grid is None:
            # Try to figure out the grid from the layer data
            hrz_grid = regrid.guess_cs_grid(layer_data.shape)    
        masked_data = np.ma.masked_where(np.abs(hrz_grid['lon'] - 180.0) < cs_threshold, layer_data)
    else:
        masked_data = layer_data
    for i_face in range(6):
        im_vec[i_face].set_array(masked_data[i_face,:,:].ravel())

def plot_cs(layer_data,hrz_grid=None,ax=None,crs_data=None,crs_plot=None,show_colorbar=True,cs_threshold=None,**kwargs):

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
       hrz_grid = regrid.guess_cs_grid(layer_data.shape)
  
    # A pending PR for Cartopy is expected to fix an issue where cells crossing the antimeridian
    # are incorrectly plotted. Until then (and for users with older versions of cartopy), setting
    # cs_threshold to a non-zero value will at least mask out these cells. A value of 5 seems to
    # work (somewhat) well, at least for C24 data.
    if cs_threshold is not None:
        masked_data = np.ma.masked_where(np.abs(hrz_grid['lon'] - 180.0) < cs_threshold, layer_data)
    else:
        masked_data = layer_data
 
    im_vec = []
    for i_face in range(6):
       lon_b = np.mod(hrz_grid['lon_b'][i_face,:,:],360.0)
       im = ax.pcolormesh(lon_b,hrz_grid['lat_b'][i_face,:,:],masked_data[i_face,:,:],transform=crs_data,**kwargs)
       im_vec.append(im)

    c_lim = [np.min(layer_data),np.max(layer_data)]
    if (c_lim[0] == c_lim[1]):
       c_lim = [c_lim[0] - 0.5,c_lim[1] + 0.5]

    set_clim(im_vec,c_lim)

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
       if c_lim is None:
          c_lim = im_obj.get_clim()
       elif np.isscalar(c_lim):
          # Assume max
          c_lim = [-c_lim,c_lim]
       im_obj.set_clim(c_lim)
       # Since Cartopy 0.19.0, there is a hidden set of polygons for GeoQuadMesh objects
       # if they cross the antimeridian. Their color limits need to be fixed separately
       wrapped_obj = getattr(im_obj, "_wrapped_collection_fix", None)
       if wrapped_obj is not None:
          wrapped_obj.set_clim(c_lim)
          if cmap is not None:
             wrapped_obj.set_cmap(cmap)
       if cmap is not None:
          im_obj.set_cmap(cmap)
    return None
