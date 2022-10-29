import xarray as xr
from . import gc_vertical, cstools, latlontools, plottools, regrid
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

def same_hrz_grid(grid_A,grid_B):
    # Determine if two grids are identical
    if grid_A is None or grid_B is None:
        return False
    lon_A = grid_A['lon'].values
    lon_B = grid_A['lon'].values
    shape_A = lon_A.shape
    shape_B = lon_B.shape
    if (len(shape_A) != len(shape_B)) or shape_A != shape_B:
        return False
    if not np.all(lon_A == lon_B):
        return False
    lat_A = grid_B['lat'].values
    lat_B = grid_A['lat'].values
    return np.all(lat_A == lat_B)

@xr.register_dataset_accessor("gcgo")
class GCGOAccessor:
    __slots__ = ['_vrt_grid','_obj','_zonal_grid','_zonal_regridder','is_cs','positive']
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.vrt_grid = None
        self._zonal_grid = None
        self._zonal_regridder = None
        self.positive = 'up' # Assume for now
        # Figure out the horizontal grid
        # Add its properties to this dataset so
        # that it can be used natively as the
        # horizontal grid descriptor
        # First: is this a cubed-sphere grid?
        self.is_cs = 'Xdim' in self._obj.dims
        if self.is_cs:
            temp_hg = cstools.extract_grid(self._obj)
            for var in ['area','lat','lon']:
                self._obj[var] = (('nf','Ydim','Xdim'),temp_hg[var].values)
            for var in ['lat_b','lon_b']:
                self._obj[var] = (('nf','YCdim','XCdim'),temp_hg[var].values)
        else:
            temp_hg = latlontools.extract_grid(self._obj)
            self._obj['area'] = (('lat','lon'),temp_hg['area'].values)
            for var in ['lat','lon','lat_b','lon_b']:
                self._obj[var]  = ((var),temp_hg[var].values)

    # Getter
    @property
    def zonal_grid(self):
        if self._zonal_grid is None:
            # If not yet set, then (obviously) set it! Use the setter so that a regridder is generated
            # Use a default - either a 1x1.25 (if CS) or just "this grid" (if lat-lon)
            if self.is_cs:
                self.zonal_grid = latlontools.gen_grid(lat_stride=1.0,lon_stride=1.25,half_polar=True,center_180=True)
            else:
                self.zonal_grid = self._obj
        return self._zonal_grid
    
    # Setter
    @zonal_grid.setter
    def zonal_grid(self,ll_grid):
        # Update the regridder whenever the lat-lon grid is updated
        # Don't do anything if the grid is already defined and matches
        # the proposed grid
        if same_hrz_grid(self._zonal_grid,ll_grid):
            return
        self._zonal_regridder = regrid.gen_regridder(self._obj,ll_grid)
        self._zonal_grid = ll_grid
    
    @property
    def zonal_regridder(self):
        # Read the zonal grid, to force the regridder to be generated
        self.zonal_grid
        return self._zonal_regridder
    
    def import_zonal_grid(self,ll_grid,regridder):
        # Allow the user to update the regridder with an externally-generated one
        # Use this to avoid repeatedly reading in regridders
        self._zonal_regridder = regridder
        self._zonal_grid = ll_grid
        
    @property
    def vrt_grid(self):
        if self._vrt_grid is None:
            # Make a couple of quick assumptions..
            lev_var = None
            for dim in self._obj.dims:
                if dim in ['lev','levs']:
                    lev_var = dim
                    break
            if lev_var is None:
                raise ValueError('Vertical grid not supplied')
            n_lev = len(self._obj[lev_var])
            if n_lev == 47:
                print('Assuming a GEOS-5 47-level reduced grid')
                vg = gc_vertical.standard_grid('GEOS_47L')
            elif n_lev == 72:
                print('Assuming a GEOS-5 72-level reduced grid')
                vg = gc_vertical.standard_grid('GEOS_72L')
            else:
                raise ValueError('Unknown vertical grid')
            self.vrt_grid = vg
        return self._vrt_grid
    
    @vrt_grid.setter
    def vrt_grid(self,vg):
        self._vrt_grid = vg
    
    def plot_layer(self,var,t=None,z=None,ax=None,**kwargs):
        if t is None:
            data = self._obj[var].mean(dim='time')
        else:
            data = self._obj[var].sel(time=t)
        if 'lev' in data.dims:
            if z is None:
                data_layer = data.mean(dim='lev')
            elif z == 'sum':
                data_layer = data.sum(dim='lev')
            else:
                data_layer = data.sel(lev=z)
        else:
            data_layer = data
        if ax is None:
            f, ax = plt.subplots(1,1,figsize=(8,5),subplot_kw={'projection': ccrs.PlateCarree()})
        im, cb = plottools.plot_layer(data_layer,hrz_grid=self._obj,ax=ax,**kwargs)
        return im, cb
    
    def update_ll(self,new_grid):
        if self.ll_data is None:
            define_new = True
        else:
            define_new = not same_hrz_grid(self._ll_grid,new_grid)
    
    def plot_zonal(self,var,t=None,ax=None,**kwargs):
        # Zonal plot will use the lat-lon grid currently held as 
        if t is None:
            data = self._obj[var].mean(dim='time')
        else:
            data = self._obj[var].sel(time=t)
        data_ll = np.squeeze(np.mean(self.zonal_regridder(data),axis=-1))
        if self.positive == 'down':
            data_ll = np.flip(data_ll,axis=0)
        if ax is None:
            f, ax = plt.subplots(1,1,figsize=(8,5))
        im, cb = plottools.plot_zonal(data_ll,hrz_grid=self.zonal_grid,vrt_grid=self.vrt_grid,ax=ax,**kwargs)
        return im, cb
