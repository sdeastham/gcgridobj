gcgridobj

This is a minimal python package which provides objects and routines useful for manipulating GEOS-Chem output data in Python 3.

## Installation

### Requirements

* https://github.com/jiaweizhuang/xesmf

### Installation steps

1. Make sure that you have the above requirements
2. Clone this repo to a safe location: `git clone https://github.com/sdeastham/gcgridobj.git /home/your_username/Python/gcgridobj`
3. Navigate to your clone of the repo: `cd /home/your_username/Python/gcgridobj`
4. Use `pip` to install it, using the `-e` option if you wish to be able to modify the install later: `pip install -e .`

If you perform these steps from within your `conda` environment, you will find that `gcgridobj` is now available whenever you use that environment. Note that you can always remove gcgridobj by calling `pip uninstall gcgridobj` while within the environment.

## Examples

### Generating a horizontal grid description

Horizontal grid descriptions are just `xarray` datasets which include either:
* 1-D `lon` and `lat` variables
* 2-D `xDim` and `yDim` variables

To generate a grid description, there are three options:
1. Use a pre-calculated grid description from `gcgridobj.gc_horizontal`. For example, GEOS-Chem's 4x5 grid is taken from GMAO, and can be retrieved as `grid = gcgridobj.gc_horizontal.gmao_4x5_global`
2. Generate a rectilinear latitude-longitude grid using `gcgridobj.latlontools`. For example, a simple 1x1 grid can be generated as `grid = gcgridobj.latlontools.gen_grid(lon_stride=1,lat_stride=1,half_polar=False,center_180=False)`. This function can also be used to generate nested grids.
3. Generate a gnomonic cubed-sphere grid using `gcridobj.cstools`. For exampe, a c24 grid can be generated as `grid = gcgridobj.cstools.gen_grid(24)`.

The horizontal grid descriptions are used to handle all plotting and regridding in `gcgridobj`. They also carry a copy of the grid cell area - for a given grid description `grid`, take a look at `grid['area'].values`.

### Generating a vertical grid description

Vertical grid descriptions are based on a rather hokey custom class in `gcgridobj.gc_vertical`. There are several pre-generated examples in there, such as the GMAO 72-layer grid (`gcgridobj.gc_vertical.GEOS_72L_grid`), as well as the GEOS-Chem 47-layer "lumped" counterpart (`GEOS_47L_grid`), and the CAM 26-layer grid (`CAM_26L_grid`). To generate a new one, all you need are the `AP` and `BP` hybrid grid values. `AP` should be in units of hPa, and `BP` should be unitless. To create a new vertical grid, call `vrt_grid = gcgridobj.gc_vertical.vert_gird(AP=new_AP,BP=new_BP)`.

The vertical grid description can be used to retrieve the pressure edges (`vrt_grid.p_edge`), and midpoints (`vrt_grid.p_mid`). These are calculated assuming a specific surface pressure - to change it, call `vrt_grid.p_sfc = new_sfc_p` (again using hPa as your units). The class also includes methods to estimate the grid mid-point and edge altitudes using the International Standard Atmosphere. Altitude edges, for example, can be found with `vrt_grid.z_edge_ISA`. The returned array is defined in meters, and again assumes that the surface pressure is equal to `vrt_grid.p_sfc`. Finally, you can generate a 3-D pressure edge field by calling `vrt_grid.gen_p_field(p_sfc_2D)`.

### Horizontal regridding

Regridding is achieved by generating a reusable `regridder` object. The object is always one-directional, but regridders can easily be specified for any combination of lat-lon and cubed-sphere grid. To do so, first specify your source and destination grids. The below example manually generates the GMAO 4x5 grid, and a target c90 cubed-sphere grid:

```
src_grid = gcgridobj.latlontools.gen_grid(lon_stride=5,lat_stride=4,half_polar=True,center_180=True)
dst_grid = gcgridobj.cstools.gen_grid(90)
```

Now, generate a regridder. This uses xESMF to do the heavy lifting. The first time you generate a regridder for a specific combination of grids, the weights will be calculated online (this can take some time!) and the calculated weights will be stored in your working directory. All future calls to create that regridder will then result in those weights being loaded, which is much faster (and much less memory-intensive):

`regridder = gcgridobj.regrid.gen_regridder(src_grid,dst_grid)`

Finally, if you have some 4x5 data (oriented a [latitude x longitude]) called, say, `data_4x5`, you can regrid it to the c90 grid by calling

`data_c90 = regridder(data_4x5)`

The regridder should automatically handle any additional leading dimensions (e.g. a [time x level x latitude x longitude] 4x5 set of data would here become a [time x level x face x yDim x xDim] c90 set of data). *IMPORTANT*: all data are assumed to be normalized per unit area (e.g. population per km2, parts per billion, kg per m2). If your units are NOT in this form (e.g. total mass), make sure to divide by the source grid area before regridding, and then multiply the result by the destination grid area.

### Vertical regridding

Once you have two vertical grid descriptions, say `src_vrt` and `dst_vrt`, you can generate a regridder in much the same way as was defined for the horizontal regridding above. Simply take your two grids:

```
src_vrt = gcgridobj.gc_vertical.standard_grid('GEOS_72L')
dst_vrt = gcgridobj.gc_vertical.standard_grid('CAM_26L')
```

and generate a vertical regridder object:

`vrt_regridder = gcgridobj.regrid.vrt_regridder(src_vrt,dst_vrt)`

Now, to regrid vertically, take any data set on your source vertical grid - say `data_src` - and call `data_dst = regridder(data_src)`. As before, the data are expected to be in units which are normalized per unit pressure (e.g. parts per billion by volume).

This vertical regrid operation does not yet account for changes in surface pressure.

### Plotting directly with gcgridobj

If you have a horizontal and vertical grid defined, you can use gcgridobj to plot both cubed sphere and lat-lon data. Assume that we have a C24 grid of data called `conc`, with dimensions [1x72x6x24x24]:

```
import gcgridobj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

# Generate the horizontal and vertical grid description
cs_grid = gcgridobj.cstools.gen_grid(24)
vrt_grid = gcgridobj.gc_vertical.standard_grid('GEOS_72L')

# Generate a world map to show the bottom layer
f_layer, ax_layer = plt.subplots(1,1,figsize=(8,5),subplot_kw={'projection': ccrs.PlateCarree()})

# plot_layer plots a global map and returns the image object (or list) along with a colorbar handle
im, cb = gcgridobj.plottools.plot_layer(conc[0,0,...],hrz_grid=cs_grid,ax=ax_layer)
# Set the color limits using the custom gcgridobj function which can handle cubed sphere data or lat lon data equally
# NB: If a third argument is passed, you can also set the color map. Here we arbitrarily use plasma
gcgridobj.plottools.set_clim(im,[0,70],cmap='plasma')

# Now generate a zonal plot
# First need to regrid to lat-lon - use the "standard" 1x1.25 grid

# Generate the target lat-lon grid
ll_grid = gcgridobj.latlontools.gen_grid(lat_stride=1.0,lon_stride=1.25,half_polar=True,center_180=True)

# Generate a regridder which can convert from CS data to lat-lon data
regridder = gcgridobj.regrid.gen_regridder(cs_grid,ll_grid)

# Regrid the data. NB: This assumes that the data is in units of something per unit area. Therefore
# regridding concentrations (kg/m3 = kg/m per m2) or mixing ratios (ppbv = vol/vol = m3/m3 = m3/m per m2)
# works fine, but mass per grid cell does not. To regrid extrinsic quantities like mass, you would first
# need to divide each grid cell by its horizontal area (cs_grid['area'].values) and then multiply the
# result by the area in the target grid (ll_grid['area'].values).
conc_ll = regridder(conc)

# Generate a set of zonal axes (latitude x altitude)
f_zonal, ax_zonal = plt.subplots(1,1,figsize=(8,5))

# plot_zonal plots a zonal map and returns the image object (or list) along with a colorbar handle
im_zonal, cb = gcgridobj.plottools.plot_zonal(np.mean(conc_ll[0,...],axis=-1),hrz_grid=ll_grid,vrt_grid=vrt_grid,ax=ax_zonal)
# Restrict the range to look at the troposphere only
ax_zonal.set_ylim(0,10) # In km
# Set the color limits and color map again
gcgridobj.plottools.set_clim(im_zonal,[0,100],cmap='viridis')
```

### Using the gcgo accessor

To make working with data a bit easier, gcgridobj now has the gcgo accessor. This allows you to work with an xarray dataset directly (including plotting) without needing to manually specify the horizontal grids. This can be used with both lat-lon and cubed-sphere grids. Below is an example, assuming that there is a standard GCHP output in the current directory.

```
import xarray as xr
from gcgridobj import gcgo, gc_vertical
# Open a GC-Classic or GCHP output file
ds = xr.open_dataset('GEOSChem.DefaultCollection.20220401_0000z.nc4')

# Important! Always set the vertical grid. This both encodes
# vertical information and initializes some internal variables
ds.gcgo.vrt_grid = gc_vertical.standard_grid('GEOS_72L')
# Set whether the dataset is encoded such that layer 1 is at the
# surface (positive = "up") or top of atmosphere (positive = "down")
# Up is assumed if not specified, so only worry about this if using
# emissions output
ds.gcgo.positive = 'up'
# Plot zonally. This will default to taking the average over the time
# axis - you can also specify time using the argument t = [time value]
ds.gcgo.plot_zonal('SpeciesConc_O3')
# Plot a layer. This will default to taking the average over the time
# axis. Z refers to the layer index, not the layer value.
ds.gcgo.plot_layer('SpeciesConc_O3',z=1)
```
