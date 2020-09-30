gcgridobj

This is a minimal python package which provides objects and routines useful for manipulating GEOS-Chem output data in Python 3.

## Installation

### Requirements

* https://github.com/jiaweizhuang/cubedsphere
* https://github.com/jiaweizhuang/xesmf

## Examples

### Generating a horizontal grid description

Horizontal grid descriptions are just `xarray` datasets which include either:
* 1-D `lon` and `lat` variables
* 2-D `xDim` and `yDim` variables
To generate a grid description, there are three options:
1. Use a pre-calculated grid description from `gcgridobj.gc_horizontal`. For example, GEOS-Chem's 4x5 grid is taken from GMAO, and can be retrieved as `grid = gcgridobj.gc_horizontal.gmao_4x5_global`
2. Generate a rectilinear latitude-longitude grid using `gcgridobj.latlontools`. For example, a simple 1x1 grid can be generated as `grid = gcgridobj.latlontools.gen_grid(lon_stride=1,lat_stride=1,half_polar=False,center_180=False)`. This function can also be used to generate nested grids.
3. Generate a gnomonic cubed-sphere grid using `gcridobj.cstools`. For exampe, a c24 grid can be generated as `grid = gcgridobj.cstools.gen_grid(24)`.

The horizontal grid descriptions are used to handle all plotting and regridding in `gcgridobj`. They also carry a copy of the grid cell area - for a given grid description `grid`, take a look at `grid.area.values`.

### Generating a vertical grid description

Vertical grid descriptions are based on a rather hokey custom class in `gcgridobj.gc_vertical`. There are several pre-generated examples in there, such as the GMAO 72-layer grid (`gcgridobj.gc_vertical.GEOS_72L_grid`), as well as the GEOS-Chem 47-layer "lumped" counterpart (`GEOS_47L_grid`), and the CAM 26-layer grid (`CAM_26L_grid`). To generate a new one, all you need are the `AP` and `BP` hybrid grid values. `AP` should be in units of hPa, and `BP` should be unitless. To create a new vertical grid, call `vrt_grid = gcgridobj.gc_vertical.vert_gird(AP=new_AP,BP=new_BP)`.

The vertical grid description can be used to easily calculate the pressure edges (`vrt_grid.p_edge`), and midpoints (`vrt_grid.p_mid`). These are calculated assuming a specific surface pressure - to change it, call `vrt_grid.p_sfc = new_sfc_p` (again using hPa as your units). The class also includes methods to estimate the grid mid-point and edge altitudes using the International Standard Atmosphere. To get the altitude edges, for example, call `vrt_grid.z_edge_ISA()`. The returned array is defined in meters, and again assumes that the surface pressure is equal to `vrt_grid.p_sfc`. Finally, you can generate a 3-D pressure edge field by calling `vrt_grid.gen_p_field(p_sfc_2D)`.

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
src_vrt = gcgridobj.gc_vertical.GMAO_72L_grid
dst_vrt = gcgridobj.gc_vertical.CAM_26L_grid
```

and generate a vertical regridder object:

`vrt_regridder = gcgridobj.regrid.vrt_regridder(src_vrt,dst_vrt)`

Now, to regrid vertically, take any data set on your source vertical grid - say `data_src` - and call `data_dst = regridder(data_src)`. As before, the data are expected to be in units which are normalized per unit pressure (e.g. parts per billion by volume).

This vertical regrid operation does not yet account for changes in surface pressure.
