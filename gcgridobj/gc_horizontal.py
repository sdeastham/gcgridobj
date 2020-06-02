from gcgridobj import latlontools, regrid
import warnings

# Define all the major GEOS-Chem grids
# Horizontal global grids first
gmao_4x5_global       = latlontools.gen_hrz_grid(lat_stride=4.0, lon_stride=5.0, half_polar=True,center_180=True)
gmao_2x25_global      = latlontools.gen_hrz_grid(lat_stride=2.0, lon_stride=2.5, half_polar=True,center_180=True)
gmao_1x1_global       = latlontools.gen_hrz_grid(lat_stride=1.0, lon_stride=1.0, half_polar=False,center_180=True)
gmao_05x0666_global   = latlontools.gen_hrz_grid(lat_stride=0.5, lon_stride=2/3, half_polar=True,center_180=True)
gmao_05x0625_global   = latlontools.gen_hrz_grid(lat_stride=0.5, lon_stride=5/8, half_polar=True,center_180=True)
gmao_025x03125_global = latlontools.gen_hrz_grid(lat_stride=0.25,lon_stride=5/16,half_polar=True,center_180=True)

# Horizontal nested grids
gmao_05x0666_us       = latlontools.gen_hrz_grid(lat_stride=0.5, lon_stride=2/3, half_polar=True,center_180=True,lon_range=[-140, -40],lat_range=[ 10, 70])
gmao_05x0666_ch       = latlontools.gen_hrz_grid(lat_stride=0.5, lon_stride=2/3, half_polar=True,center_180=True,lon_range=[  70, 150],lat_range=[-11, 55])

gmao_05x0625_as       = latlontools.gen_hrz_grid(lat_stride=0.5, lon_stride=5/8, half_polar=True,center_180=True,lon_range=[  60, 150],lat_range=[-11, 55])
gmao_05x0625_eu       = latlontools.gen_hrz_grid(lat_stride=0.5, lon_stride=5/8, half_polar=True,center_180=True,lon_range=[ -30,  50],lat_range=[ 30, 70])
gmao_05x0625_us       = latlontools.gen_hrz_grid(lat_stride=0.5, lon_stride=5/8, half_polar=True,center_180=True,lon_range=[-140, -40],lat_range=[ 10, 70])

# All grids
global_grid_inventory = [gmao_4x5_global,
                         gmao_2x25_global,
                         gmao_1x1_global,
                         gmao_05x0666_global,
                         gmao_05x0625_global,
                         gmao_025x03125_global]

nested_grid_inventory = [gmao_05x0666_us,
                         gmao_05x0666_ch,
                         gmao_05x0625_as,
                         gmao_05x0625_eu,
                         gmao_05x0625_us]

def get_grid(grid_shape,is_nested=None,first_call=True):
   warnings.warn('gc_horizontal.get_grid is deprecated. Please change code to use regrid.guess_ll_grid instead')
   # Changed ordering fron [lon, lat] to [lat, lon]
   return regrid.guess_ll_grid([grid_shape[1],grid_shape[0]],is_nested,first_call)
