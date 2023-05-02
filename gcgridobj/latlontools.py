import numpy as np
import xarray as xr
from . import physconstants

# Must have:
# 1. extract_grid (returns an xarray Dataset)
# 2. grid_area (returns a 6xNxN array)
# 3. gen_grid (returns an xarray Dataset)

def find_idx(targ_val,bound_vec_full,allow_loop=False):
    # Find the index, assuming evenly-spaced bounds (and allowing for half-polar)
    bound_vec = bound_vec_full.copy()
    # Force monotonicity (assuming this is longitude data)
    bound_correction = 0.0
    n_circles = 0
    for idx in range(len(bound_vec)-1):
       if bound_vec[idx+1] < bound_vec[idx]:
          n_circles += 1
          bound_vec[(idx+1):] += 360.0
    assert n_circles <= 1, 'Bounds vector looped more than once'
    # Get dx based on the average, excluding polar cells
    if len(bound_vec) < 4:
       # Argh - assume no polar cells, as we only have < 3 cells
       dx = np.asscalar((bound_vec[-1] - bound_vec[1])/len(bound_vec))
    else:
       dx = np.asscalar((bound_vec[-2] - bound_vec[1])/(len(bound_vec)-3))
    is_found = False
    n_loops = 0
    max_loops = 2
    while not is_found:
       # Have we started going off the deep end?
       if n_loops > max_loops:
          raise ValueError('Could not find target value in given range')
       # If above the first "definitely non-polar" cell edge..
       if targ_val > bound_vec[1]:
          # Within maximum bounds?
          if targ_val > bound_vec[-1]:
             if allow_loop:
                n_loops += 1
                targ_val -= 360.0
             else:
                raise ValueError('Target value above maximum')
          else:
             # Yes!
             is_found = True
             idx_val = 1 + np.floor((targ_val - bound_vec[1])/dx)
             if (idx_val == len(bound_vec)):
                # Edge case; force down into final cell
                idx_val -= 1
       elif targ_val < bound_vec[0]:
          if allow_loop:
             n_loops += 1
             targ_val += 360.0
          else:
             raise ValueError('Target value below minimum')
       else:
          # In first cell
          is_found = True
          idx_val = 0
    return int(idx_val)

def latlon_extract(nc_file,force_poles=True):
    # Attempt to extract lat and lon data from a netCDF4 dataset
    lon_name = None
    lat_name = None
    nc_vars = nc_file.variables.keys()
    if 'lon' in nc_vars:
        lon_name = 'lon'
    elif 'longitude' in nc_vars:
        lon_name = 'longitude'
    else:
        raise ValueError('No longitude information found')
    if 'lat' in nc_vars:
        lat_name = 'lat'
    elif 'latitude' in nc_vars:
        lat_name = 'latitude'
    else:
        raise ValueError('No latitude information found')
    lon = np.ma.filled(nc_file[lon_name][:],0.0)
    lat = np.ma.filled(nc_file[lat_name][:],0.0)
    lon_b = latlon_est_bnds(lon)
    lat_b = latlon_est_bnds(lat,force_poles=force_poles)
    return lon_b, lat_b, lon, lat

def extract_grid(nc_file,force_poles=True):
    # Extract lat/lons from netCDF4 dataset but return as an xarray object
    [lon_b,lat_b,lon,lat] = latlon_extract(nc_file,force_poles=force_poles)
    return xr.Dataset({'lat': (['lat'], lat),'lon': (['lon'], lon),
                     'lat_b': (['lat_b'], lat_b),'lon_b': (['lon_b'], lon_b),
                     'area': (['lat','lon'], latlon_gridarea(lon_b,lat_b))})

def grid_area(lon_b=None, lat_b=None, hrz_grid=None, r_earth=None):

    if hrz_grid is not None:
       assert lon_b is None and lat_b is None, "Must provide either a grid object or both the latitude and longitude aedges"
       lon_b = hrz_grid['lon_b']
       lat_b = hrz_grid['lat_b']
    else:
       assert lon_b is not None and lat_b is not None, "Need both lon_b and lat_b if grid object not supplied"

    if r_earth is None:
       r_earth = physconstants.R_earth

    # Calculate grid areas (m2) for a rectilinear grid
    lon_abs = []
    lastlon = lon_b[0]
    for i,lon in enumerate(lon_b):
        while lon < lastlon:
            lon += 360.0
        lon_abs.append(lon)
        lastlon = lon
   
    n_lat = lat_b.size - 1
    n_lon = lon_b.size - 1

    # Total surface area in each meridional band (allows for a regional domain)
    merid_area = 2*np.pi*r_earth*r_earth*(lon_abs[-1]-lon_abs[0])/(360.0*n_lon)
    grid_area = np.empty([n_lon,n_lat])
    lat_b_rad = np.pi * lat_b / 180.0
    for i_lat in range(n_lat):
        # Fraction of meridional area which applies
        sin_diff = np.sin(lat_b_rad[i_lat+1])-np.sin(lat_b_rad[i_lat])
        grid_area[:,i_lat] = sin_diff * merid_area

    # Transpose this - convention is [lat, lon]
    grid_area = np.transpose(grid_area)
    return grid_area

def latlon_est_bnds(indata,force_poles=False):
    # Estimate lat/lon edges based on a vector of mid-points
    dx = np.median(np.diff(indata))
    x0 = indata.data[0] - (dx/2.0)
    outdata = np.array([x0 + i*dx for i in range(0,indata.size + 1)])
    if force_poles:
        outdata[outdata<-90] = -90.0
        outdata[outdata>90] = 90.0
    return outdata

def latlon_est_mid(indata):
    # Calculate midpoints from edges
    return np.array([0.5*(indata[i] + indata[i+1]) for i in range(len(indata)-1)])
    #return (indata[1:] + indata[:-1]) / 2.0

def make_llvec(lbnd,dl):
    n_mid = int(np.round((lbnd[1]-lbnd[0])/dl))
    ledge = np.linspace(start=lbnd[0],stop=lbnd[1],num=n_mid+1)
    lmid  = latlon_est_mid(ledge)
    return lmid,ledge

def gen_grid(lon_stride,lat_stride,half_polar=False,center_180=False,lon_range=None,lat_range=None):
    # Define a simple rectilinear grid
    
    # Generate longitude edge vector
    n_lon = int(np.round(360.0 / lon_stride))
    if center_180:
        start_lon = (-180.0) - (lon_stride/2.0)
    else:
        start_lon = -180.0
    lon_b = np.linspace(start_lon,start_lon+360.0,n_lon + 1)
    
    # Generate latitude edge vector
    # If half-polar, first and last cell are centered on the poles
    n_lat = int(np.round(180.0/lat_stride))
    if half_polar:
        n_lat += 1
        lat_max = 90.0 + (lat_stride/2.0)
    else:
        lat_max = 90.0
    lat_b = np.linspace(-lat_max,lat_max,n_lat + 1)
    
    # This will deal with the half-polar issue
    lat_b[0]  = -90.0
    lat_b[-1] =  90.0
    
    lat = latlon_est_mid(lat_b)
    lon = latlon_est_mid(lon_b)
    
    if lon_range is not None:
        lon = lon[lon>=(lon_range[0] - lon_stride*0.01)]
        lon = lon[lon<=(lon_range[-1] + lon_stride*0.01)]
        lon_b = latlon_est_bnds(lon)
        
        lat = lat[lat>=(lat_range[0] - lat_stride*0.01)]
        lat = lat[lat<=(lat_range[-1] + lat_stride*0.01)]
        lat_b = latlon_est_bnds(lat)
    
    return xr.Dataset({'lat': (['lat'], lat),'lon': (['lon'], lon),
                     'lat_b': (['lat_b'], lat_b),'lon_b': (['lon_b'], lon_b),
                     'area': (['lat','lon'], latlon_gridarea(lon_b,lat_b))})

def gen_grid_from_vec(lon,lat):
    # Generate a simple rectilinear grid from a vector of lats and lons
    lon_a = np.asarray(lon)
    lat_a = np.asarray(lat)
    nlon = np.size(lon_a)
    nlat = np.size(lat_a)
    lon_b = np.zeros(nlon + 1)
    lat_b = np.zeros(nlat + 1)
    lon_b[1:-1] = (lon_a[1:] + lon_a[:-1])/2.0
    lat_b[1:-1] = (lat_a[1:] + lat_a[:-1])/2.0
    lon_b[0]  = lon_b[1] - (2.0*(lon_b[1]-lon[0]))
    lat_b[0]  = lat_b[1] - (2.0*(lat_b[1]-lat[0]))
    lon_b[-1]  = lon_b[-2] + (2.0*(lon[-1]-lon_b[-2]))
    lat_b[-1]  = lat_b[-2] + (2.0*(lat[-1]-lat_b[-2]))
    return xr.Dataset({'lat': (['lat'], lat_a),'lon': (['lon'], lon_a),
                     'lat_b': (['lat_b'], lat_b),'lon_b': (['lon_b'], lon_b),
                     'area': (['lat','lon'], latlon_gridarea(lon_b,lat_b))})

# Old aliases
latlon_gridarea = grid_area
latlon_extract_grid = extract_grid
gen_hrz_grid = gen_grid
