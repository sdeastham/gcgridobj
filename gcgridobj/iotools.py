#!/usr/bin/env python3
import os
from netCDF4 import Dataset
from datetime import datetime,timedelta
from . import latlontools
import numpy as np
import xarray as xr
try:
    from tqdm import tqdm
    use_tqdm = True
except:
    use_tqdm = False

class url_reader():
    def __init__(self,url,t_range=None,t_list=None,half_dt_offset=False,verbose=False,pbar=None):
        # url:            URL of the NetCDF file
        # t_range:        Start and end of the time range covered by the NetCDF file
        # t_list:         List of times to be read in
        # half_dt_offset: Apply a half-dt offset in timestamping (i.e. is this time-averaged)?
        self.url = url
        self.t_range = t_range
        self.t_list = []
        if t_list is not None:
            self.add_time(t_list)
        self.half_dt_offset = half_dt_offset
        self.verbose = verbose
        if use_tqdm:
            self.pbar = pbar
        else:
            self.counter = 0
        return

    def update_user(self,n=1):
        if use_tqdm:
            self.pbar.set_description('Reading from {:s}'.format(os.path.basename(self.url)))
            self.pbar.update(n)
        else:
            self.counter += n
            print('{:6d} entries read from {:s}'.format(self.counter,os.path.basename(self.url)))
        return
                
    def close(self):
        return

    def add_time(self,t_list):
        # Add times to be read in
        if self.t_range is not None:
            t_list_filtered = [t for t in t_list if t >= self.t_range[0] and t < self.t_range[1]]
        else:
            t_list_filtered = t_list
        self.t_list = self.t_list + t_list_filtered#.append(t_list_filtered[:])
        return
    def read_data(self,var_list,lon_bounds=None,lat_bounds=None,t_offset=None,hrz_grid=None,latlon_idx=None,read_single_times=True,t_ref=None):
        # In case we're called before add_time is used
        if len(self.t_list) == 0:
            print('No times found for URL {}'.format(self.url))
            return None
        # Reference time to be used for output
        if t_ref is None:
            t_ref = datetime(1970,1,1,0,0,0)
        # Open the file
        nc = Dataset(self.url,'r')
        try:
            for var in var_list:
                assert var in nc.variables, 'Variable {} not in variable list {}'.format(var,list(nc.variables))
            if hrz_grid is None:
                hrz_grid = latlontools.extract_grid(nc)
                # Need lon to be monotonically increasing
                #TODO: Modify to handle longitudes provided as [0,360]
                #TODO: Modify to handle bounds which cross the date line
                lon   = hrz_grid['lon'].values
                lon_b = hrz_grid['lon_b'].values
                lat   = hrz_grid['lat'].values
                lat_b = hrz_grid['lat_b'].values
                if lon_bounds is not None:
                    lon_bounds_mod = np.mod(np.array(lon_bounds) + 180,360) - 180
                    lon0 = np.argmax(lon_b > lon_bounds_mod[0]) - 1
                    lon1 = np.argmax(lon_b > lon_bounds_mod[1])
                else:
                    lon0 = 0
                    lon1 = len(hrz_grid['lon']) + 1
                if lat_bounds is not None:
                    lat0 = np.argmax(lat_b > lat_bounds[0]) - 1
                    lat1 = np.argmax(lat_b > lat_bounds[1])
                else:
                    lat0 = 0
                    lat1 = len(hrz_grid['lat']) + 1
                if not (lon_bounds is None and lat_bounds is None):
                    hrz_grid = latlontools.gen_grid_from_vec(lon[lon0:lon1],lat[lat0:lat1])
                latlon_idx = [lon0,lon1,lat0,lat1]
            else:
                if latlon_idx is None:
                    lon0 = 0
                    lon1 = len(hrz_grid['lon']) + 1
                    lat0 = 0
                    lat1 = len(hrz_grid['lat']) + 1
                else:
                    lon0, lon1, lat0, lat1 = latlon_idx[:]
            # Start with the grid definition only
            data = hrz_grid.copy(deep=True)

            # Figure out how this file is dealing with time
            t_units_str = nc['time'].units
            t_units = t_units_str.split(' ')[0]
            if t_units == 'seconds':
               dt_to_sec = 1.0
            elif t_units == 'minutes':
               dt_to_sec = 60.0
            elif t_units == 'hours':
               dt_to_sec = 60.0 * 60.0
            elif t_units == 'days':
               dt_to_sec = 60.0 * 60.0 * 24.0

            # GEOS-FP data files seem to be off by 48 hours
            if t_offset is None:
                t_offset_sec = 0
            else:
                t_offset_sec = t_offset.total_seconds()

            # Get the time vector after applying any necessary corrections
            t_nc = nc['time'][:].copy() * dt_to_sec + t_offset_sec

            # t0 is the time of the first sample in seconds since the base time
            t0 = t_nc[0]
            dt = t_nc[1] - t0
            # The "base" time
            # Remove milliseconds, if present
            if t_units_str.endswith(':0.0'):
                t_units_str = t_units_str.replace('0.0','00')
            # Special processing for GEOS nonsense
            if len(t_units_str.split(':')) == 3:
                fmt = '%Y-%m-%d %H:%M:%S'
            else:
                fmt = '%Y-%m-%d %H:%M'
            t_base_str = ' '.join(t_units_str.split(' ')[2:])
            # GEOS-FP uses a format which datetime does not like
            # Hard-code for now
            if t_base_str == '1-1-1 00:00:00':
                t_base = datetime(1,1,1,0,0,0)
            else:
                t_base = datetime.strptime(t_base_str,fmt)
            t_index_list = []
            t_stored = []
            # Time-averaged data is timestamped at the mid-point of the averaging period,
            # whereas instantaneous data is timestamped at the sampling time. To be
            # consistent we add dt/2 to the assessed target time and then take the floor
            if self.half_dt_offset:
                dt_first = dt/2.0
            else:
                dt_first = 0.0
            dt_base_ref = (t_base - t_ref).total_seconds()
            t0_dt = t_base + timedelta(seconds=t_nc[0])
            for t in self.t_list:
                # Figure out which index corresponds to the target time
                # If time-averaged, this is the averaging period covering
                # the target time. If instantaneous, this is the most recent
                # instantaneous sample taken (zero-order hold)
                t_index = int(np.floor((dt_first + (t - t0_dt).total_seconds())/dt))
                # Don't read the same data twice!
                if t_index not in t_index_list:
                    t_index_list.append(t_index)
                    # nc[...]*dts:  seconds between slice X and the base time
                    # dt_base_ref:  seconds between the base time and the reference date
                    t_true = (t_nc[t_index] + dt_base_ref)/3600.0
                    t_stored.append(t_true)

            # Advance the counter if there were duplicate requests
            if self.verbose and (len(t_index_list) < len(self.t_list)):
                self.update_user((len(self.t_list) - len(t_index_list))*len(var_list))

            data.coords['time'] = (('time'), t_stored)
            data['time'].attrs['units'] = 'hours since {:s}'.format(t_ref.strftime('%Y-%m-%d %H:%M:%S'))
            for var in var_list:
                data_coords = nc[var].dimensions
                for c in data_coords:
                    if c not in data.coords:
                        data.coords[c] = ((c), nc[c][...])
                # This can be used to prevent requests from getting too big
                if read_single_times:
                    for i_t, t in enumerate(t_index_list):
                        data_slice = nc[var][t,...,lat0:lat1,lon0:lon1]
                        if var not in data:
                            data[var] = (data_coords, np.zeros([len(t_index_list)] + list(data_slice.shape)))
                        data[var][i_t,...] = data_slice.copy()
                        if self.verbose:
                            self.update_user(1)
                else:
                    data[var] = (data_coords, nc[var][t_index_list,...,lat0:lat1,lon0:lon1].copy())
                    if self.verbose:
                        #print('Reading all {:d} slices for variable {:s} from {:s}'.format(len(t_index_list),var,os.path.basename(self.url)))
                        self.update_user(len(t_index_list))
                for attr in ['units','standard_name','long_name']:
                    if attr in nc[var].ncattrs():
                        attr_val = getattr(nc[var],attr)
                    else:
                        attr_val = 'unknown'
                    data[var].attrs[attr] = attr_val
        except:
            print('Failure while reading {}. If you see Access Denied but are able to acquire the file through (eg) ncdump, try updating your version of the netCDF4 module'.format(self.url))
            raise
        finally:
            nc.close()
        return data, hrz_grid, latlon_idx
    
def determine_M2_runID(collection,targ_date):
    # Determine the MERRA-2 filename based on the collection and target date
    # https://disc.gsfc.nasa.gov/information/documents?title=MERRA-2%20Data%20Access%20%E2%80%93%20Quick%20Guide#Data%20Filename%20Convention
    # Four data streams
    if targ_date >= datetime(2011,1,1,0,0,0):
        stream = 4
    elif targ_date >= datetime(2001,1,1,0,0,0):
        stream = 3
    elif targ_date >= datetime(1992,1,1,0,0,0):
        stream = 2
    else:
        stream = 1
    # Special cases where reprocessing was performed
    vn = 0
    if stream == 4:
        if (targ_date >= datetime(2021,5,1,0,0,0) and targ_date < datetime(2021,10,1,0,0,0)):
            # Fix for near-surface warm bias
            vn = 1
        elif (targ_date >= datetime(2020,9,1,0,0,0) and targ_date < datetime(2020,10,1,0,0,0)):
            # Fix to deal with problem in AIRS data
            vn = 1
    return '{:d}{:02d}'.format(stream,vn)
    
def read_GEOS(t_list,var_list,f_type,lon_bounds=None,lat_bounds=None,t_offset=None,src='GEOS-FP',
              verify_f_type=False,MERRA2_version='5.12.4',MERRA2_runID=None,verbose=True):
    if not isinstance(t_list,list):
        t_list = [t_list]
    if verbose and use_tqdm:
        pbar = tqdm(total=len(t_list)*len(var_list))
    else:
        pbar = None
    # Use try/finally to ensure that pbar is closed, if present
    try:
        hrz_grid=None
        if src == 'GEOS-FP':
            # Time values in GEOS-FP are a little off - best to set this explicitly
            if t_offset is None:
                #t_base = datetime(2017,12,1,0,0,0)
                #t_base = datetime(1,1,1,0,0,0) - timedelta(hours=48)
                t_offset = timedelta(hours=-48)
            url = 'https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/'
            url_full = url + f_type
            # Only one URL needed for GEOS-FP data
            is_inst = f_type.startswith('inst')
            url_list = [url_reader(url_full,t_list=t_list,verbose=verbose,pbar=pbar,half_dt_offset=(not is_inst))]
        elif src == 'MERRA-2':
            # Need Earthdata access
            assert os.path.isfile(os.path.join(os.path.expanduser('~'),'.netrc')), 'No .netrc file detected - have you set up your Earthdata credentials? https://disc.gsfc.nasa.gov/data-access'
            # Can allow t_base to be None in this case
            # Parse the file type using the data collection naming convention
            assert len(f_type) == len('M2TFHVGGG'), 'File type {} is not valid'.format(f_type)
            grid_type = f_type[4:6]
            # NX is 2D; NV, NP, NE are 3D
            is_2D = grid_type == 'NX'
            if is_2D:
                n_dims = 2
            else:
                n_dims = 3
            time_descriptor = f_type[2]
            freq = f_type[3]
            group = f_type[6:]
            # Optional additional checks
            if verify_f_type:
                assert grid_type in ['NX','NV','NP','NE'], 'Invalid grid type {}'.format(grid_type)
                assert f_type[:2] == 'M2', 'File type must begin with M2'
                assert time_descriptor in ['T','I'], 'Time descriptor must be T or I - got {}'.format(time_descriptor)
                assert freq in ['1','3','6','M','D','U','C'], 'Invalid frequency descriptor {}'.format(freq)
                assert group in ['ANA','ASM','AER','ADG','TDT','UDT','QDT','ODT',
                                 'GAS','GLC','CHM','OCN','LND','LFO','FLX','MST',
                                 'CLD','RAD','CSP','TRB','SLV','INT','NAV'], 'Invalid group {}'.format(group)
            # Build the collection name
            is_inst = time_descriptor == 'I'
            if is_inst:
                collection_t = 'inst'
            else:
                collection_t = 'tavg'
            collection = f'{collection_t:s}{freq:s}_{n_dims:d}d_{group.lower():s}_N{grid_type[-1].lower():s}'
                
            # 2D and 3D data stored in different places
            if is_2D:
                m2_src = 4
            else:
                m2_src = 5
            url = f'https://goldsmr{m2_src:d}.gesdisc.eosdis.nasa.gov/opendap/MERRA2/' + f_type + '.' + MERRA2_version + '/'
            # Generate the full list of URLs (one per day) to be read in
            day_list = []
            url_list = []
            for t in t_list:
                ymd = t.strftime('%Y%m%d')
                if ymd not in day_list:
                    day_list.append(ymd)
                    day_start = datetime(t.year,t.month,t.day,0,0,0)
                    day_end   = day_start + timedelta(days=1)
                    if MERRA2_runID is None:
                        runID = determine_M2_runID(collection,day_start)
                    else:
                        runID = MERRA2_runID
                    f_name = 'MERRA2_{:s}.{:s}.{:s}.nc4'.format(runID,collection,day_start.strftime('%Y%m%d'))
                    url_full = os.path.join(url,'{:4d}'.format(t.year),'{:02d}'.format(t.month),f_name)
                    url_obj = url_reader(url_full,t_range=[day_start,day_end],t_list=t_list,half_dt_offset=(not is_inst),verbose=verbose,pbar=pbar)
                    url_list.append(url_obj)
        else:
            raise ValueError(f'{src:s} not recognized as a source')
        data_vec = []
        hrz_grid = None
        ll_idx   = None
        for url_obj in url_list:
            data_item, hrz_grid, ll_idx = url_obj.read_data(var_list,lon_bounds=lon_bounds,lat_bounds=lat_bounds,
                                                            t_offset=t_offset,hrz_grid=hrz_grid,latlon_idx=ll_idx)
            data_vec.append(data_item)
        if verbose and use_tqdm:
            pbar.close()
        data = xr.concat(data_vec,'time')
    finally:
        if pbar is not None:
            pbar.close()
    return data, hrz_grid

def read_GEOS_range(t_range,src,f_type,**kwargs):
    # Need to determine the file frequency
    if src == 'GEOS-FP':
        freq = int(f_type.split('_')[0][-1])
    elif src == 'MERRA-2':
        freq = f_type[3]
        if freq not in ['1','3','6']:
            raise ValueError('Only 1/3/6-hourly data retrieval enabled for range requests')
        freq = int(freq)
    else:
        raise ValueError('Source {} not recognized'.format(src))
    t0 = min(t_range)
    t1 = max(t_range)
    n_reads = int(np.ceil((t1 - t0).total_seconds()/(3600.0 * freq)))
    t_list = [t0 + timedelta(hours=i*freq) for i in range(n_reads)]
    # Append the final time to be safe
    t_list.append(t1)
    return read_GEOS(t_list=t_list,f_type=f_type,src=src,**kwargs)
