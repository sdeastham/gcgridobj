#!/usr/bin/env python3
import os
from netCDF4 import Dataset
from datetime import datetime,timedelta
from . import latlontools
import numpy as np

class url_reader():
    def __init__(self,url,t_range=None,t_list=None):
        # url: URL of the NetCDF file
        # t_range: Start and end of the time range covered by the NetCDF file
        # t_list: 
        self.url = url
        self.t_range = t_range
        self.t_list = []
        if t_list is not None:
            self.add_time(t_list)
        return
    def add_time(self,t_list):
        # Add times to be read in
        if self.t_range is not None:
            t_list_filtered = t_list[np.logical_and(t_list >= self.t_range[0],t_list < self.t_range[1])]
        else:
            t_list_filtered = t_list
        self.t_list = self.t_list + t_list_filtered#.append(t_list_filtered[:])
        return
    def read_data(self,var_list,lon_bounds=None,lat_bounds=None,t_base=None,get_hrz_grid=True,read_single_times=True):
        # In case we're called before add_time is used
        if len(self.t_list) == 0:
            print('No times found for URL {}'.format(self.url))
            return None
        # Output object
        data = {}
        # Open the file
        nc = Dataset(self.url,'r')
        try:
            for var in var_list:
                assert var in nc.variables, 'Variable {} not in variable list {}'.format(var,list(nc.variables))
            if get_hrz_grid:
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
                data['hrz_grid'] = hrz_grid

            # Date of first entry, in days since 0001-01-01 00:00:00
            # In reality there seems to be some confusion, so just use as an offset and
            # assume it corresponds to 2017-12-01 plus dt/2
            t0 = float(nc['time'][0])
            dt = 60*60*24*(float(nc['time'][1]) - t0)
            t_index_list = []
            for t in self.t_list:
                # For time-averaged fields, t_index is the entry spanning the period requested by t
                # For instantaneous fields, t_index is the entry prior to t
                t_index = int(np.floor((t - t_base).total_seconds()/dt))
                # Don't read the same data twice!
                if t_index not in t_index_list:
                    t_index_list.append(t_index)
            data['time'] = [t_base + timedelta(seconds=int((t-t0)*60*60*24)) for t in nc['time'][t_index_list].copy()]
            # This can be used to prevent requests from getting too big
            if read_single_times:
                for var in var_list:
                    for i_t, t in enumerate(t_index_list):
                        data_slice = nc[var][t,...,lat0:lat1,lon0:lon1]
                        if var not in data:
                            data[var] = np.zeros([len(t_index_list)] + list(data_slice.shape))
                        data[var][i_t,...] = data_slice.copy()
            else:
                for var in var_list:
                    data[var] = nc[var][t_index_list,...,lat0:lat1,lon0:lon1].copy()
        finally:
            nc.close()
        return data

def stitch_data(data_vec,cleanup_data=False):
    if len(data_vec) == 1:
        return data_vec[0]
    data_out = {'time': []}
    n_times = 0
    for data_in in data_vec:
        n_times += len(data_in['time'])
        data_out['time'].append(data_in['time'])
        if ('hrz_grid' not in data_out) and (hrz_grid in data_in):
            data_out['hrz_grid'] = data_in['hrz_grid']
    var_list = data_vec[0].keys()
    for var in var_list:
        if var not in data_out:
            data_out[var] = np.zeros([n_times] + list(data_vec[0][var].shape[1:]))
            i1 = 0
            for data_in in data_vec:
                i0 = i1
                i1 += len(data_in['time'])
                data_out[var][i0:i1,...] = data_in[var].copy()
                if cleanup_data:
                    data_in[var] = None
    return data_out
    
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
    if stream == 4:
        if (targ_date >= datetime(2021,5,1,0,0,0) and targ_date < datetime(2021,10,1,0,0,0)):
            # Fix for near-surface warm bias
            vn = 1
        elif (targ_date >= datetime(2020,9,1,0,0,0) and targ_date < datetime(2020,10,1,0,0,0)):
            # Fix to deal with problem in AIRS data
            vn = 1
    return '{:d}{:02d}'.format(stream,vn)
    
def read_GEOS(t_list,var_list,f_type,lon_bounds=None,lat_bounds=None,t_base=None,src='GEOS-FP',
              verify_f_type=False,MERRA2_version='5.12.4',MERRA2_runID=None):
    if t_base is None:
        t_base = datetime(2017,12,1,0,0,0)
    if not isinstance(t_list,list):
        t_list = [t_list]
    hrz_grid=None
    if src == 'GEOS-FP':
        url = 'https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/'
        url_full = url + f_type
        # Only one URL needed for GEOS-FP data
        url_list = [url_reader(url_full,t_list=t_list)]
    elif src == 'MERRA-2':
        # Parse the file type using the data collection naming convention
        assert len(f_type) == len('M2TFHVGGG'), 'File type {} is not valid'.format(f_type)
        grid_type = f_type[4:5]
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
        if time_descriptor == 'I':
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
                url_obj = url_reader(url_full,t_range=[day_start,day_end],t_list=t_list)
                url_list.append(url_obj)
    data_vec = []
    hrz_grid = None
    for url_obj in url_list:
        data_vec.append(url_obj.read_data(var_list,lon_bounds=lon_bounds,lat_bounds=lat_bounds,t_base=t_base,
                                          get_hrz_grid=hrz_grid is None))
        if data_vec[-1] is not None and hrz_grid is None and 'hrz_grid' in data_vec[-1]:
            hrz_grid = data_vec[-1]['hrz_grid']
    data = stitch_data(data_vec)
    return data
