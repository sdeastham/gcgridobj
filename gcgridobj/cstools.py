import numpy as np
import xarray as xr
import cubedsphere

# For point-finding
import pyproj
import shapely.ops
import shapely.geometry

# Must have:
# 1. extract_grid (returns an xarray Dataset)
# 2. grid_area (returns a 6xNxN array)
# 3. gen_grid (returns an xarray Dataset)

def extract_grid(ds,src_var='Xdim'):
    # Extract grid from xarray dataset but return a cubedsphere grid
    n_cs = ds[src_var].shape[-1]
    #return cubedsphere.csgrid_GMAO(n_cs)
    return gen_grid(n_cs)

def face_area(lon_b, lat_b, r_sphere = 6.375e6):
    """Calculate area of cubed-sphere grid cells on one face
    Inputs must be in degrees. Edge arrays must be
    shaped [N+1 x N+1]
    """
    
    # Convert inputs to radians
    lon_b_rad = lon_b * np.pi / 180.0
    lat_b_rad = lat_b * np.pi / 180.0
    
    r_sq = r_sphere * r_sphere
    n_cs = lon_b.shape[1] - 1
    
    # Allocate output array
    cs_area = np.zeros((n_cs,n_cs))
    
    # Ordering
    valid_combo = np.array([[1,2,4],[2,3,1],[3,2,4],[4,1,3]]) - 1
    
    for i_lon in range(n_cs):
        for i_lat in range(n_cs):
            lon_corner = np.zeros(4)
            lat_corner = np.zeros(4)
            xyz_corner = np.zeros((4,3))
            for i_vert in range(4):
                x_lon = i_lon + (i_vert > 1)
                x_lat = i_lat + (i_vert == 0 or i_vert == 3)
                lon_corner[i_vert] = lon_b_rad[x_lon,x_lat]
                lat_corner[i_vert] = lat_b_rad[x_lon,x_lat]
            for i_vert in range(4):
                xyz_corner[i_vert,:] = ll2xyz(lon_corner[i_vert],lat_corner[i_vert])
            tot_ang = 0.0
            for i_corner in range(4):
                curr_combo = valid_combo[i_corner,:]
                xyz_mini = np.zeros((3,3))
                for i_mini in range(3):
                    xyz_mini[i_mini,:] = xyz_corner[curr_combo[i_mini],:]
                curr_ang = sphere_angle(xyz_mini[0,:],xyz_mini[1,:],xyz_mini[2,:])
                tot_ang += curr_ang
            cs_area[i_lon,i_lat] = r_sq * (tot_ang - (2.0*np.pi))
    
    return cs_area

def ll2xyz(lon_pt,lat_pt):
    """Converts a lon/lat pair (in radians) to cartesian co-ordinates
    Vector should point to the surface of the unit sphere"""

    xPt = np.cos(lat_pt) * np.cos(lon_pt)
    yPt = np.cos(lat_pt) * np.sin(lon_pt)
    zPt = np.sin(lat_pt)
    return [xPt,yPt,zPt]

def sphere_angle(e1,e2,e3):
    # e1: Mid-point
    # e2 and e3 to either side
    pVec = np.ones(3)
    qVec = np.ones(3)
    pVec[0] = e1[1]*e2[2] - e1[2]*e2[1]
    pVec[1] = e1[2]*e2[0] - e1[0]*e2[2]
    pVec[2] = e1[0]*e2[1] - e1[1]*e2[0]

    qVec[0] = e1[1]*e3[2] - e1[2]*e3[1]
    qVec[1] = e1[2]*e3[0] - e1[0]*e3[2]
    qVec[2] = e1[0]*e3[1] - e1[1]*e3[0]
    ddd = np.sum(pVec*pVec) * np.sum(qVec*qVec)
    if ddd <= 0.0:
        angle = 0.0;
    else:
        ddd = np.sum(pVec*qVec)/np.sqrt(ddd);
        if (np.abs(ddd)>1.0):
            angle = np.pi/2.0;
        else:
            angle = np.arccos(ddd);

    return angle

def grid_area(cs_grid=None,cs_res=None):
    """Return area in m2 for each cell in a cubed-sphere grid
    Uses GMAO indexing convention (6xNxN)
    """
    # Calculate area on a cubed sphere
    if cs_res is None:
        cs_res = cs_grid['lon_b'].shape[-1] - 1
    elif cs_grid is None:
        cs_grid = cubedsphere.csgrid_GMAO(cs_res)
    elif cs_grid is not None and cs_res is not None:
        assert cs_res == cs_grid['lon_b'].shape[-1], 'Routine grid_area received inconsistent inputs' 
    cs_area = np.zeros((6,cs_res,cs_res))
    cs_area[0,:,:] = face_area(cs_grid['lon_b'][0,:,:],cs_grid['lat_b'][0,:,:])
    for i_face in range(1,6):
        cs_area[i_face,:,:] = cs_area[0,:,:].copy()
    return cs_area

def gen_grid(n_cs):
    cs_temp = cubedsphere.csgrid_GMAO(n_cs)
    return xr.Dataset({'nf':     (['nf'],np.array(range(6))),
                       'Ydim':   (['Ydim'],np.array(range(n_cs))),
                       'Xdim':   (['Xdim'],np.array(range(n_cs))),
                       'Ydim_b': (['Ydim_b'],np.array(range(n_cs+1))),
                       'Xdim_b': (['Xdim_b'],np.array(range(n_cs+1))),
                       'lat':    (['nf','Ydim','Xdim'], cs_temp['lat']),
                       'lon':    (['nf','Ydim','Xdim'], cs_temp['lon']),
	     	  'lat_b':  (['nf','Ydim_b','Xdim_b'], cs_temp['lat_b']),
	     	  'lon_b':  (['nf','Ydim_b','Xdim_b'], cs_temp['lon_b']),
                       'area':   (['nf','Ydim','Xdim'], grid_area(cs_temp))})

def corners_to_xy(xc, yc):
    """ Creates xy coordinates for each grid-box. The shape is (n, n, 5) where n is the cubed-sphere size.
    Developed, tested, and supplied by Liam Bindle.

    :param xc: grid-box corner longitudes; shape (n+1, n+1)
    :param yc: grid-box corner latitudes; shape (n+1, n+1)
    :return: grid-box xy coordinates
    """
    p0 = slice(0, -1)
    p1 = slice(1, None)
    boxes_x = np.moveaxis(np.array([xc[p0, p0], xc[p1, p0], xc[p1, p1], xc[p0, p1], xc[p0, p0]]), 0, -1)
    boxes_y = np.moveaxis(np.array([yc[p0, p0], yc[p1, p0], yc[p1, p1], yc[p0, p1], yc[p0, p0]]), 0, -1)
    return np.moveaxis(np.array([boxes_x, boxes_y]), 0, -1)


def central_angle(x0, y0, x1, y1):
    """ Returns the distance (central angle) between coordinates (x0, y0) and (x1, y1). This is vectorizable.
    Developed, tested, and supplied by Liam Bindle.

    :param x0: pt0's longitude (degrees)
    :param y0: pt0's latitude  (degrees)
    :param x1: pt1's longitude (degrees)
    :param y1: pt1's latitude  (degrees)
    :return: Distance          (degrees)
    """
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180
    x0 = x0 * DEG2RAD
    x1 = x1 * DEG2RAD
    y0 = y0 * DEG2RAD
    y1 = y1 * DEG2RAD
    return np.arccos(np.sin(y0) * np.sin(y1) + np.cos(y0) * np.cos(y1) * np.cos(np.abs(x0-x1))) * RAD2DEG


def find_index(lat,lon,grid):
    # Based on a routine developed, tested, and supplied by Liam Bindle.
    lon_vec = np.asarray(lon)
    lat_vec = np.asarray(lat)
    n_find = lon_vec.size

    # Get the corners
    x_corners = grid['lon_b'].values
    y_corners = grid['lat_b'].values
    x_centers = grid['lon'].values
    y_centers = grid['lat'].values
    x_centers_flat = x_centers.flatten()
    y_centers_flat = y_centers.flatten()

    cs_size = x_centers.shape[-1]

    # Generate everything that will be reused
    # Get XY polygon definitions for grid boxes
    xy = np.zeros((6, cs_size, cs_size, 5, 2))  # 5 (x,y) points defining polygon corners (first and last are same)
    for nf in range(6):
        xy[nf, ...] = corners_to_xy(xc=x_corners[nf, :, :], yc=y_corners[nf, :, :])
    latlon_crs = pyproj.Proj("+proj=latlon")

    # Find 4 shortest distances to (x_find, y_find)
    idx = np.full((3,n_find),np.int(0))
    for x_find, y_find, i_find in zip(np.nditer(lon_vec),np.nditer(lat_vec),list(range(n_find))):
        # Center on x_find, y_find
        gnomonic_crs = pyproj.Proj(f'+proj=gnom +lat_0={y_find} +lon_0={x_find}')

        # Generate all distances
        distances = central_angle(x_find, y_find, x_centers_flat, y_centers_flat)
        four_nearest_indexes = np.argpartition(distances, 4)[:4]

        # Unravel 4 smallest indexes
        four_nearest_indexes = np.unravel_index(four_nearest_indexes, (6, cs_size, cs_size))
        four_nearest_xy = xy[four_nearest_indexes]
        four_nearest_polygons = [shapely.geometry.Polygon(polygon_xy) for polygon_xy in four_nearest_xy]

        # Transform to gnomonic projection
        gno_transform = pyproj.Transformer.from_proj(latlon_crs, gnomonic_crs, always_xy=True).transform
        four_nearest_polygons_gno = [shapely.ops.transform(gno_transform, polygon) for polygon in four_nearest_polygons]

        # Figure out which polygon contains the point
        Xy_find = shapely.geometry.Point(x_find, y_find)
        Xy_find_GNO = shapely.ops.transform(gno_transform, Xy_find)
        polygon_contains_point = [polygon.contains(Xy_find_GNO) for polygon in four_nearest_polygons_gno]

        #assert np.count_nonzero(polygon_contains_point) == 1
        assert np.count_nonzero(polygon_contains_point) > 0
        # The first will be selected, if more than one
        polygon_with_point = np.argmax(polygon_contains_point)

        # Get original index
        nf = four_nearest_indexes[0][polygon_with_point]
        YDim= four_nearest_indexes[1][polygon_with_point]
        XDim= four_nearest_indexes[2][polygon_with_point]

        idx[:,i_find] = [nf,YDim,XDim]
    return idx
