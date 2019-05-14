import numpy as np
import scipy.sparse
from . import atmos_isa_mini

class vert_grid:
    def __init__(self,AP=None,BP=None,p_sfc=1013.25):
        if (AP.size != BP.size) or (AP is None):
            # Throw error?
            print('Inconsistent vertical grid specification')
        self.AP = np.array(AP)
        self.BP = np.array(BP)
        self.p_sfc = p_sfc
    def p_edge(self):
        # Calculate pressure edges using eta coordinate
        return self.AP + self.BP * self.p_sfc
    def p_mid(self):
        p_edge = self.p_edge()
        return (p_edge[1:]+p_edge[:-1])/2.0
    def z_edge_ISA(self):
        return atmos_isa_mini.pressure_to_altitude(self.p_edge()*100.0)
    def z_mid_ISA(self):
        return atmos_isa_mini.pressure_to_altitude(self.p_mid()*100.0)

# Standard vertical grids
_GEOS_72L_AP = np.array([ 0.000000e+00, 4.804826e-02, 6.593752e+00, 1.313480e+01,
         1.961311e+01, 2.609201e+01, 3.257081e+01, 3.898201e+01,
         4.533901e+01, 5.169611e+01, 5.805321e+01, 6.436264e+01,
         7.062198e+01, 7.883422e+01, 8.909992e+01, 9.936521e+01,
         1.091817e+02, 1.189586e+02, 1.286959e+02, 1.429100e+02,
         1.562600e+02, 1.696090e+02, 1.816190e+02, 1.930970e+02,
         2.032590e+02, 2.121500e+02, 2.187760e+02, 2.238980e+02,
         2.243630e+02, 2.168650e+02, 2.011920e+02, 1.769300e+02,
         1.503930e+02, 1.278370e+02, 1.086630e+02, 9.236572e+01,
         7.851231e+01, 6.660341e+01, 5.638791e+01, 4.764391e+01,
         4.017541e+01, 3.381001e+01, 2.836781e+01, 2.373041e+01,
         1.979160e+01, 1.645710e+01, 1.364340e+01, 1.127690e+01,
         9.292942e+00, 7.619842e+00, 6.216801e+00, 5.046801e+00,
         4.076571e+00, 3.276431e+00, 2.620211e+00, 2.084970e+00,
         1.650790e+00, 1.300510e+00, 1.019440e+00, 7.951341e-01,
         6.167791e-01, 4.758061e-01, 3.650411e-01, 2.785261e-01,
         2.113490e-01, 1.594950e-01, 1.197030e-01, 8.934502e-02,
         6.600001e-02, 4.758501e-02, 3.270000e-02, 2.000000e-02,
         1.000000e-02 ])

_GEOS_72L_BP = np.array([ 1.000000e+00, 9.849520e-01, 9.634060e-01, 9.418650e-01,
         9.203870e-01, 8.989080e-01, 8.774290e-01, 8.560180e-01,
         8.346609e-01, 8.133039e-01, 7.919469e-01, 7.706375e-01,
         7.493782e-01, 7.211660e-01, 6.858999e-01, 6.506349e-01,
         6.158184e-01, 5.810415e-01, 5.463042e-01, 4.945902e-01,
         4.437402e-01, 3.928911e-01, 3.433811e-01, 2.944031e-01,
         2.467411e-01, 2.003501e-01, 1.562241e-01, 1.136021e-01,
         6.372006e-02, 2.801004e-02, 6.960025e-03, 8.175413e-09,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
         0.000000e+00 ])

GEOS_72L_grid = vert_grid(_GEOS_72L_AP, _GEOS_72L_BP)

# Reduced grid
_GEOS_47L_AP = np.zeros(48)
_GEOS_47L_BP = np.zeros(48)

# Fill in the values for the surface
_GEOS_47L_AP[0] = _GEOS_72L_AP[0]
_GEOS_47L_BP[0] = _GEOS_72L_BP[0]

# Build the GEOS 72-layer to 47-layer mapping matrix at the same time
__xmat_i = np.zeros((72))
__xmat_j = np.zeros((72))
__xmat_s = np.zeros((72))

# Index here is the 1-indexed layer number
for __i_lev in range(1,37):
    # Map from 1-indexing to 0-indexing
    __x_lev = __i_lev - 1
    # Sparse matrix for regridding
    # Below layer 37, it's 1:1
    __xct = __x_lev
    __xmat_i[__xct] = __x_lev
    __xmat_j[__xct] = __x_lev
    __xmat_s[__xct] = 1.0
    # Copy over the pressure edge for the top of the grid cell
    _GEOS_47L_AP[__i_lev] = _GEOS_72L_AP[__i_lev]
    _GEOS_47L_BP[__i_lev] = _GEOS_72L_BP[__i_lev]

# Now deal with the lumped layers
__skip_size_vec = [2,4]
__number_lumped = [4,7]

# Initialize
__i_lev = 36
__i_lev_72 = 36
for __lump_seg in range(2):
    __skip_size = __skip_size_vec[__lump_seg]
    # 1-indexed starting point in the 47-layer grid
    __first_lev_47 = __i_lev + 1
    __first_lev_72 = __i_lev_72 + 1
    
    # Loop over the coarse vertical levels (47-layer grid)
    for __i_lev_offset in range(__number_lumped[__lump_seg]):
        # __i_lev is the index for the current level on the 47-level grid
        __i_lev = __first_lev_47 + __i_lev_offset
        # Map from 1-indexing to 0-indexing
        __x_lev = __i_lev - 1
        
        # Get the 1-indexed location of the last layer in the 72-layer grid 
        # which is below the start of the current lumping region
        __i_lev_72_base = __first_lev_72 + (__i_lev_offset*__skip_size) - 1
        
        # Get the 1-indexed location of the uppermost level in the 72-layer
        # grid which is within the target layer on the 47-layer grid
        __i_lev_72 = __i_lev_72_base + __skip_size
        
        # Do the pressure edges first
        # These are the 0-indexed locations of the upper edge for the 
        # target layers in 47- and 72-layer grids
        _GEOS_47L_AP[__i_lev] = _GEOS_72L_AP[__i_lev_72]
        _GEOS_47L_BP[__i_lev] = _GEOS_72L_BP[__i_lev_72]
        
        # Get the total pressure delta across the layer on the lumped grid
        # We are within the fixed pressure levels so don't need to account
        # for variations in surface pressure
        __dp_total = _GEOS_47L_AP[__i_lev-1] - _GEOS_47L_AP[__i_lev]
        
        # Now figure out the mapping
        for __i_lev_offset_72 in range(__skip_size):
            # Source layer in the 72 layer grid (0-indexed)
            __x_lev_72 = __i_lev_72_base + __i_lev_offset_72
            __xct = __x_lev_72
            __xmat_i[__xct] = __x_lev_72
            # Target in the 47 layer grid
            __xmat_j[__xct] = __x_lev
            
            # Proportion of 72-layer grid cell, by pressure, within expanded layer
            __xmat_s[__xct] = (_GEOS_72L_AP[__x_lev_72] - _GEOS_72L_AP[__x_lev_72+1])/__dp_total
    __start_pt = __i_lev

# Do last entry separately (no layer to go with it)
__xmat_72to47 = scipy.sparse.coo_matrix((__xmat_s,(__xmat_i,__xmat_j)),shape=(72,47))

GEOS_47L_grid = vert_grid(_GEOS_47L_AP, _GEOS_47L_BP)

# CAM 26-layer grid
_CAM_26L_AP = np.flip(np.array([ 219.4067,   489.5209,   988.2418,   1805.201,        
                                2983.724,   4462.334,   6160.587,   7851.243,        
                                7731.271,   7590.131,   7424.086,   7228.744,        
                                6998.933,   6728.574,   6410.509,   6036.322,        
                                5596.111,   5078.225,   4468.96,    3752.191,        
                                2908.949,   2084.739,   1334.443,   708.499,         
                                252.136,    0.,         0.  ]),axis=0)*0.01
_CAM_26L_BP = np.flip(np.array([ 0.,         0.,         0.,         0.,             
                                0.,         0.,         0.,         0.,             
                                0.01505309, 0.03276228, 0.05359622, 0.07810627,     
                                0.1069411,  0.14086370, 0.180772,   0.227722,       
                                0.2829562,  0.3479364,  0.4243822,  0.5143168,      
                                0.6201202,  0.7235355,  0.8176768,  0.8962153,      
                                0.9534761,  0.9851122,  1.        ]),axis=0)

CAM_26L_grid = vert_grid(_CAM_26L_AP, _CAM_26L_BP)
