import numpy as np

# Persistent (module) variables
z_int = None
p_int = None

def altitude_to_pressure(z_m):
    '''Lightweight version of atmosisa from the Aerospace Toolbox'''

    # Sort from lowest to highest altitude
    sort_idx = np.argsort(z_m)
    #z_sorted = z_m[sort_idx]
    
    # COESA data
    height_vec = np.array([-0.1,0,11,20,32,47,51,71,84.8520,1e6]) # km
    lapse_vec = np.array([0.0,-6.5,0.0,1.0,2.8,0.0,-2.8,-2.0,0.0]) # K/km
    T_reference = 288.15
    height_delta = np.diff(height_vec)

    # Change in temperature between each level
    T_delta = height_delta*lapse_vec
    T_vec = np.cumsum(np.concatenate([[T_reference],T_delta]))

    # Gas composition
    gas_MW = [28.0134,31.9988,39.948,44.00995,20.183,4.0026,83.80,131.30,16.04303,2.01594]

    gas_frac = [0.78084,0.209476,0.00934,0.000314,0.00001818,0.00000524,0.00000114,0.000000087,0.000002,0.0000005]

    # Normalize, to be 100% safe
    gas_frac = gas_frac/np.sum(gas_frac)
    
    n_vals = z_m.size
    #T_K = np.zeros(n_vals)
    #a_ms = np.zeros(n_vals)
    p_pa = np.zeros(n_vals)
    #rho_kgm3 = np.zeros(n_vals)

    # Temperature at target altitudes
    T_K = np.interp(z_m/1000.0,height_vec,T_vec)

    R_star = 8.31432
    N_avog = 6.022169e23 # molec/mol

    #dynVisc = (T_K.^1.5 .* 1.458e-6)./(T_K + 110.4);
    #kinVisc = dynVisc./rho_kgm3;

    g0 = 9.80665 # m/s2

    iLo = 0
    iHi = 1
    zLo = height_vec[0] * 1000.0
    zHi = height_vec[1] * 1000.0
    MgR = sum(gas_frac*gas_MW)*1e-3*g0/R_star
    TLo = T_vec[iLo]
    alphaTemp = 0
    # Exponential offset
    P_base = 101325 * np.exp(-MgR*zLo/TLo)
    for iPoint in range(T_K.size):
        i_sort = sort_idx[iPoint]
        zCurr = z_m[i_sort]
        while zCurr > zHi:
            if np.abs(alphaTemp) > 0:
                PNew = P_base * np.power(T_vec[iHi]/T_vec[iLo],MgR/-alphaTemp)
            else:
                PNew = P_base * np.exp(MgR*(zLo-zHi)/TLo)
            
            #fprintf('%5.2f km, %5.2f K, %9.2f -> %9.2f hPa, %8.5f K/m,\n',HVec(iLo),TVec(iLo),PBase./100,PNew./100,alphaTemp);
            P_base = PNew
            iLo    = iHi
            iHi   += 1
            zLo    = zHi
            zHi    = height_vec[iHi] * 1000
            TLo    = T_vec[iLo]
            alphaTemp = lapse_vec[iLo] / 1000
            
        if np.abs(alphaTemp) > 0:
            p_pa[i_sort] = P_base * np.power(T_K[i_sort]/TLo,MgR/-alphaTemp)
        else:
            p_pa[i_sort] = P_base * np.exp(MgR*(zLo-zCurr)/TLo)
    
    return p_pa

def pressure_to_altitude(p_pa):
    '''ATMOSPALT Lightweight version of atmospalt from the MATLAB Aerospace Toolbox
    Returns the COESA estimate of altitude (in m) for a pressure (in Pa)'''
    # Use the module-level variables
    global z_int, p_int

    # Generate interpolation vecor
    if z_int is None or p_int is None:
        z_int = np.arange(82e3,-1e3,-1.0)
        p_int = altitude_to_pressure(z_int)

    z_m = np.interp(np.array(p_pa),p_int,z_int,np.inf,z_int[-1])
    return z_m
