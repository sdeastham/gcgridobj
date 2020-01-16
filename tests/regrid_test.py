import numpy as np
import gcgridobj
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cubedsphere

# Generate test grids
# Cubed sphere grids (C24, C90)
c24 = cubedsphere.csgrid_GMAO(24)
c90 = cubedsphere.csgrid_GMAO(90)
# Lat-lon grids (Generic 2x2, standard 4x5)
ll2x2 = gcgridobj.latlontools.gen_hrz_grid(lon_stride=2,lat_stride=2,half_polar=False,center_180=False)
ll4x5 = gcgridobj.gc_horizontal.gmao_4x5_global

# Lat-lon grid data comes pre-packaged with areas in m2 ("ll2x2['area']), but for cubed sphere we must generate it ourselves
c24_area = cubedsphere.calc_cs_area(c24)
c90_area = cubedsphere.calc_cs_area(c90)

# Regridder objects, all generated through a single helper function
# Lat-lon (2x2) to cubed-sphere (c90)
ro_2x2_c90 = gcgridobj.regrid.gen_regridder(ll2x2,c90)
# Cubed-sphere (c90) to cubed-sphere (c24)
ro_c90_c24 = gcgridobj.regrid.gen_regridder(c90,c24)
# Cubed-sphere (c24) to lat-lon (2x2)
ro_c24_2x2 = gcgridobj.regrid.gen_regridder(c24,ll2x2)
# Lat-lon (2x2) to lat-lon (4x5)
ro_2x2_4x5 = gcgridobj.regrid.gen_regridder(ll2x2,ll4x5)

# Generate test data which has 4 dimensions
# First two are "arbitrary"; last two are spatial
temp_2x2 = np.zeros((3,6,90,180))
template = np.zeros((90,180))
for i_x in range(180):
    lon_val = np.cos(4 * ll2x2['lon'][i_x] * np.pi / 180.0)
    for i_y in range(90):
        lat_val = np.cos(6 * ll2x2['lat'][i_y] * np.pi / 180.0)
        template[i_y,i_x] = 100 + (lon_val * lat_val * 100.0)
for i_a in range(3):
    for i_b in range(6):
        temp_2x2[i_a,i_b,...] = np.cos(i_a*np.pi/9) * np.cos(i_b*np.pi/12) * template

# Use a single regrid function to perform regridding
# 2x2 -> C90
temp_c90 = gcgridobj.regrid.regrid(temp_2x2,ro_2x2_c90)
# C90 -> C24
temp_c24 = gcgridobj.regrid.regrid(temp_c90,ro_c90_c24)
# C24 -> 2x2
redo_2x2 = gcgridobj.regrid.regrid(temp_c24,ro_c24_2x2)
# 2x2 -> 4x5
temp_4x5 = gcgridobj.regrid.regrid(temp_2x2,ro_2x2_4x5)

# Plot results!
f, ax_arr = plt.subplots(3,2,figsize=(8,5),subplot_kw={'projection': ccrs.PlateCarree()},squeeze=False)

i_a = 2
i_b = 4

ps = lambda name,dat,grd : print('{:15s}: {:16.4f} vs {:16.4f}'.format(
    name,np.sum(dat * grd['area'].values),np.sum(temp_2x2[i_a,i_b,...] * ll2x2['area'].values)))
psc = lambda name,dat,grd : print('{:15s}: {:16.4f} vs {:16.4f}'.format(
    name,np.sum(dat * grd),np.sum(temp_2x2[i_a,i_b,...] * ll2x2['area'].values)))

ax = ax_arr[0,0]
ax.set_title('Original 2x2')
curr_data = temp_2x2[i_a,i_b,...]
gcgridobj.plottools.plot_layer(curr_data,hrz_grid=ll2x2,ax=ax)
ps('Original',curr_data,ll2x2)

ax = ax_arr[0,1]
ax.set_title('C90')
curr_data = temp_c90[i_a,i_b,...]
gcgridobj.plottools.plot_layer(curr_data,hrz_grid=c90,ax=ax)
psc('C90',curr_data,c90_area)

ax = ax_arr[1,0]
ax.set_title('C24')
curr_data = temp_c24[i_a,i_b,...]
gcgridobj.plottools.plot_layer(curr_data,hrz_grid=c24,ax=ax)
psc('C24',curr_data,c24_area)

ax = ax_arr[1,1]
ax.set_title('Rebuilt 2x2')
curr_data = redo_2x2[i_a,i_b,...]
gcgridobj.plottools.plot_layer(curr_data,hrz_grid=ll2x2,ax=ax)
ps('Rebuild',curr_data,ll2x2)

ax = ax_arr[2,0]
ax.set_title('4x5')
curr_data = temp_4x5[i_a,i_b,...]
gcgridobj.plottools.plot_layer(curr_data,hrz_grid=ll4x5,ax=ax)
ps('4x5',curr_data,ll4x5)

ax = ax_arr[2,1]
ax.set_title('Rebuilt - original')
im,cb = gcgridobj.plottools.plot_layer(redo_2x2[i_a,i_b,...] - temp_2x2[i_a,i_b,...],hrz_grid=ll2x2,ax=ax)
cl = np.max(np.abs(im.get_clim()))
im.set_clim([-cl,cl])
im.set_cmap('RdBu_r')
