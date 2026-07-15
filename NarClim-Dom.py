import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
narclim = "/data/ajay/MHW-THW//pre_run/sst_clim/orog_AUS-18_Earth-Consortium-EC-Earth3-Veg_ssp126_r1i1p1f1_NSW-DPE-WRF412C1_v1_fx.nc"
geo_em = "/data/ajay/dom_sel/geo_em.d02.nc"

nds1 = xr.open_dataset(narclim)
nds2 = xr.open_dataset(geo_em)

# Define rotated pole coordinates
pole_latitude = 60.31
pole_longitude1 = 141.38  # For NARCLIM plot
pole_longitude2 = 147.63  # For WRF LANDMASK plot

rotated_pole1 = ccrs.RotatedPole(pole_latitude=pole_latitude, pole_longitude=pole_longitude1, central_rotated_longitude=180)
rotated_pole2 = ccrs.RotatedPole(pole_latitude=pole_latitude, pole_longitude=pole_longitude2, central_rotated_longitude=180)

# Figure subplots
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.37, 1], figure=fig, wspace=0.2, bottom=0.01)

#cmap = "gist_earth"
#cmap = "GnBu"

ncolors = 256
blend_n = 20  # Number of colors to blend from white

colors = plt.get_cmap("terrain")

## Create a gradient at the beginning: blend the first blend_n colors from white to the first non-white color

first_nonwhite = np.array(colors(blend_n / ncolors))  # sample the color map
color_list = []

for i in range(blend_n):
    t = i / (blend_n - 1)
    blended = (1 - t) * np.array([0.8, 0.8, 0.8, 1]) + t * first_nonwhite
    color_list.append(tuple(blended))

# Add the rest of the original colormap after the blended section
for i in range(blend_n, ncolors):
    color_list.append(colors(i / ncolors))
cmap = LinearSegmentedColormap.from_list("terrain_modified", color_list, N=ncolors)



vmin, vmax = 0, 800

# First subplot (NARCLIM) 
ax1 = fig.add_subplot(gs[0], projection=rotated_pole1)
var1 = nds1.orog
lons1 = np.mod(nds1.lon, 360)
lats1 = nds1.lat

cont_plot1 = ax1.pcolormesh(lons1, lats1, var1, shading='auto', cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, edgecolors='face', antialiased=False)
#levels = np.linspace(vmin, vmax, 100)

#cont_plot1=ax1.contourf(lons1, lats1, var1, levels=levels, cmap=cmap,
ax1.set_title("")
ax1.coastlines(linewidth=1.2)
ax1.set_extent([90, 220, -30, 20])
ax1.set_xlim(-37.8, 49.7)
ax1.set_ylim(-23.2, 33.7)

latlon_tick_spacing=10
gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,linestyle='--')
  # turn off top and right lat-lon ticks
gl1.top_labels = False
gl1.right_labels = False
# define the lat-lon tick marks on the x and y axis
# we are just getting the max and min, and having the ticks every x degrees
min_lon = (np.ceil(np.min(lons1))) # ceil rounds to next highest integer
max_lon = (np.floor(np.max(lons1))) # floor ronds to next lowest integer
lon_points = (np.arange(min_lon,max_lon,latlon_tick_spacing))
gl1.xlocator = mticker.FixedLocator(lon_points)
min_lat = (np.ceil(np.min(lats1)))
max_lat = (np.floor(np.max(lats1)))
lat_points = (np.arange(min_lat,max_lat,latlon_tick_spacing))
gl1.ylocator = mticker.FixedLocator(lat_points)

# Define the Box Coordinates (for NARCLIM plot)
box_lon = [103.684, 103.684, 125.608, 125.608, 103.684]
box_lat = [-39.1578, -21.443, -21.443, -39.1578, -39.1578]
ax1.plot(box_lon, box_lat, color="red", linewidth=2, transform=ccrs.PlateCarree(), label="Region of Interest")

# Second subplot (WRF Dom 2)
ax2 = fig.add_subplot(gs[1], projection=rotated_pole2)
var2 = nds2.HGT_M[0, :, :]
lons2 = np.mod(nds2.XLONG_M[0, :, :], 360)
lats2 = nds2.XLAT_M[0, :, :]
#cont_plot2 = ax2.contourf(lons2, lats2, var2, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)
cont_plot1 = ax2.pcolormesh(lons2, lats2, var2, shading='auto', cmap=cmap, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, edgecolors='face', antialiased=False)
ax2.set_title("")
ax2.coastlines(linewidth=1.2)

# Gridlines for second subplot
gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,linestyle='--')
gl2.top_labels = gl2.right_labels = False

tickspace = 4
min_lon, max_lon = np.ceil(np.min(lons2)), np.floor(np.max(lons2))
gl2.xlocator = mticker.FixedLocator(np.arange(min_lon, max_lon, tickspace))
min_lat, max_lat = np.ceil(np.min(lats2)), np.floor(np.max(lats2))
gl2.ylocator = mticker.FixedLocator(np.arange(min_lat, max_lat, tickspace))

ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='left')

#colorbar
cbar = fig.colorbar(cont_plot1, ax=[ax1, ax2], orientation='vertical', fraction=0.04, anchor=(0.4,0.435), shrink=0.4) #pad=0.02)
cbar.ax.set_ylabel("surface altitude(m)")

# Save and show figure
plt.tight_layout()

plt.savefig("Narclim-WRF_orog.jpg", dpi=300, bbox_inches="tight")
plt.show()

