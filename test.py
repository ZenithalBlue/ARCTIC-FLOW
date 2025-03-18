from FluxTool import calculate_all_fluxes
import xarray as xr

nc_name = './cmems_mod_arc_phy_anfc_6km_detided_P1D-m_1742301711644.nc'
ds = xr.open_dataset(nc_name, engine='netcdf4')

ds = ds.isel({'depth': 0})

kwargs = {
        'sss': 'so',
        'sst': 'thetao',
        'lat': 'latitude', 'lon': 'longitude', 'time': 'time',
        'time_resolution': 'D', 
        'spatial_resolution': 'DEG',
        'scalar': '',
        'to_netcdf': True,
        'u': 'vxo', 'v': 'vyo',
        'mld': 'mlotst'}

ds = calculate_all_fluxes(ds, kwargs)


