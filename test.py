from FluxTool import calculate_all_fluxes
import xarray as xr

nc_name = 'test_data.nc'
ds = xr.open_dataset(nc_name, engine='netcdf4')

ds = ds.isel({'depth': 0})

kwargs = {
        'sss': 'so',
        'sst': 'to',
        'lat': 'latitude', 'lon': 'longitude', 'time': 'time',
        'time_resolution': 'M', 
        'spatial_resolution': 'DEG',
        'scalar': '',
        'to_netcdf': True,
        'u': 'vgo', 'v': 'vgo', 'mld': 'mlotst'}
ds = calculate_all_fluxes(ds, kwargs)

