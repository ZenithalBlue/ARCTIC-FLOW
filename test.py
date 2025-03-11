from simple_dflux import calculate_all_fluxes
import xarray as xr

nc_name = '../test_data.nc'

ds = xr.open_dataset(nc_name)
#NOTE i think including the depth dimension as 0 might be a good thing (we explicitly state 
# that the output is relative to the first depth [it might be 5 meters or surface])
ds = ds.isel({'depth': 0})
#ds = ds.coarsen(latitude=6,longitude=1, boundary='trim').mean()

kwargs = {
        'sss': 'so',
        'sst': 'to',
        'lat': 'latitude', 'lon': 'longitude', 'time': 'time',
        'time_resolution': 'M', 
        'spatial_resolution': 'DEG',
        'scalar': '',
        'to_netcdf': True,
        'u': 'ugo', 'v': 'vgo',
        'mld': 'mlotst'}

ds = calculate_all_fluxes(ds, kwargs)


