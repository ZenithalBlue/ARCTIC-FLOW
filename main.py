from simple_dflux import calculate_all_fluxes
import xarray as xr

#nc_name = '/Users/ap/Downloads/cmems_mod_bal_phy_my_P1D-m_multi-vars_9.04E-30.21E_53.01N-65.89N_0.50m_2023-01-01-2023-12-31.nc'
#nc_name = '/Users/ap/Downloads/cmems_mod_arc_phy_my_topaz4_P1D-m_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00m_2014-01-01-2014-12-31.nc'
#nc_name = '/Users/ap/Downloads/dataset-armor-3d-nrt-weekly_multi-vars_179.88W-179.88E_82.12S-89.88N_0.00m_2020-01-01-2020-12-30.nc'
nc_name = '/Users/ap/Downloads/dataset-armor-3d-nrt-monthly_1741160246984.nc'

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


