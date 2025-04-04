import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from pyFlux import pyFlux as pf

# Load data
ds = xr.open_dataset(\
        './dataset-armor-3d-rep-monthly_multi-vars_179.88W-179.88E_82.12S-89.88N_0.00m_1993-01-01-2021-12-01.nc')
ds = ds.isel({'depth': 0})


# Create an example keyword argument dictionary
kwargs = {'sss': 'so',
          'sst': 'to',
          'u': 'ugo',
          'v': 'vgo',
          'mld': 'mlotst',
          'time': 'time', 
          'lat': 'latitude', 
          'lon': 'longitude',
          'time_resolution': 'M',
          'to_netcdf': False}
## Create an example keyword argument dictionary for satellites
# Calculate sea surface density and adding it to the dataset
ssd = gsw.rho(ds[kwargs['sss']].data, ds[kwargs['sst']].data, 0)
ds['ssd'] = ([kwargs['time'], kwargs['lat'], kwargs['lon']], ssd)
# Create a pyFlux object
p = pf(ds, **kwargs)
# Calculate the fluxes
fluxes = p.calculate_all_fluxes(**kwargs)
