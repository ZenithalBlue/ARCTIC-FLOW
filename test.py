import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw
from pyFlux import pyFlux as pf

# Load data
ds = xr.open_dataset('test_data.nc')
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
          'to_netcdf': True}

# Calculate sea surface density and adding it to the dataset
ssd = gsw.rho(ds[kwargs['sss']].data, ds[kwargs['sst']].data, 0)
ds['ssd'] = ([kwargs['time'], kwargs['lat'], kwargs['lon']], ssd)

# Create a pyFlux object
p = pf(ds)

# Calculate the fluxes
fluxes = p.calculate_all_fluxes(**kwargs)

