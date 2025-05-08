import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import gsw
import preprocessor as pp
from pyFlux import pyFlux as pf

# 1. preprocessing data
# defining a data dictionary of variable names as keys and file paths as items.
data_dict = {
    'so': '/Users/ap/Documents/.tools/cmems_mod_arc_phy_my_topaz4_P1D-m_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00m_2017-01-01-2017-12-31.nc',
    'thetao': '/Users/ap/Documents/.tools/cmems_mod_arc_phy_my_topaz4_P1D-m_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00m_2017-01-01-2017-12-31.nc',
    'vxo': '/Users/ap/Documents/.tools/cmems_mod_arc_phy_my_topaz4_P1D-m_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00m_2017-01-01-2017-12-31.nc',
    'vyo': '/Users/ap/Documents/.tools/cmems_mod_arc_phy_my_topaz4_P1D-m_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00m_2017-01-01-2017-12-31.nc',
    'mlotst': '/Users/ap/Documents/.tools/cmems_mod_arc_phy_my_topaz4_P1D-m_multi-vars_180.00W-179.88E_50.00N-90.00N_0.00m_2017-01-01-2017-12-31.nc',
}
# now making the pyFlux Object and calculating fluxes
kwargs = {'sss': 'so',
          'sst': 'thetao',
          'u': 'vxo',
          'v': 'vyo',
          'mld': 'mlotst',
          'ssd': 'ssd',
          'time': 'time', 
          'lat': 'y', 
          'lon': 'x',
          'ease': True,
          'ease_res_km': 25,
          'calculate_ssd': True,
          'to_netcdf': False,
          'isglobal': False,
          'extent': [-70, 10, 50, 80]}
# loading and reinterpolating the data
datasets = pp.load_data_variables(data_dict, times=range(10))
target_grid = xr.open_dataset("ease25_grid.nc").isel({'time': 0})
ds = pp.batch_regrid_all(datasets, target_grid_ds=target_grid, **kwargs)
# then making the pyflux Object
p = pf(ds, **kwargs)
# 2. calculating fluxes
fluxes = p.calculate_all_fluxes(**kwargs)
