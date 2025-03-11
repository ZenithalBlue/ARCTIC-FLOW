# ARCTIC-FLOW Repository for the Open Science Code for the surface density and
freshwater flux

# Example data The Near real-time monthly ARMOR3D data is a good starting point
to assess the functionality and performance of the tool. The data can be found
![here](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012/download?dataset=dataset-armor-3d-nrt-monthly_202012)
For the variables choose the following:

- Sea Water Salinity
- Sea Water Temperature
- Ocean Mixed Layer Thickness
- Geostrophic eastward sea water velocity 
- Geostrophic northward sea water velocity 

For visualisation of density flux variability, it is best to choose a full year
of data (e.g. 01/01/2020->12/31/2020) **Make sure to choose only the surface
data (i.e. Depth range from 0m to 0m)** 

You will need to create a user account and save the file in the directory where
you cloned the repository.  With the example data downloaded, you can proceed
to modify the test.py file in the following way:

- nc_name = the filename of the example data you just downloaded

the best way to run the tool is to:

- cd into the cloned repository directory, 
- run ipython 
- and type `run test.py`

# Pre-requisites

You must have the following packages installed:

- xarray: pip install xarray -> for all things related to netcdf 
- numpy: pip install numpy -> for numeric operations
- gsw: pip install gsw -> Thermodynamic equation of state for seawater related
  functions

# To-Do

- [ ] define simple_dflux.py as a class (with the related functions being
  recoded as methods). Thus, taking advantage of pythons OOP functionality.
- [ ] update this README to describe density flux, what it is, how it's
  calculated, it's importance.
- [ ] complete the preprocessor (preprocessor.py), which will eventually be a
  scrippt to take all the necessary input variable from disparate netCDF files
  and map them to a unified space and time grid to then be passed to the tool
  for calculating outputs. 
