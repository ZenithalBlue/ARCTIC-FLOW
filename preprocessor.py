import gsw
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import xarray as xr
import dask
from collections import defaultdict
import numpy as np
from scipy.interpolate import griddata
from pyFlux import pyFlux as pf
import re

plt.ion()


"""
the preprocessor takes a dictionary of variable names and their corresponding NetCDF file paths as 
input. the ultimate goal is to place all the datasets on uniform time and space grids. 
specifically, the space grids will be ease nl 25km, and the time grids will be the coarsest from 
amongst the data.
step 1. use xarray to load the specified variables from the NetCDF files, creating a dictionary 
        of xarray Datasets (using dask for chunking).
"""
def apply_nan_mask(data, mask):
    """
    Masks a DataArray by the NaN locations of another DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        The DataArray to be masked.
    mask : xarray.DataArray

    Returns
    -------
    xarray.DataArray
        The masked DataArray, where NaN values in the mask are applied to the data.
    """
    return data.where(~np.isnan(mask.values))

def regrid_to_ease25(source_ds, target_grid_ds, varname, datasets, method='linear'):
    """
    Regrids a 2D or 3D variable (time, y, x) to a 25 km EASE grid.

    This function performs horizontal interpolation of geophysical data (e.g., SST, SSS) 
    from a regular latitude-longitude grid to a target EASE grid defined by 2D latitude 
    and longitude coordinates. If the source dataset is already on a 2D projected grid, 
    regridding is skipped and the original variable is returned as-is.

    Parameters
    ----------
    source_ds : xarray.Dataset
        The dataset containing the variable to be interpolated. Should include 1D 'lat'/'lon' or 'latitude'/'longitude'.
    
    target_grid_ds : xarray.Dataset
        The dataset containing the 2D EASE grid coordinates to which the variable will be regridded.
        This must include a variable (e.g., 'sss') used to mask the result after interpolation.
    
    varname : str
        Name of the variable in `source_ds` to interpolate.
    
    datasets : dict
        Dictionary containing the latitude and longitude from the source dataset.
        Example: {'lat': source_ds['lat'], 'lon': source_ds['lon']}

    method : str, optional
        Interpolation method to use. Options are:
        - 'linear' (default)
        - 'nearest'
        - 'cubic'
        Method is passed to `xarray.DataArray.interp`.

    Returns
    -------
    xarray.DataArray
        The interpolated variable regridded to the EASE grid if lat/lon are 1D,
        or the original variable if the coordinates are already 2D.

    Notes
    -----
    - The function checks if the source coordinates are 2D; if so, interpolation is skipped.
    - Regridding uses `xarray`'s native interpolation routines.
    - If the data is 3D (e.g., time, lat, lon), interpolation is applied slice-by-slice over time.
    - The function uses a mask (from 'sss' in the target grid) to clean up edge effects after regridding.

    Output
    ------
    Example output:
    Regridding started...
    Regridding sst using xarray interp (regular lat/lon)...
    sst Regridding Progress: 100%|██████████| ...
    Regridding complete.
    """
    print('Regridding started...')

    def get_coord_name(ds, options):
        for name in options:
            if name in ds.coords:
                return name
        raise KeyError(f"None of {options} found in dataset coordinates.")

    lat_name = get_coord_name(source_ds, ['lat', 'latitude'])
    lon_name = get_coord_name(source_ds, ['lon', 'longitude'])

    tgt_lat_name = get_coord_name(target_grid_ds, ['lat', 'latitude'])
    tgt_lon_name = get_coord_name(target_grid_ds, ['lon', 'longitude'])

    tgt_lat = target_grid_ds[tgt_lat_name]
    tgt_lon = target_grid_ds[tgt_lon_name]

    src_lat = source_ds[lat_name].values
    src_lon = source_ds[lon_name].values
    lat_is_2d = src_lat.ndim == 2
    lon_is_2d = src_lon.ndim == 2

    src_data = source_ds[varname]

    if lat_is_2d and lon_is_2d:
        print(f"2D lat/lon grid detected — skipping interpolation and returning dataset unchanged for '{varname}'")
        result = src_data
        return result

    # Regular 1D lat/lon interpolation
    print(f"Regridding {varname} using xarray interp (regular lat/lon)...")
    if src_data.ndim == 2:
        result = src_data.interp({lat_name: tgt_lat, lon_name: tgt_lon}, method=method)
        result = apply_nan_mask(result, target_grid_ds['sss'])
    elif src_data.ndim == 3:
        times = src_data.coords['time']
        interpolated_list = []
        for t in tqdm(range(len(times)), desc=f"{varname} Regridding Progress"):
            slice_2d = src_data.isel(time=t)
            interp_slice = slice_2d.interp({lat_name: tgt_lat, lon_name: tgt_lon}, method=method)
            interp_slice = interp_slice.expand_dims(time=[times[t].values])
            interpolated_list.append(interp_slice)
        result = xr.concat(interpolated_list, dim='time')
        result = apply_nan_mask(result, target_grid_ds['sss'])
    else:
        raise ValueError("Variable must be 2D or 3D.")

    print("Regridding complete.")
    return result

def detect_spatial_resolution(ds, name=None):
    """
    Detects and prints the spatial resolution of a dataset.

    This function estimates the approximate spatial resolution of an xarray Dataset 
    by analyzing the differences between adjacent latitude and longitude values. 
    It supports both 1D (curvilinear) and 2D (structured grid) coordinate systems. 
    Additionally, it detects whether the dataset uses an EASE or EASE2 grid, and if so, 
    attempts to extract and display the nominal grid resolution from the dataset's 
    global attributes.

    Parameters
    ----------
    ds : xarray.Dataset
        The input dataset containing geospatial coordinates (lat/lon or latitude/longitude).
    name : str, optional
        An optional name for the dataset, used in the printout header.

    Returns
    -------
    dict
        A dictionary containing the latitude and longitude variables:
        {
            "lat": xarray.DataArray,
            "lon": xarray.DataArray
        }

    Notes
    -----
    - The function will automatically look for standard coordinate names: 
      "lat" / "lon" or "latitude" / "longitude".
    - For 2D coordinates, the median difference is taken along both dimensions.
    - If the dataset contains a global attribute named "Grid" and it includes 
      the substring "EASE", the function will identify it as an EASE grid.
    - Attempts to extract a numeric resolution (in km) from the grid metadata 
      using regex.

    Output
    ------
    Exanple output:
    Dataset:
      Approximate spatial resolution:
        ∆lat ≈ 0.0417°, ∆lon ≈ 0.0417°
      Grid type detected: EASE grid
      EASE grid resolution: 25 km
    """
    name = name or "Dataset"
    print(f"\n{name}:")
    
    # Extract lat/lon
    try:
        lat = ds['latitude']
        lon = ds['longitude']
    except KeyError:
        lat = ds['lat']
        lon = ds['lon']

    # Check if it's 2D or 1D
    if lat.ndim == 2:
        dlat = np.nanmedian(np.abs(lat[1:, :] - lat[:-1, :]))
        dlon = np.nanmedian(np.abs(lon[:, 1:] - lon[:, :-1]))
    else:
        dlat = np.nanmedian(np.abs(np.diff(lat)))
        dlon = np.nanmedian(np.abs(np.diff(lon)))

    print(f"  Approximate spatial resolution:")
    print(f"    ∆lat ≈ {dlat:.4f}°, ∆lon ≈ {dlon:.4f}°")

    # Check for 'Grid' global attribute
    grid_attr = ds.attrs.get("Grid", "").lower()
    if "ease" in grid_attr:
        print("  Grid type detected: EASE grid")

        # Try to extract grid spacing (e.g., "25km", "12.5km")
        import re
        match = re.search(r"(\d+(\.\d+)?)\s*km", grid_attr)
        if match:
            print(f"  EASE grid resolution: {match.group(1)} km")

    return {
        "lat": lat,
        "lon": lon
    }

def batch_regrid_all(datasets, target_grid_ds, method='linear', **kwargs): 
    """
    Batch regridding utility for multiple variables to the 25 km EASE grid.

    Iterates over a dictionary of xarray Datasets and regrids each variable using `regrid_to_ease25`. 
    Appends regridded variables into a new Dataset structured on the EASE25 grid coordinates.

    Parameters:
    - datasets (dict): Dictionary of {variable_name: xarray.Dataset} with each Dataset containing that variable.
    - target_grid_ds (xarray.Dataset): Dataset containing the target EASE grid lat/lon (2D) and x/y coordinates.
    - method (str): Interpolation method to use. Options: 'linear', 'nearest', 'cubic'. Default is 'linear'.
    - **kwargs: Must include `'sss'`, the key corresponding to a reference Dataset with time dimension.

    Returns:
    - xarray.Dataset: Combined Dataset with each variable regridded and aligned to the EASE25 grid.
    """
    # Start from a copy of the target EASE grid
    combined_ds = xr.Dataset(coords={
        "x": target_grid_ds["x"].values,
        "y": target_grid_ds["y"].values,
        "lat": (("y", "x"), target_grid_ds["lat"].values),
        "lon": (("y", "x"), target_grid_ds["lon"].values),
        "time": datasets[kwargs['sss']]["time"],  # real time
    })

    for name, ds in datasets.items():
        print("=" * 40)
        info = detect_spatial_resolution(ds, name=name)
        print("  Please wait: Regridding to EASE 25 km...\n")

        regridded = regrid_to_ease25(
            ds,
            target_grid_ds,
            name,
            datasets,  # which includes 'lat' and 'lon'
            method=method
        )

        # Add the regridded data as a new variable inside combined_ds
        combined_ds[name] = (["time", "y", "x"], regridded.values)

    return combined_ds

def load_data_variables(data_dict, times=0):
    """
    Lazily load specified variables from NetCDF files into xarray Datasets using dask.

    For each variable in the dictionary, this function opens the NetCDF file with dask,
    selects the variable (and optionally the surface layer), and converts units from
    Kelvin to Celsius if detected.

    Parameters:
    - data_dict (dict): Mapping of variable name (str) to NetCDF file path (str).
    - times (int): Index along the time dimension to slice. Defaults to 0.

    Returns:
    - datasets (dict): Dictionary of variable name -> xarray.Dataset with just that variable.
    """
    datasets = {}
    total = len(data_dict)
    
    print("Initializing lazy load of variables...\n")

    for idx, (var_name, file_path) in enumerate(data_dict.items()):
        try:
            ds = xr.open_dataset(file_path, chunks={})
            
            # Attempt to select the top level if a vertical dimension exists
            for depth_dim in ['depth', 'lev', 'depthu', 'depthv']:
                if depth_dim in ds.dims:
                    ds = ds.isel({depth_dim: 0})
                    break

            if var_name in ds:
                # Select the variable
                selected_var = ds[[var_name]].isel({'time': times})

                # --- Check and Convert Kelvin to Celsius if needed ---
                var_attrs = selected_var[var_name].attrs
                units_attr = var_attrs.get('units', '').lower()
                if 'kelvin' in units_attr or units_attr.strip() == 'k':
                    print(f"\nDetected Kelvin units for {var_name}. Converting to Celsius...")
                    selected_var[var_name] = selected_var[var_name] - 273.15
                    selected_var[var_name].attrs['units'] = 'degrees Celsius'
                    selected_var[var_name].attrs['note'] = 'Converted from Kelvin to Celsius automatically.'

                datasets[var_name] = selected_var

            else:
                raise KeyError(f"Variable '{var_name}' not found in {os.path.basename(file_path)}")

            progress = int(100 * (idx + 1) / total)
            sys.stdout.write(f"\rLoading {var_name:<15} [{progress}%]")
            sys.stdout.flush()

        except Exception as e:
            raise RuntimeError(f"\nError while loading '{var_name}' from '{file_path}': {e}")

    print("\nAll variables loaded into memory (lazily).")
    return datasets

