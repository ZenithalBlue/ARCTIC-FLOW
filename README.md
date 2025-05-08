# **pyFlux: A Kinematic Framework for Water Mass Transformation**

## **📌 Description**  
`pyFlux` is a computational tool for estimating oceanic **density flux** using a **kinematic approach**, as outlined in [Piracha et al. (2023)](https://doi.org/10.3389/fmars.2023.1020153). Unlike traditional thermodynamic methods relying on heat and freshwater fluxes, this tool leverages **satellite-derived material derivatives** of **Sea Surface Temperature (SST)** and **Sea Surface Salinity (SSS)** to compute density flux at the ocean surface.  

Additionally, the tool provides:  
- Material derivatives of **SSS** **SSD** (where, SSS and SSD refer to the Sea Surface Salinity and 
Density, respectively)
- Their respective products with **Mixed Layer Depth (MLD)**  
- An analysis of the role of **freshwater fluxes** on **deep water formation** in the **Nordic Seas**  

Essentially, pyFlux acts as a wrapper around an existing xarray dataset, allowing users to compute density fluxes and their components with ease. 
Also, it allows for a better calculation of derivatives and the material derivative (which is central to the approach employed). 
The tool is designed to be user-friendly, making it accessible for researchers and practitioners in oceanography and related fields.

---

## **📖 Methodology**
This tool follows the **kinematic framework** described in **Piracha et al. (2023)**:  

1. **Compute the Material Derivative of Surface Density**  
   ```math
   \frac{D \rho}{Dt} = \frac{\partial \rho}{\partial t} + u \frac{\partial \rho}{\partial x} + v \frac{\partial \rho}{\partial y}
   ```  
   where \( u, v \) are near-surface velocities.

2. **Multiply by Mixed Layer Depth (MLD)**  
   ```math
   \text{Density Flux} = \left( \frac{D \rho}{Dt} \right) \times \text{MLD}
   ```  
   - This step accounts for entrainment and subduction effects.  
   - The result is expressed in **kg/m²/s**.  

3. **Compute Thermal & Haline Contributions**  
   - The density flux is decomposed into **thermal** and **haline** components using **temperature and salinity material derivatives**:
     ```math
     -\rho\alpha\frac{D T}{Dt}, \quad \rho\beta\frac{D S}{Dt}
     ```  
   where \( \alpha, \beta, \rho \) are the thermal expansion, haline contraction coefficients, and 
   density, respectively.

4. **Analyze Freshwater’s Role in Deep Water Formation**  
   - Freshwater forcing impacts buoyancy and water mass stability in the **Nordic Seas**.  
   - This tool enables **regional analysis** of the influence of evaporation, precipitation, and ice melt.  

---

## **🛠️ Installation**
### **Dependencies**

You must have the following packages installed:

- xarray: `pip install xarray` -> for all things related to netCDF 
- netCDF4: `pip install netCDF4` -> for working with netCDF4 files
- numpy: `pip install numpy` -> for numeric operations
- gsw: `pip install gsw` -> Thermodynamic equation of state for seawater related functions
- cartopy: `pip install cartopy` -> for geospatial plotting  
- matplotlib: `pip install matplotlib` -> for plotting

### **Installation Steps**
```sh
git clone https://github.com/ZenithalBlue/ARCTIC-FLOW.git
cd ARCTIC-FLOW
```

---

## **🚀 Usage**
### **Basic Example**
**note: if your data has a depth dimension it should be flattened before computing
the fluxes:
you can do this by running: 

 - `ds.isel({'depth': 0})`**

where, 'depth' is the name of the depth dimension in the xarray dataset object (ds)

```python
import xarray as xr
import preprocessor as pp
from pyFlux import pyFlux as pf

# 1. define data dictionary (see next section)
# 2. preprocess the data to a common ease grid
datasets = pp.load_data_variables(data_dict, times=range(10))
target_grid = xr.open_dataset("ease25_grid.nc").isel({'time': 0})
ds = pp.batch_regrid_all(datasets, target_grid_ds=target_grid, **kwargs)
# 3. define kwargs dictionary (see next section)
# 4. then make the pyflux object
#  ds is an xarray dataset with all necessary variable (typically output from the preprocessing step above)
p = pf(ds, **kwargs)
# 5. calculating fluxes
fluxes = p.calculate_all_fluxes(**kwargs)
# fluxes will be a pyFlux object with an xarray datasets (ds) as one of its members containing all the fluxes
```

The `fluxes` output is itself a pyFlux object and so you can call all the methods defined in the pyFlux class on it. 
**!The time dimension of the output will be 1 less then the input (this is due to the fact that the 
fluxes are computed via derivatives)**

### **Defining the data dictionary**

to use the `preprocessor` you need to define a data dictionary, which can be done in the following way:

|          key         | value                                                  |
|:---------------------|:-------------------------------------------------------|
| name of sss variable | path to netCDF file containing sss data                |
| name of sst variable | path to netCDF file containing sst data                |
| name of u variable   | path to netCDF file containing eastward velocity data  |
| name of v variable   | path to netCDF file containing northward velocity data |
| name of mld variable | path to netCDF file containing mixed layer depth data  |

**Even if multiple variables are stored in the same file, you still need to specify each variable separately in the dictionary with the same file path.**

### **Defining the keyword arguments dictionary**

a **keywords dictionary** is central to the functionality of the pyFlux class. It is defined in the 
following way:

|          key          | value                                                  |
|:--------------------- |:-------------------------------------------------------|
| sss                   |name of the salinity variable in the provided netCDF    |
| sst                   |name of the temperature variable in the provided netCDF |
| u                     |name of the eastward velocity n the provided netCDF     |
| v                     |name of the nortward velocity variable in the provided netCDF  |
| mld                   |name of the mld variable in the provided netCDF         |
| lat                   |name of the latitude dimension in the provided netCDF   |
| lon                   |name of the latitude dimension in the provided netCDF   |
| time                  |name of the time dimension in the provided netCDF       |
| time_resolution       |time resolution of the provided data: `D`: daily, `W`: weekly, `M`:monthly|
| calculate_ssd         |boolean of whether to calculate sea surface density     |
| ease                  |boolean of whether the data is on an ease Grid          |
| ease_res              |spatial resolution of ease Grid [km]                    |
| to_netcdf             |boolean of whether to save output to a `fluxes.nc` file |


as well as this there are a lot of additional optional keyword arguments that can be passed to the pyFlux class.
these include:

these are limited to the `map_plot` method of the `pyFlux class`

|          key          | value                                                  |
|:--------------------- |:-------------------------------------------------------|
| cmap                  | colormap name: Default `turbo`                         |
| cbar                  | boolean of whether to draw colorbar: Default `True`    |
| title                 | title of plot: Default blank                           |
| savefig               | boolean of whether to savefig: Default `False`         |
| imname                | if `savefig` then defines name of saved plot: Default blank           |
| isglobal              | boolean of whether to plot a global map: Default `true`|
| extent                | if not `isglobal` then defines region to plot: `[lon_min, lon_max, lat_min, lat_max]`|
| cbar_label            | if `cbar` then defines colorbar title: Default blank   |

### **A test case**

The Near real-time monthly **ARMOR3D** data is a good **starting point**
to assess the functionality and performance of the tool. The data can be found
[here](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012/download?dataset=dataset-armor-3d-nrt-monthly_202012)
For the **variables** choose the following:

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

- `nc_name = [the filename of the example data you just downloaded]`
the best way to **run the tool** is to:

- cd into the cloned repository directory, 
- run ipython 
- and type `run test.py`

---

## **📊 Outputs**
- **Net Density Flux** (`kg/m²/s`)  
- **Thermal & Haline Contributions**  
- **Material Derivatives of SSS and SSD**  
- **freshwater fluxes**

---

## list of all functions and usage

### Pre-processor


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


## **📜 Citation**
If you use this tool in your research, please cite:  
> Piracha, A., Olmedo, E., Turiel, A., Portabella, M., & González-Haro, C. (2023).  
> Using satellite observations of ocean variables to improve estimates of water mass (trans)formation.  
> *Frontiers in Marine Science, 10*, 1020153. [https://doi.org/10.3389/fmars.2023.1020153](https://doi.org/10.3389/fmars.2023.1020153)  

---

## **📬 Contact**
For questions or collaboration opportunities, reach out to:  
📧 **Aqeel Piracha** - [piracha.aqeel1@gmail.com](mailto:piracha.aqeel1@gmail.com)  
