# **DensityFluxTool: A Kinematic Framework for Water Mass Transformation**

## **📌 Description**  
`FluxTool` is a computational tool for estimating oceanic **density flux** using a **kinematic approach**, as outlined in [Piracha et al. (2023)](https://doi.org/10.3389/fmars.2023.1020153). Unlike traditional thermodynamic methods relying on heat and freshwater fluxes, this tool leverages **satellite-derived material derivatives** of **Sea Surface Temperature (SST)** and **Sea Surface Salinity (SSS)** to compute density flux at the ocean surface.  

Additionally, the tool provides:  
- Material derivatives of **SST** and **SSS**  
- Their respective products with **Mixed Layer Depth (MLD)**  
- An analysis of the role of **freshwater fluxes** on **deep water formation** in the **Nordic Seas**  

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

     where \alpha and \beta are the thermal and haline expantion and contraction coefficients, 
     respectively.

     the role of \alpha and \beta are to quantify how density changes with respect to temperature and 
     salinity

4. **Analyze Freshwater’s Role in Deep Water Formation**  
   - Freshwater forcing impacts buoyancy and water mass stability in the **Nordic Seas**.  
   - This tool enables **regional analysis** of the influence of evaporation, precipitation, and ice melt.  

---

## **🛠️ Installation**
### **Dependencies**

You must have the following packages installed:

- xarray: `pip install xarray` -> for all things related to netcdf 
- numpy: `pip install numpy` -> for numeric operations
- gsw: `pip install gsw` -> Thermodynamic equation of state for seawater related
  functions

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

a **keywords dictionary** is central to the functionality of the tool. It is defined in the 
following way:
`
kwargs = {
        'sss': 'name of the salinity variable in the provided netCDF',
        'sst': 'name of the temperature variable in the provided netCDF',
        'lat': 'name of the latitude dimension in the provided netCDF',
        'lon': 'name of the longitude dimension in the provided netCDF', 
        'time': 'name of the time dimension in the provided netCDF',
        'time_resolution': 'time resolution of netCDF dataset', 
        'spatial_resolution': 'spatial resolution of netCDF dataset',
        'to_netcdf': True|False (to additionally save output as a netCDF file named fluxes.nc),
        'u': 'name of the eastward component of velocity variable in the provided netCDF',
        'v': 'name of the northward component of velocity variable in the provided netCDF',
        'mld': 'name of the mixed layer depth variable in the provided netCDF'}
`

```python
import xarray as xr
From FluxTool import calculate_all_fluxes

# Load dataset
'''
make sure that your provided dataset has the following variables
    - sea surface salinity/temperature
    - sea surface currrents (the eastward and northward components)
    - mixed layer depth 

'''
ds = open_dataset(PATH TO YOUR NETCDF FILE)

# define a key word argument dictionary
# see the preceeding section for how to define the dictionary

# Compute density flux
fluxes = calculate_all_fluxes(ds, kwargs)
```

The `fluxes` output is a xarray Dataset object with the same dimension as the provided netCDF.
**!The time dimension of the output will be 1 less then the input (this is due to the fact that the 
fluxes are computed from via derivatives)**

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

## **📜 Citation**
If you use this tool in your research, please cite:  
> Piracha, A., Olmedo, E., Turiel, A., Portabella, M., & González-Haro, C. (2023).  
> Using satellite observations of ocean variables to improve estimates of water mass (trans)formation.  
> *Frontiers in Marine Science, 10*, 1020153. [https://doi.org/10.3389/fmars.2023.1020153](https://doi.org/10.3389/fmars.2023.1020153)  

---

## **📬 Contact**
For questions or collaboration opportunities, reach out to:  
📧 **Aqeel Piracha** - [piracha.aqeel1@gmail.com](mailto:piracha.aqeel1@gmail.com)  

# To-Do

- [ ] define simple_dflux.py as a class (with the related functions being
  recoded as methods). Thus, taking advantage of pythons OOP functionality.
- [ ] update this README to describe density flux, what it is, how it's
  calculated, it's importance.
- [ ] complete the preprocessor (preprocessor.py), which will eventually be a
  scrippt to take all the necessary input variable from disparate netCDF files
  and map them to a unified space and time grid to then be passed to the tool
  for calculating outputs. 
