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

a **keywords dictionary** is central to the functionality of the tool. It is defined in the 
following way:
```python
kwargs = {
        'sss': 'name of the salinity variable in the provided netCDF', 
        'sst': 'name of the temperature variable in the provided netCDF',
        'lat': 'name of the latitude dimension in the provided netCDF',
        'lon': 'name of the longitude dimension in the provided netCDF', 
        'time': 'name of the time dimension in the provided netCDF',
        'time_resolution': 'time resolution of netCDF dataset', 
        'to_netcdf': True|False (to additionally save output as a netCDF file named fluxes.nc),
        'u': 'name of the eastward component of velocity variable in the provided netCDF',
        'v': 'name of the northward component of velocity variable in the provided netCDF',
        'mld': 'name of the mixed layer depth variable in the provided netCDF'}
```

```python
import xarray as xr
From pyFlux import pyFlux as pf

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

## Create an example keyword argument dictionary for satellites
# Calculate sea surface density and add it to the dataset (later versions of the tool will do this automatically)
ssd = gsw.rho(ds[kwargs['sss']].data, ds[kwargs['sst']].data, 0)
ds['ssd'] = ([kwargs['time'], kwargs['lat'], kwargs['lon']], ssd)
# Create a pyFlux object
p = pf(ds, **kwargs)
# Calculate the fluxes
fluxes = p.calculate_all_fluxes(**kwargs)
```

The `fluxes` output is itself a pyFlux object and so you can call all the methods defined in the pyFlux class on it. 
**!The time dimension of the output will be 1 less then the input (this is due to the fact that the 
fluxes are computed via derivatives)**

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
