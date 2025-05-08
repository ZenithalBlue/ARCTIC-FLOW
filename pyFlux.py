# A collection of functions for flux calculations
# along with some helper functions for plotting and computing informative statistics
import pandas as pd
import xarray as xr
import numpy as np
import gsw
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
import warnings
import os
import platform
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define constants and parameters
earth_radius = 6371000  # Earth radius in meters
degree_distance = 2 * np.pi * earth_radius / 360  # Distance in meters for 1 degree
month_num = 12
months_day_count = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
months_week_count = [0, 5, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 5]

# Define a class for the pyFlux object

class pyFlux:
    """
    A class for computing flux-related diagnostics in oceanographic datasets.

    Attributes:
        ds (xarray.Dataset): The dataset containing oceanographic variables.
    """

    def __init__(self, ds, **kwargs):
        """
        Initialize the pyFlux class with an xarray Dataset containing oceanographic variables.

        This constructor sets up necessary metadata, computes spatial and temporal resolutions,
        and prepares the dataset for flux and material derivative calculations. Optionally, it 
        can compute and add sea surface density (SSD) to the dataset using salinity and temperature.

        Parameters:
            ds (xarray.Dataset): 
                The input dataset that must contain at least sea surface temperature (SST),
                sea surface salinity (SSS), and optionally mixed layer depth (MLD) and velocity fields.

            kwargs:
                time (str): 
                    Name of the time coordinate in `ds`.
                lat (str): 
                    Name of the latitude (or y) coordinate in `ds`.
                lon (str): 
                    Name of the longitude (or x) coordinate in `ds`.
                sss (str): 
                    Name of the sea surface salinity variable in `ds`. Required if `calculate_ssd=True`.
                sst (str): 
                    Name of the sea surface temperature variable in `ds`. Required if `calculate_ssd=True`.
                calculate_ssd (bool): 
                    If True, compute surface density using TEOS-10 equation of state and add as variable `ssd`. 
                    Default is False.
                ease (bool): 
                    If True, assumes dataset is on a projected EASE grid. Default is False.
                ease_res (float): 
                    EASE grid resolution in kilometers. Only used if `ease=True`. Default is 25.

        Attributes:
            ds (xarray.Dataset): 
                The dataset, with `ssd` added if `calculate_ssd=True`.
            ease (bool): 
                Flag indicating if dataset is on EASE grid.
            ease_res_km (float): 
                EASE grid resolution in kilometers.
            resolution_latitude_meters (float or np.ndarray): 
                Resolution of latitude in meters. Scalar if EASE grid; array otherwise.
            resolution_longitude_meters (float or np.ndarray): 
                Resolution of longitude in meters. Scalar if EASE grid; array with latitude correction otherwise.
            resolution_time (np.ndarray or None): 
                Temporal resolution in units of the time coordinate (e.g., np.timedelta64).
            crs (cartopy.crs.Projection): 
                Coordinate reference system for plotting. Uses North Polar Stereographic for EASE; 
                Robinson projection for geographic.

        Notes:
            - If `calculate_ssd=True`, SSD is computed as in-situ density at 0 dbar using the Gibbs 
              Seawater (GSW) toolbox: rho = gsw.rho(S, T, 0).
            - For geographic grids, spatial resolutions are calculated in meters assuming Earth's radius
              and adjusted for latitude curvature using the cosine of latitude.
        """
        # Extract coordinate names
        time = kwargs['time']
        lat = kwargs['lat']
        lon = kwargs['lon']
        self.ds = ds

# first getting ssd
        if kwargs['calculate_ssd']:
            ssd = np.full(ds[kwargs['sss']].shape, np.nan)
            ssd = gsw.rho(ds[kwargs['sss']].values, ds[kwargs['sst']].values, 0)
            ds['ssd'] = ([kwargs['time'], kwargs['lat'], kwargs['lon']], ssd)

        # Flag for EASE grid
        self.ease = kwargs.get('ease', False)
        self.ease_res_km = kwargs.get('ease_res', 25)

        # Compute spatial resolutions
        if self.ease:
            # EASE grid: resolution is fixed
            res_meters = self.ease_res_km * 1000
            self.resolution_latitude_meters = res_meters
            self.resolution_longitude_meters = res_meters
            self.crs = ccrs.NorthPolarStereo()  # or change if needed
        else:
            # Geographic grid: compute from degrees and convert to meters
            self.resolution_latitude = np.gradient(self.ds[lat].values)
            self.resolution_longitude = np.gradient(self.ds[lon].values)
            self.resolution_latitude_meters = self.resolution_latitude * degree_distance

            # Compute longitude resolution with latitude correction
            meshgrid_lon, meshgrid_lat = np.meshgrid(self.ds[lon].values, self.ds[lat].values)
            self.resolution_longitude_meters = (
                np.gradient(meshgrid_lon)[1] * degree_distance *
                np.cos(np.radians(meshgrid_lat))
            )

            self.crs = ccrs.Robinson()

        # Compute time resolution if available
        if time in self.ds.dims:
            self.resolution_time = np.gradient(self.ds[time].values)
        else:
            self.resolution_time = None

    def compute_derivatives(self, var, timeInd, **kwargs):
        """
        Compute the first-order derivatives of a scalar variable with respect to time,
        latitude, and longitude using forward-time centered-space (FTCS) differencing.

        This method supports both EASE-grid (projected) and geographic (lat/lon) grids.
        The time derivative is scaled appropriately based on the temporal resolution.
        Spatial derivatives are normalized by actual physical distances in meters.

        Parameters
        ----------
        var : str
            Name of the scalar variable in the dataset to differentiate.
        timeInd : int
            Index of the time step to compute the derivatives at.
        **kwargs :
            Additional keyword arguments:
            
            time : str
                Name of the time dimension in the dataset.
            lat : str
                Name of the latitude (or y) dimension in the dataset.
            lon : str
                Name of the longitude (or x) dimension in the dataset.
            time_resolution : {'D', 'W', 'M'}
                Temporal resolution of the dataset. Supported values:
                - 'D': daily
                - 'W': weekly
                - 'M': monthly (requires 'days_in_month' if non-uniform)
            days_in_month : int, optional
                Number of days in the month, only used if time_resolution='M'.
            ease : bool, optional
                If True, assumes a uniform EASE grid with fixed spatial resolution.

        Returns
        -------
        pyFlux
            A new `pyFlux` object initialized with a dataset containing the computed
            derivatives:
                - derivative_time: Time derivative (units: 1/s)
                - derivative_latitude: Meridional spatial derivative (units: 1/m)
                - derivative_longitude: Zonal spatial derivative (units: 1/m)

        Notes
        -----
        - Derivatives are computed using numpy's `np.gradient`, which applies central
          differences to the interior points and first differences to the boundaries.
        - For EASE grid, spatial resolutions are constant. For geographic grids, spatial
          resolutions are computed from coordinate gradients and converted to meters.
        - The time derivative is computed between two time steps and placed at the midpoint
          time, which is used as the single time coordinate in the returned dataset.
        """
        kwargs['calculate_ssd'] = False
        # extract relevant variable names
        time, lat, lon = kwargs['time'], kwargs['lat'], kwargs['lon']

        # select two consecutive time steps
        timeInd_range = [timeInd, timeInd + 1]
        ds = self.ds.isel(time=timeInd_range)

        # Compute spatial and temporal derivatives
        derivative_time, derivative_latitude, derivative_longitude = np.gradient(ds[var])

        # Convert time derivative to appropriate time units
        if kwargs['time_resolution'] == 'D':
            derivative_time = derivative_time[0] / 86400  # Convert to per second
        elif kwargs['time_resolution'] == 'W':
            derivative_time = derivative_time[0] / (7 * 86400)  # Weekly resolution
        elif kwargs['time_resolution'] == 'M':
            derivative_time = derivative_time[0] / (months_day_count[timeInd_range[0] % 12] * 86400)

        # Compute mean derivatives in the spatial domain
        derivative_longitude = np.nanmean(derivative_longitude, axis=0)
        derivative_latitude = np.nanmean(derivative_latitude, axis=0)

        # Normalize spatial derivatives by grid distances
        if kwargs['ease']:
            # EASE grid: use fixed resolution
            derivative_longitude /= self.resolution_longitude_meters
            derivative_latitude /= self.resolution_latitude_meters
        else:
            for lat_index in range(self.ds.sizes[lat]):
                derivative_longitude[lat_index, :] /= (2 * self.resolution_longitude_meters[lat_index, :])
            for lon_index in range(self.ds.sizes[lon]):
                derivative_latitude[:, lon_index] /= (2 * self.resolution_latitude_meters)

        # Compute midpoint time for reference
        ds_diff = self.ds.time.isel(time=timeInd_range[0]) + (self.ds.time.isel(time=timeInd_range[1]) -
                                                              self.ds.time.isel(time=timeInd_range[0])) / 2

        # Store computed derivatives in an xarray dataset
        if kwargs['ease']:
            ds = xr.Dataset({
                "derivative_time": ([lat, lon], derivative_time),
                "derivative_latitude": ([lat, lon], derivative_latitude),
                "derivative_longitude": ([lat, lon], derivative_longitude)
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})
        
            ds = ds.assign_coords(latitude=((lat, lon), self.ds['lat'].values), 
                                  longitude=((lat, lon), self.ds['lon'].values))
        else:
            ds = xr.Dataset({
                "derivative_time": ([lat, lon], derivative_time),
                "derivative_latitude": ([lat, lon], derivative_latitude),
                "derivative_longitude": ([lat, lon], derivative_longitude)
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})

        return pyFlux(ds, **kwargs)

    def material_derivative(self, var, timeInd, **kwargs):
        """
        Compute the material derivative of a scalar field in the ocean.

        The material derivative (also called the total or substantial derivative) quantifies
        how a scalar field (e.g., temperature, salinity, or density) changes following the
        motion of the ocean, accounting for both local (temporal) changes and advection by
        ocean currents.

        Parameters
        ----------
        var : str
            Name of the scalar variable to differentiate (e.g., 'sst', 'sss').
        timeInd : int
            Index of the starting time step at which the material derivative is computed.
        **kwargs :
            Additional keyword arguments:

            u : str
                Name of the eastward velocity component in the dataset.
            v : str
                Name of the northward velocity component in the dataset.
            time : str
                Name of the time dimension.
            lat : str
                Name of the latitude (or y) dimension.
            lon : str
                Name of the longitude (or x) dimension.
            time_resolution : {'D', 'W', 'M'}
                Temporal resolution of the data. Used for scaling the time derivative.
            days_in_month : int, optional
                Number of days in the current month (used if time_resolution='M').
            ease : bool, optional
                Whether the dataset uses a fixed-resolution EASE grid (True) or not (False).

        Returns
        -------
        pyFlux
            A `pyFlux` object initialized with a dataset containing:
            - material_derivative: The total rate of change of the variable (units: 1/s).
            - advective_derivative: The advective component of the material derivative 
              (units: 1/s), i.e., u ∂var/∂x + v ∂var/∂y.

        Notes
        -----
        - This method internally calls `compute_derivatives` to compute spatial and 
          temporal gradients in consistent physical units (per meter, per second).
        - The velocity fields (u, v) are averaged over two time steps before computing advection.
        - The time coordinate for the returned dataset corresponds to the midpoint between the 
          selected time steps, consistent with central differencing schemes.
        - Assumes 2D horizontal fields with dimensions [lat, lon].
        """
        kwargs['calculate_ssd'] = False
        # Extract velocity component names
        u, v = kwargs['u'], kwargs['v']
        time, lat, lon = kwargs['time'], kwargs['lat'], kwargs['lon']

        # Select two time steps
        timeInd_range = [timeInd, timeInd + 1]
        ds = self.ds.isel({'time': timeInd_range})

        # Compute mean velocity components over the two selected time steps
        mean_u = np.nanmean(ds[u].values, axis=0)
        mean_v = np.nanmean(ds[v].values, axis=0)

        # Compute spatial and temporal derivatives
        pyFlux_derivatives = self.compute_derivatives(var, timeInd, **kwargs)

        # Compute the advection term (u * ∂var/∂x + v * ∂var/∂y)
        advective_derivative = (
            pyFlux_derivatives.ds.derivative_longitude * mean_u +
            pyFlux_derivatives.ds.derivative_latitude * mean_v
        )

        # Compute the material derivative: Dvar/Dt = ∂var/∂t + (u ∂var/∂x + v ∂var/∂y)
        material_derivative = pyFlux_derivatives.ds.derivative_time + advective_derivative

        # Assign time coordinate as midpoint between two selected timesteps
        ds_diff = self.ds.time.isel(time=timeInd_range[0]) + (
            self.ds.time.isel(time=timeInd_range[1]) - self.ds.time.isel(time=timeInd_range[0])
        ) / 2

        # Store computed material derivative in an xarray dataset
        if kwargs['ease']:
            ds = xr.Dataset({
                "material_derivative": ([lat, lon], material_derivative.values),
                "advective_derivative": ([lat, lon], advective_derivative.values)
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})
        
            ds = ds.assign_coords(latitude=((lat, lon), self.ds['lat'].values), 
                                  longitude=((lat, lon), self.ds['lon'].values))
        else:
            ds = xr.Dataset({
                "material_derivative": ([lat, lon], material_derivative.values),
                "advective_derivative": ([lat, lon], advective_derivative.values)
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})

        return pyFlux(ds, **kwargs)

    def flux(self, var, timeInd, **kwargs):
        """
        Compute the vertical surface flux of a scalar field in the ocean.

        This method calculates the surface flux as the product of the scalar's material derivative 
        (which includes both local temporal change and horizontal advection) and the mixed layer depth (MLD).
        The result represents the vertical flux of the variable into or out of the ocean surface mixed layer.

        Parameters
        ----------
        var : str
            Name of the scalar variable to compute the flux of (e.g., 'sst', 'sss', 'ssd').
        timeInd : int
            Index of the time step at which to compute the flux.
        **kwargs :
            Additional keyword arguments:

            mld : str
                Name of the mixed layer depth variable in the dataset.
            u : str
                Name of the eastward velocity component.
            v : str
                Name of the northward velocity component.
            time : str
                Name of the time dimension.
            lat : str
                Name of the latitude dimension.
            lon : str
                Name of the longitude dimension.
            time_resolution : {'D', 'W', 'M'}
                Temporal resolution of the data, used to scale time differencing.
            days_in_month : int, optional
                Number of days in the month (if using monthly data).
            ease : bool, optional
                Whether the data grid is a uniform EASE grid (True) or not (False).

        Returns
        -------
        pyFlux object containing the flux of the given variable 

        Notes
        -----
        - The mixed layer depth is averaged across two time steps to align with the central differencing
          used in the material derivative.
        - The time coordinate of the output flux is set to the midpoint between the two time steps.
        - This is a surface-integrated flux per unit area, assuming vertical homogeneity over the mixed layer.
        """
        kwargs['calculate_ssd'] = False
        mld = kwargs['mld']
        time, lat, lon = kwargs['time'], kwargs['lat'], kwargs['lon']

        # Select two time steps
        timeInd_range = [timeInd, timeInd + 1]
        ds = self.ds.isel(time=timeInd_range)

        # Compute the material derivative
        material_derivative = self.material_derivative(var, timeInd, **kwargs)

        # Compute mean mixed layer depth over the two time steps
        mean_mld = np.nanmean(ds[mld].values, axis=0)

        # Compute flux as the product of material derivative and mixed layer depth
        flux = material_derivative.ds.material_derivative * mean_mld

        # Assign time coordinate as midpoint between two selected timesteps
        ds_diff = self.ds.time.isel(time=timeInd_range[0]) + (
            self.ds.time.isel(time=timeInd_range[1]) - self.ds.time.isel(time=timeInd_range[0])
        ) / 2

        # Store computed flux in an xarray dataset

        if kwargs['ease']:
            ds = xr.Dataset({
                "flux": ([lat, lon], flux.values)
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})
        
            ds = ds.assign_coords(latitude=((lat, lon), self.ds['lat'].values), 
                                  longitude=((lat, lon), self.ds['lon'].values))
        else:
            ds = xr.Dataset({
                "flux": ([lat, lon], flux.values)
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})
        return pyFlux(ds, **kwargs), material_derivative

    def calculate_density_flux(self, flux_components, timeInd, **kwargs):
        """
        Compute the density flux through the sea surface.

        This method calculates the surface density flux as the sum of its thermal and haline components, 
        based on the material derivatives of sea surface temperature (SST) and sea surface salinity (SSS),
        weighted by the local mixed layer depth and coefficients of thermal expansion and haline contraction.

        Parameters
        ----------
        flux_components : tuple of xarray.DataArray
            A tuple containing the haline flux and thermal flux, typically returned by the `flux()` method:
            (flux_sss, flux_sst).
        timeInd : int
            Index of the time step at which to compute the density flux.
        **kwargs :
            Additional keyword arguments:

            sss : str
                Name of the sea surface salinity variable.
            sst : str
                Name of the sea surface temperature variable.
            ssd : str
                Name of the sea surface density variable (used to estimate in situ density).
            time : str
                Name of the time dimension.
            lat : str
                Name of the latitude dimension.
            lon : str
                Name of the longitude dimension.

        Returns
        -------
        pyFlux
            pyFlux object containing both thermal (dflux_sst) and haline (dflux_sss) components 
            of the density flux in an xarray.Dataset.

        Notes
        -----
        - The density flux is calculated as:
            Dρ/Dt = -ρ α DSST/Dt + ρ β DSSS/Dt,
          where α and β are the thermal expansion and haline contraction coefficients,
          and ρ is the in situ surface density (SSD).
        - α and β are computed using the Gibbs Seawater (GSW) library at p = 0 dbar.
        - Averages are computed over the two selected time steps to align with centered differencing.
        - The time dimension in the input dataset must include at least two consecutive time steps.
        - This method assumes that the `flux` method was previously called to obtain the flux components.
        """
        kwargs['calculate_ssd'] = False
        time, lat, lon = kwargs['time'], kwargs['lat'], kwargs['lon']

        # Select two time steps
        timeInd_range = [timeInd, timeInd + 1]
        ds = self.ds.isel(time=timeInd_range)

        # calculate the mean density, thermal expansion, and haline contraction coefficients
        alpha = gsw.alpha(ds[kwargs['sss']].values, ds[kwargs['sst']].values, 0)
        beta = gsw.beta(ds[kwargs['sss']].values, ds[kwargs['sst']].values, 0)
        mean_rho = np.nanmean(ds.ssd.values, 0) 
        mean_alpha = np.nanmean(alpha, 0) 
        mean_beta = np.nanmean(beta, 0) 

        # the components of the density flux and their sum-the density flux 
        dflux_sss = mean_rho * mean_beta * flux_components[0]
        dflux_sst = -(mean_rho * mean_alpha * flux_components[1])

        # Assign time coordinate as midpoint between two selected timesteps
        ds_diff = self.ds[time].isel(time=timeInd_range[0]) + (
            self.ds[time].isel(time=timeInd_range[1]) - self.ds[time].isel(time=timeInd_range[0])
        ) / 2
        # Store computed flux in an xarray dataset

        if kwargs['ease']:
            ds = xr.Dataset({
                "dflux_sss": ([lat, lon], dflux_sss),
                "dflux_sst": ([lat, lon], dflux_sst),
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})
        
            ds = ds.assign_coords(latitude=((lat, lon), self.ds['lat'].values), 
                                  longitude=((lat, lon), self.ds['lon'].values))
        else:
            ds = xr.Dataset({
                "dflux_sss": ([lat, lon], dflux_sss),
                "dflux_sst": ([lat, lon], dflux_sst),
            }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})

        return pyFlux(ds, **kwargs)

    def calculate_all_fluxes(self, **kwargs):
        """
        Compute freshwater, heat, and density fluxes using surface ocean variables.

        This method calculates the material derivatives of sea surface density (SSD), salinity (SSS), 
        and temperature (SST), and then computes the respective fluxes using the mixed layer depth (MLD). 
        It also separates thermal and haline contributions to the density flux.

        Parameters:
            kwargs:
                sst (str): Name of the sea surface temperature variable (e.g., 'thetao').
                sss (str): Name of the sea surface salinity variable (e.g., 'so').
                ssd (str): Name of the sea surface density variable (optional if calculated).
                u (str): Name of the eastward surface velocity component (e.g., 'vxo').
                v (str): Name of the northward surface velocity component (e.g., 'vyo').
                mld (str): Name of the mixed layer depth variable (e.g., 'mlotst').
                lat (str): Name of the latitude coordinate (e.g., 'y').
                lon (str): Name of the longitude coordinate (e.g., 'x').
                time (str): Name of the time coordinate (e.g., 'time').
                time_resolution (str): Temporal resolution of the dataset, 'D' for daily, 'M' for monthly.
                ease (bool): Whether to assume EASE grid geometry (default: False).
                ease_res_km (int): Resolution of the EASE grid in kilometers (required if ease=True).
                calculate_ssd (bool): If True, compute density internally using sst/sss and equation of state.
                to_netcdf (bool): If True, save the output flux dataset to a file named 'fluxes.nc'.

        Returns:
            pyFlux: A new instance containing an xarray.Dataset with computed variables:
                - material_derivative_ssd (xarray.DataArray)
                - material_derivative_sss (xarray.DataArray)
                - material_derivative_sst (xarray.DataArray)
                - flux_ssd (xarray.DataArray)
                - flux_sss (xarray.DataArray)
                - flux_sst (xarray.DataArray)
                - dflux_sss (xarray.DataArray)
                - dflux_sst (xarray.DataArray)

        Notes:
            - Uses a Forward-Time Centered-Space (FTCS) differencing scheme for material derivatives.
            - At least 3 time steps are required to compute forward differences.
            - Output time axis is mid-shifted (between time steps) and shortened by one.
            - Assumes time is increasing monotonically and regularly spaced.
            - Results are stored in an xarray.Dataset and wrapped in a pyFlux object.
        """
        kwargs['calculate_ssd'] = False
        time, lat, lon = kwargs['time'], kwargs['lat'], kwargs['lon']
        sss, sst = kwargs['sss'], kwargs['sst']

        print("=" * 40)
        print('Initialising flux variables...', flush=True)

        ds_original = self.ds
        time_size_minus_one = ds_original.sizes[time] - 1
        lat_size = ds_original.sizes[lat]
        lon_size = ds_original.sizes[lon]

        material_derivative_ssd = np.full((time_size_minus_one, lat_size, lon_size), np.nan)
        material_derivative_sss = np.full((time_size_minus_one, lat_size, lon_size), np.nan)
        material_derivative_sst = np.full((time_size_minus_one, lat_size, lon_size), np.nan)
        flux_ssd = np.full((time_size_minus_one, lat_size, lon_size), np.nan)
        flux_sss = np.full((time_size_minus_one, lat_size, lon_size), np.nan)
        flux_sst = np.full((time_size_minus_one, lat_size, lon_size), np.nan)
        dflux_sst = np.full((time_size_minus_one, lat_size, lon_size), np.nan)
        dflux_sss = np.full((time_size_minus_one, lat_size, lon_size), np.nan)

        if ds_original.sizes[kwargs['time']] <= 2:
            sys.exit('Dataset must have more than 2 time steps')

        for timeInd in tqdm(range(time_size_minus_one), desc="Calculating fluxes", unit="step"):
            flux, material_derivative = self.flux('ssd', timeInd, **kwargs)
            flux_ssd[timeInd, :, :] = flux.ds.flux.values
            material_derivative_ssd[timeInd, :, :] = material_derivative.ds.material_derivative.values

            flux, material_derivative = self.flux(sss, timeInd, **kwargs)
            flux_sss[timeInd, :, :] = flux.ds.flux.values
            material_derivative_sss[timeInd, :, :] = material_derivative.ds.material_derivative.values

            flux, material_derivative = self.flux(sst, timeInd, **kwargs)
            flux_sst[timeInd, :, :] = flux.ds.flux.values
            material_derivative_sst[timeInd, :, :] = material_derivative.ds.material_derivative.values

            flux_components = [flux_sss[timeInd, :, :], flux_sst[timeInd, :, :]]
            dflux_components = self.calculate_density_flux(flux_components, timeInd, **kwargs)
            dflux_sss[timeInd, :, :] = dflux_components.ds.dflux_sss.values
            dflux_sst[timeInd, :, :] = dflux_components.ds.dflux_sst.values

        first_time_halfshift = (ds_original[kwargs['time']][0] + (ds_original[kwargs['time']].diff('time') / 2))[0]
        rest_time_halfshift = ds_original[kwargs['time']][:-1] + (ds_original[kwargs['time']].diff('time') / 2)
        time_shifted = xr.concat([first_time_halfshift, rest_time_halfshift], dim='time')

        if kwargs['ease']:
            ds = xr.Dataset({
                "material_derivative_ssd": ([time, lat, lon], material_derivative_ssd),
                "material_derivative_sss": ([time, lat, lon], material_derivative_sss),
                "material_derivative_sst": ([time, lat, lon], material_derivative_sst),
                "flux_ssd": ([time, lat, lon], flux_ssd),
                "flux_sss": ([time, lat, lon], flux_sss),
                "flux_sst": ([time, lat, lon], flux_sst),
                "dflux_sss": ([time, lat, lon], dflux_sss),
                "dflux_sst": ([time, lat, lon], dflux_sst),
            }, coords={time: time_shifted, lat: ds_original[lat], lon: ds_original[lon]})
        
            ds = ds.assign_coords(latitude=((lat, lon), ds_original['lat'].values), 
                                  longitude=((lat, lon), ds_original['lon'].values))
        else:
            ds = xr.Dataset({
                "material_derivative_ssd": ([time, lat, lon], material_derivative_ssd),
                "material_derivative_sss": ([time, lat, lon], material_derivative_sss),
                "material_derivative_sst": ([time, lat, lon], material_derivative_sst),
                "flux_ssd": ([time, lat, lon], flux_ssd),
                "flux_sss": ([time, lat, lon], flux_sss),
                "flux_sst": ([time, lat, lon], flux_sst),
                "dflux_sss": ([time, lat, lon], dflux_sss),
                "dflux_sst": ([time, lat, lon], dflux_sst),
            }, coords={time: time_shifted, lat: ds_original[lat], lon: ds_original[lon]})


        if kwargs.get('to_netcdf', False):
            filename = "fluxes.nc"
            print('\nWriting all fluxes to ' + filename + '...', flush=True)
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(filename, encoding=encoding)

        print('Done', flush=True)
        return pyFlux(ds, **kwargs)

    def map_plot(self, var, timeInd, vmax=0, vmin=0, **kwargs):
        """
        Plot a 2D spatial map of a dataset variable using Cartopy and pcolormesh.

        Parameters:
            var (str): Name of the variable in self.ds to plot.
            timeInd (int or str): Time index to select:
                - Use an integer >= 0 to select a specific time step.
                - Use -1 to plot the time mean.
                - Use -2 to plot the time sum.
            vmax (float): Upper color limit for the plot. If 0, auto-calculated as mean + std.
            vmin (float): Lower color limit for the plot. If 0, auto-calculated as mean - std.
            kwargs:
                lat (str): Name of the latitude coordinate (required).
                lon (str): Name of the longitude coordinate (required).
                cmap (str): Matplotlib colormap name (default: 'turbo').
                cbar (bool): Whether to include a colorbar (default: True).
                cbar_label (str): Label for the colorbar (default: '').
                title (str): Title of the plot (default: '').
                savefig (bool): If True, saves the plot to disk instead of showing (default: False).
                imname (str): Filename to save the image as (without extension, default: 'figure').
                isglobal (bool): If True, uses global map projection; otherwise applies `extent` (default: True).
                central_lon (float): Central longitude for map projection (optional; not currently used).
                extent (list of float): Geographic extent [lon_min, lon_max, lat_min, lat_max] if isglobal is False.
                isrect (bool): Whether to draw a rectangle (not implemented).
                rect (list): Rectangle coordinates or bounds (not implemented).

        Returns:
            None

        Notes:
            - Uses Cartopy with `self.crs` as the projection for map visualization.
            - Latitude and longitude must be provided as keyword arguments.
            - Automatically computes color scale limits if vmax and vmin are not provided.
            - Designed for 2D horizontal data; if time dimension exists, it is indexed.
            - Output is either shown interactively or saved as a high-resolution PNG file.
        """
        # --- Check lat/lon kwargs
        if 'lat' not in kwargs or 'lon' not in kwargs:
            raise ValueError("Both 'lat' and 'lon' must be specified in kwargs.")

        latname = kwargs['lat']
        lonname = kwargs['lon']

        # --- Set other defaults
        cmap = kwargs.get('cmap', 'turbo')
        cbar = kwargs.get('cbar', True)
        title = kwargs.get('title', '')
        savefig = kwargs.get('savefig', False)
        imname = kwargs.get('imname', 'figure')
        isglobal = kwargs.get('isglobal', True)
        extent = kwargs.get('extent', [0, 0, 0, 0])
        cbar_label = kwargs.get('cbar_label', '')

        # --- Select data
        if 'time' in self.ds.dims:
            if timeInd >= 0:
                data = self.ds[var].isel(time=timeInd)
            elif timeInd == -1:
                data = self.ds[var].mean(dim='time')
            elif timeInd == -2:
                data = self.ds[var].sum(dim='time')
        else:
            data = self.ds[var]

        # --- Extract lat/lon
        try:
            lats = self.ds[latname]
            lons = self.ds[lonname]
        except KeyError:
            raise KeyError(f"Latitude '{latname}' or longitude '{lonname}' not found in dataset.")

        # --- Handle color limits
        if vmax == 0 and vmin == 0:
            vmax = float(data.mean() + data.std())
            vmin = float(data.mean() - data.std())

        # --- Set up the figure
        plt.rcParams.update({'font.size': 11})
        fig, ax = plt.subplots(figsize=(10, 5),
                               subplot_kw={'projection': self.crs})

        # --- Set extent if needed
        if not isglobal:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # --- Add map features
        ax.coastlines(resolution='110m', color='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False

        # --- Plot
        if kwargs['ease']:
            pcm = ax.pcolor(
                self.ds.longitude.values, self.ds.latitude.values, data.values,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                shading='auto',
                vmin=vmin, vmax=vmax
            )
        else:
            pcm = ax.pcolor(
                lons.values, lats.values, data.values,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                shading='auto',
                vmin=vmin, vmax=vmax
            )

        # --- Colorbar
        if cbar:
            cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', fraction=0.046, pad=0.06)
            cbar.set_label(cbar_label)

        # --- Title
        ax.set_title(title, fontsize=16, fontweight='bold')

        # --- Save figure or show
        if savefig:
            plt.savefig(imname + '.png', dpi=600, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
