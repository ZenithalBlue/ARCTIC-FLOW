# A collection of functions for flux calculations
# along with some helper functions for plotting and computing informative statistics
import xarray as xr
import numpy as np
import gsw
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define constants and parameters
earth_radius = 6371000  # Earth radius in meters
degree_distance = 2 * np.pi * earth_radius / 360  # Distance in meters for 1 degree
months_day_count = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Define a class for the pyFlux object

class pyFlux:
    """
    A class for computing flux-related diagnostics in oceanographic datasets.

    Attributes:
        ds (xarray.Dataset): The dataset containing oceanographic variables.
    """

    def __init__(self, ds):
        """
        Initialize the pyFlux class with an xarray dataset.

        Parameters:
            ds (xarray.Dataset): The dataset containing oceanographic variables.
        """
        self.ds = ds  # Store dataset

        # Compute spatial resolution in degrees
        self.resolution_latitude = np.gradient(self.ds.latitude.values)
        self.resolution_longitude = np.gradient(self.ds.longitude.values)

        # Convert latitude resolution to meters
        self.resolution_latitude_meters = self.resolution_latitude * degree_distance

        # Compute longitude resolution in meters, considering latitude dependence
        meshgrid_lon, meshgrid_lat = np.meshgrid(self.ds.longitude.values, self.ds.latitude.values)
        self.resolution_longitude_meters = np.gradient(meshgrid_lon)[1] * degree_distance * np.cos(np.radians(meshgrid_lat))

        # Compute time resolution
        self.resolution_time = np.gradient(self.ds.time.values)

    def derivative_degrees2meters(self, var, timeInd, **kwargs):
        """
        Compute first-order derivatives (time, latitude, longitude) using FTCS differencing.

        Parameters:
            timeInd (int): Index of the time step for which the derivative is computed.
            kwargs:
                var (str): Name of the scalar variable to differentiate.
                time (str): Name of the time dimension.
                lat (str): Name of the latitude dimension.
                lon (str): Name of the longitude dimension.
                time_resolution (str): Temporal resolution ['D' (daily), 'W' (weekly), 'M' (monthly)].
                days_in_month (int, optional): Number of days in a month (if 'M' resolution is used).

        Returns:
            pyFlux object: Containing the computed derivatives.
        """
        # Extract relevant variable names
        time, lat, lon = kwargs['time'], kwargs['lat'], kwargs['lon']

        # Select two consecutive time steps
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
        for lat_index in range(self.ds.sizes[lat]):
            derivative_longitude[lat_index, :] /= (2 * self.resolution_longitude_meters[lat_index, :])
        for lon_index in range(self.ds.sizes[lon]):
            derivative_latitude[:, lon_index] /= (2 * self.resolution_latitude_meters)

        # Compute midpoint time for reference
        ds_diff = self.ds.time.isel(time=timeInd_range[0]) + (self.ds.time.isel(time=timeInd_range[1]) -
                                                              self.ds.time.isel(time=timeInd_range[0])) / 2

        # Store computed derivatives in an xarray dataset
        ds = xr.Dataset({
            "derivative_time": ([lat, lon], derivative_time),
            "derivative_latitude": ([lat, lon], derivative_latitude),
            "derivative_longitude": ([lat, lon], derivative_longitude)
        }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})

        return pyFlux(ds)

    def material_derivative(self, var, timeInd, **kwargs):
        """
        Compute the material derivative of an ocean scalar.

        The material derivative accounts for changes in a variable due to both time evolution
        and advection by ocean currents.

        Parameters:
            timeInd (int): Index of the time step for which the derivative is computed.
            kwargs:
                var (str): Name of the scalar variable to differentiate.
                u (str): Name of the eastward velocity component.
                v (str): Name of the northward velocity component.
                time (str): Name of the time dimension.
                lat (str): Name of the latitude dimension.
                lon (str): Name of the longitude dimension.

        Returns:
            pyFlux object: Containing the computed material derivative.
        """
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
        pyFlux_derivatives = self.derivative_degrees2meters(var, timeInd, **kwargs)

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
        ds = xr.Dataset({
            "material_derivative": ([lat, lon], material_derivative.values)
        }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})

        return pyFlux(ds)

    def flux(self, var, timeInd, **kwargs):
        """
        Compute the surface flux of an ocean scalar.

        The flux is computed as the product of the material derivative of the scalar
        and the mixed layer depth.

        Parameters:
            timeInd (int): Index of the time step for flux calculation.
            kwargs:
                var (str): Name of the scalar variable.
                mld (str): Name of the mixed layer depth variable.
                time (str): Name of the time dimension.
                lat (str): Name of the latitude dimension.
                lon (str): Name of the longitude dimension.

        Returns:
            tuple: (pyFlux object containing the flux, pyFlux object containing the material derivative)
        """
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
        ds = xr.Dataset({
            "flux": ([lat, lon], flux.values)
        }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})

        return pyFlux(ds), material_derivative

    def calculate_density_flux(self, flux_components, timeInd, **kwargs):
        """
        Calculate the density flux through the sea surface.

        The density flux is computed as the sum of the thermal and haline contributions, 
        which are determined by the product of the mixed layer depth and the material 
        derivative of sea surface temperature (SST) and sea surface salinity (SSS).

        Parameters:
            timeInd (int): Index of the time step for density flux calculation.
            flux_components (tuple): Tuple containing the sss and sst flux.
            kwargs:
                time (str): Name of the time dimension.
                lat (str): Name of the latitude dimension.
                lon (str): Name of the longitude dimension.

        Returns:
            tuple: (pyFlux object containing the thermal density flux, 
                   pyFlux object containing the haline density flux)

        Notes:
            This function requires a prior call to ``flux`` to obtain the necessary flux components. 
            The density flux is calculated as the sum of thermal and haline components, each of which is based 
            on the respective coefficients of thermal expansion and haline contraction, averaged over the time dimension.
            The time dimension must have a length of 2 to avoid temporal grid mismatches in the material derivative.
        """
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
        ds = xr.Dataset({
            "dflux_sss": ([lat, lon], dflux_sss),
            "dflux_sst": ([lat, lon], dflux_sst),
        }, coords={lat: ds[lat], lon: ds[lon], time: ds_diff})

        return pyFlux(ds)

    def calculate_all_fluxes(self, **kwargs):
        """
        Compute freshwater, heat, and density fluxes using surface ocean variables.

        This method calculates the material derivatives of sea surface density (SSD), salinity (SSS), 
        and temperature (SST), and then computes the respective fluxes using the mixed layer depth (MLD). 

        Parameters:
            kwargs:
                sst (str): Name of the sea surface temperature variable.
                sss (str): Name of the sea surface salinity variable.
                u (str): Name of the eastward surface velocity component.
                v (str): Name of the northward surface velocity component.
                mld (str): Name of the mixed layer depth variable.
                lat (str): Name of the latitude coordinate.
                lon (str): Name of the longitude coordinate.
                time (str): Name of the time coordinate.
                time_resolution (str): Temporal resolution ['D' (daily), 'M' (monthly)].
                to_netcdf (bool, optional): Whether to save results as a NetCDF file (default: False).

        Returns:
            pyFlux object: Containing an xarray Dataset with computed flux variables:
                - material_derivative_ssd (xarray.DataArray): Material derivative of surface density.
                - material_derivative_sss (xarray.DataArray): Material derivative of surface salinity.
                - material_derivative_sst (xarray.DataArray): Material derivative of surface temperature.
                - flux_ssd (xarray.DataArray): Density flux (product of SSD material derivative and MLD).
                - flux_sss (xarray.DataArray): Freshwater flux (product of SSS material derivative and MLD).
                - flux_sst (xarray.DataArray): Heat flux (product of SST material derivative and MLD).

        Notes:
            - Uses a **Forward-Time Centered-Space (FTCS) differencing scheme** to compute derivatives.
            - Outputs have **one less time step** than the input dataset due to differencing.
            - Time coordinates are adjusted to **midpoints between time steps** for consistency.
        """
        time, lat, lon = kwargs['time'], kwargs['lat'], kwargs['lon']
        sss, sst = kwargs['sss'], kwargs['sst']

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

        for timeInd in range(time_size_minus_one):
            print("\rProgress:" + ' ' + str(round(timeInd/(time_size_minus_one - 1) * 100)) + '%',
                  end="", flush=True)

            flux, material_derivative = self.flux('ssd', timeInd, **kwargs)
            flux_ssd[timeInd, :, :] = flux.ds.flux.values
            material_derivative_ssd[timeInd, :, :] =\
                    material_derivative.ds.material_derivative.values

            flux, material_derivative = self.flux(sss, timeInd, **kwargs)
            flux_sss[timeInd, :, :] = flux.ds.flux.values
            material_derivative_sss[timeInd, :, :] =\
                    material_derivative.ds.material_derivative.values

            flux, material_derivative = self.flux(sst, timeInd, **kwargs)
            flux_sst[timeInd, :, :] = flux.ds.flux.values
            material_derivative_sst[timeInd, :, :] =\
                    material_derivative.ds.material_derivative.values

            flux_components = [flux_sss[timeInd, :, :], flux_sst[timeInd, :, :]]
            dflux_components = self.calculate_density_flux(flux_components, timeInd, **kwargs)
            dflux_sss[timeInd, :, :] = dflux_components.ds.dflux_sss.values
            dflux_sst[timeInd, :, :] = dflux_components.ds.dflux_sst.values

        first_time_halfshift = (ds_original[kwargs['time']][0] + (ds_original[kwargs['time']].diff('time') / 2))[0]
        rest_time_halfshift = ds_original[kwargs['time']][:-1] + (ds_original[kwargs['time']].diff('time') / 2)
        time_shifted = xr.concat([first_time_halfshift, rest_time_halfshift], dim = 'time')

        ds = xr.Dataset({
            "material_derivative_ssd": ([time, lat, lon], material_derivative_ssd),
            "material_derivative_sss": ([time, lat, lon], material_derivative_sss),
            "material_derivative_sst": ([time, lat, lon], material_derivative_sst),
            "flux_ssd": ([time, lat, lon], flux_ssd),
            "flux_sss": ([time, lat, lon], flux_sss),
            "flux_sst": ([time, lat, lon], flux_sst),
            "dflux_sss": ([time, lat, lon], dflux_sss),
            "dflux_sst": ([time, lat, lon], dflux_sst),
        }, coords={time: time_shifted, lat: ds_original[lat], 
                   lon: ds_original[lon]})

        if kwargs['to_netcdf']:
            filename = "fluxes.nc"
            print('\nWriting all fluxes to ' + filename + '...', end="" , flush=True)
            comp = dict(zlib=True, complevel=5)
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(filename, encoding=encoding)

        print('\nDone', flush=True)

        return pyFlux(ds)

    def map_plot(self, var, timeInd, **kwargs):
        """
        Plot a spatial dataset on a map using Cartopy.

        Parameters:
            var (str): Variable to plot from self.ds.
            timeInd (int or str): Time index to select for plotting or "mean" for time-averaged plot.
            kwargs:
                vmax (float): Maximum color scale value. Default: mean + std.
                vmin (float): Minimum color scale value. Default: mean - std.
                cmap (str): Colormap to use. Default: 'turbo'.
                cbar (int): Whether to add a colorbar (1 = Yes, 0 = No). Default: 1.
                title (str): Title of the plot. Default: ''.
                savefig (int): Whether to save figure (1 = Yes, 0 = No). Default: 0.
                imname (str): Filename if saving figure. Default: 'figure'.
                isglobal (int): Whether the map is global (1 = Yes, 0 = No). Default: 1.
                central_lon (float): Central longitude for projection. Default: 0.
                extent (list): Map extent [lon_min, lon_max, lat_min, lat_max]. Default: [0, 0, 0, 0].
                isrect (int): Whether to draw a rectangle (1 = Yes, 0 = No). Default: 1.
                rect (list): Rectangle [lat_min, lon_min, width, height]. Default: [0, 0, 0, 0].
        """
        # Select dataset based on time index
        if 'time' in self.ds.dims:
            if timeInd >= 0:
                ds = self.ds[var].isel(time=timeInd)
            elif timeInd == "mean":
                ds = self.ds[var].mean(dim='time')
        else:
            ds = self.ds[var]

        # Set default values if not provided
        kwargs.setdefault('vmax', ds.mean() + ds.std())
        kwargs.setdefault('vmin', ds.mean() - ds.std())
        kwargs.setdefault('cmap', 'turbo')
        kwargs.setdefault('cbar', 1)
        kwargs.setdefault('title', '')
        kwargs.setdefault('savefig', 0)
        kwargs.setdefault('imname', 'figure')
        kwargs.setdefault('isglobal', 1)
        kwargs.setdefault('central_lon', 0)
        kwargs.setdefault('extent', [0, 0, 0, 0])
        kwargs.setdefault('isrect', 1)
        kwargs.setdefault('rect', [0, 0, 0, 0])

        # Set up the map projection
        proj = ccrs.Robinson(central_longitude=kwargs['central_lon'])
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection=proj)

        # Apply extent if not global
        if kwargs['isglobal'] == 0:
            ax.set_extent(kwargs['extent'], ccrs.PlateCarree())

        # Add coastlines and gridlines
        ax.coastlines(color='.5')
        gl = ax.gridlines(draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_right = False

        # Plot dataset with colorbar
        ds.plot(
            vmax=kwargs['vmax'], vmin=kwargs['vmin'],
            transform=ccrs.PlateCarree(), cmap=kwargs['cmap'], ax=ax,
            cbar_kwargs={'fraction': 0.046, 'pad': 0.06} if kwargs['cbar'] else {'add_colorbar': False}
        )

        # Show or save the plot
        plt.title(kwargs['title'], fontdict={'fontsize': 15, 'fontweight': 'bold'})
        if kwargs['savefig']:
            plt.savefig(kwargs['imname'] + '.jpg', dpi=600)
            plt.close()


    def print_info(self):
        """
        Print a summary of the pyFlux object, including dataset size and variable details.
        """
        print('pyFlux Object Initialized:')
        print(f'\n\tDataset size: {self.ds.sizes}')
        print(f'\tDataset variables: {list(self.ds.data_vars)}')
        print(f"\tSpatial resolution: {np.nanmean(self.resolution_latitude)}° x {np.nanmean(self.resolution_longitude)}°")
        print(f"\t\tLatitude resolution in meters: {np.nanmean(self.resolution_latitude_meters)}")
        print(f"\t\tLongitude resolution range: {np.nanmin(self.resolution_longitude_meters[0, :])}m - {np.nanmax(self.resolution_longitude_meters[:, 0])}m")

