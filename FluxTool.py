import xarray as xr 
import numpy as np
import matplotlib.pyplot as plt
import gsw
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def derivative_degrees2meters(ds, kwargs):
    """
    derivative_degrees2meters(xarray_dataset ds, dict kwargs)

    Parameters
    ----------
    xarray_dataset_object ds : An xarray Dataset that includes 1 3D (time, lat, lon) variable 
                               as well as auxhiliary information on dimensions.
                               'variable 1', sea surface scalar
                               
    dict kwargs : dictionary of required arguments.
                   -u, name of the variable to be derived,
                   -v, name of the variable to be derived,
                   -mld, name of the variable to be derived,
                   -lat, name of the latitude dimension,
                   -lon, name of the longitude dimension,
                   -scalar, name of the variable to be derived (FOR CALL TO DERIVATIVE_DEG2M),
                   -time_res, temporal resolution of variable [dAILY | mONTHLY]
TODO                   -spatial_res_type, type of spatial grid [DEG | KM]

    Returns
    -------
    3 numpy matrices : 1, first order derivative in time
                       2, first order derivative in latitude 
                       3, first order derivative in longitude

    Notes
    -----
    ``derivative_degrees2meters`` computes first order derivatives of an ocean scalar, using
    the FTCS (Forward time-centered space) differncing scheme. Deals with irregular grid sizes by 
    calculating latitudinal distances at each latitude (instead of assigning a constant distance)
    and the same is done with the longitudes (also the latitudes are taken into account to calculate
    longitudinal distances).
    """
    # checking if the time dimension is of size 2 
    if ds.sizes[kwargs['time']] > 2:
        print('the size of the time dimension should be 2/n')
        exit
    derivative_time, derivative_latitude, derivative_longitude = np.gradient(ds[kwargs['scalar']])
    if kwargs['time_resolution'] == 'D':
        derivative_time = derivative_time[0] / (86400)
    elif kwargs['time_resolution'] == 'W':
        derivative_time = derivative_time[0] / (7 * 86400)
    elif kwargs['time_resolution'] == 'M':
        derivative_time = derivative_time[0] / (kwargs['days_in_month'] * 86400)


    if kwargs['spatial_resolution'] == 'DEG':
        earth_radius = 6370 * 1000
        lat_degree_distance = (2 * np.pi * earth_radius) / 360

        distance_lat = np.gradient(ds[kwargs['lat']]) * lat_degree_distance 
        lon_grid, lat_grid = np.meshgrid(ds[kwargs['lon']], ds[kwargs['lat']])
        # the distances of each longitude in meters (without the latitudinal dependence)
        res_lon_meters = np.abs(np.gradient(lon_grid)[1] * lat_degree_distance)
        # ...and with taking the latitudinal dependence into account
        distance_lon = res_lon_meters * np.cos(np.radians(lat_grid))
        # averaging over 2 timesteps (results in the same time grid as `derivative_time`)
        derivative_longitude = np.nanmean(derivative_longitude, 0) 
        derivative_latitude = np.nanmean(derivative_latitude, 0) 
        # looping through all lats and longitudes and dividing by their respective grid distances
        for lat in range(ds.sizes[kwargs['lat']]):
            derivative_longitude[lat, :] = derivative_longitude[lat, :] / (2*distance_lon[lat, :])
        for lon in range(ds.sizes[kwargs['lon']]):
            derivative_latitude[:, lon] = derivative_latitude[:, lon] / (2 * distance_lat)
    elif kwargs['spatial_resolution'] == 'Km':
        # here, the user specifies that the resolution is already in km (in the netcdf file)
        # the rest of the operations are the same 
        res_lat_meters = np.gradient(ds[kwargs['lat']]) * lat_degree_distance 
        lat_grid, lon_grid = np.meshgrid(ds[kwargs['lon']], ds[kwargs['lat']])
        res_lon_meters = np.gradient(lon_grid)[0] * 1000
        distance_lon = res_lon_meters * np.cos(np.radians(lat_grid))

        derivative_longitude = np.nanmean(derivative_longitude, 0) 
        derivative_latitude = np.nanmean(derivative_latitude, 0) 
        for lat in range(ds.sizes[kwargs['lat']]):
            derivative_longitude[lat, :] = derivative_longitude[lat, :] / distance_lon[lat, :]
        for lon in range(ds.sizes[kwargs['lon']]):
            derivative_latitude[:, lon] = derivative_latitude[:, lon] / distance_lat


    return derivative_time, derivative_latitude, derivative_longitude

def calculate_material_derivative(ds, kwargs):
    """
    calculate_material_derivative(xarray_dataset ds, dict kwargs)

    Parameters
    ----------
    xarray_dataset_object ds : An xarray Dataset that includes 1 3D (time, lat, lon) variable 
                               as well as auxhiliary information on dimensions.
                               'variable 1', sea surface scalar
                               'variable 3', sea surface currents (u)
                               'variable 4', sea surface currents (v)
                               
    dict kwargs : dictionary of required arguments.
                   -u, name of the variable to be derived,
                   -v, name of the variable to be derived,
                   -lat, name of the latitude dimension,
                   -lon, name of the longitude dimension,
                   -scalar, name of the variable to be derived (FOR CALL TO DERIVATIVE_DEG2M),
                   -time_res, temporal resolution of variable [dAILY | mONTHLY]
                   -spatial_res_type, type of spatial grid [DEG | KM]

    Returns
    -------
    numpy matrix : a 2D numpy matrix with shape of original dataset minus the time dimension
    (dataset should only have 2 timesteps). The timestamp of the dataset will be in between the 2 
    timesteps of the original input data

    Notes
    ----
    ``calculate_material_derivative`` computes the material derivative of an ocean scalar, using
    the FTCS (Forward time-centered space) differncing scheme (via a call to 
    ``derivative_degrees2meters``) 
    """
    # checking if the time dimension is of size 2 
    if ds.sizes[kwargs['time']] > 2:
        print('the size of the time dimension should be 2/n')
        exit
    derivative_time, derivative_latitude, derivative_longitude = derivative_degrees2meters(ds, kwargs)
    '''
    averaging the sea surface currents over 2 timesteps (i.e. keeping the same temporal grid)
    ''' 
    mean_u = np.nanmean(ds[kwargs['u']].data, 0) 
    mean_v = np.nanmean(ds[kwargs['v']].data, 0) 

    advective_derivative = derivative_longitude * mean_u + derivative_latitude * mean_v
    material_derivative = derivative_time + advective_derivative

    return material_derivative

def calculate_flux(ds, kwargs):
    """
    calculate_flux(xarray_dataset ds, dict kwargs)

    Parameters
    ----------
    xarray_dataset_object ds : An xarray Dataset that includes 1 3D (time, lat, lon) variable 
                               as well as auxhiliary information on dimensions.
                               'variable 1', sea surface scalar
                               'variable 3', sea surface currents (u)
                               'variable 4', sea surface currents (v)
                               'variable 5', mixed layer depth
                               
    dict kwargs : dictionary of required arguments.
                   -u, name of the variable to be derived,
                   -v, name of the variable to be derived,
                   -mld, name of the variable to be derived,
                   -lat, name of the latitude dimension,
                   -lon, name of the longitude dimension,
                   -scalar, name of the variable to be derived (FOR CALL TO DERIVATIVE_DEG2M),
                   -time_res, temporal resolution of variable [dAILY | mONTHLY]
                   -spatial_res_type, type of spatial grid [DEG | KM]

    Returns
    -------
    numpy matrix : a 2D numpy matrix with shape of original dataset minus the time dimension
    (dataset should only have 2 timesteps). The timestamp of the dataset will be in between the 2 
    timesteps of the original input data

    Notes
    -----
    ``calculate_flux`` computes the flux through the sea surface, given a ocean scalar, as the 
    product between the mixed layer depth and the material derivative of sea surface density.
    The specific differencing scheme is FTCS (Forward time-centered space) and is done with a 
    call to ``derivative_deg2m`` (through ``calculate_material_derivative``). 
    """
    # checking if the time dimension is of size 2 
    if ds.sizes[kwargs['time']] > 2:
        print('the size of the time dimension should be 2/n')
        exit
    material_derivative = calculate_material_derivative(ds, kwargs)

    mean_mld = np.nanmean(ds[kwargs['mld']].data, 0) 
    flux = material_derivative * mean_mld

    return flux, material_derivative

def calculate_density_flux(ds, kwargs):
    """
    calculate_density_flux(xarray_dataset ds, dict kwargs)

    Parameters
    ----------
    xarray_dataset_object ds : An xarray Dataset that includes 2 3D (time, lat, lon) variables 
                               as well as auxhiliary information on dimensions.
                               NOTE: time must be only of len 2                                
                               'variable 1', sea surface temperature
                               'variable 2', sea surface salinity
                               
    dict kwargs : dictionary of required arguments.
                   -sst, name of the variable to be derived,
                   -sss, name of the variable to be derived,
                   -flux_sss, 2D matrix (lat, lon) for freshwater flux (see ``calculate_flux``),
                   -flux_sst, 2D matrix (lat, lon) for thermal flux (see ``calculate_flux``),

    Returns
    -------
    3 numpy matrices:
        -dflux = the net density flux (the sum of the thermal and haline contributions)
        -dflux_sst = the thermal component of density flux 
        -dflux_sss = the haline component of density flux


    Notes
    -----
    before calling this function, you should make a prior call to ``calculate_flux``
    density_flux computes the respective thermal and haline components of density flux 
    through the sea surface as the product between the mixed layer depth and the material 
    derivative of sea surface salinity and temperature. Additionally, the density flux is a simple
    addition of these thermal and haline components.  ``calculate_density_flux`` estimates the 
    thermal and haline components of the density flux. requires the input data to have a time 
    dimension of size 2 (as the function averages data over 2 timesteps to avoid temporal grid
    mismatches with the material derivative)
    """
    '''
    calculating:
        rho = sea surface density
        alpha = thermal expansion coefficient (i.e. how temperature changes surface density)
        beta = haline contraction coefficient (i.e. how salinity changes surface density)
    '''
    # checking if the time dimension is of size 2 
    if ds.sizes[kwargs['time']] > 2:
        print('the size of the time dimension should be 2/n')
        exit
    rho, alpha, beta = gsw.rho_alpha_beta(ds[kwargs['sss']].data, ds[kwargs['sst']].data, 0)
    # averaging over 2 timesteps to avoid mismatched temporal grids
    mean_rho = np.nanmean(rho, 0) 
    mean_alpha = np.nanmean(alpha, 0) 
    mean_beta = np.nanmean(beta, 0) 

    # the components of the density flux and their sum-the density flux 
    dflux_sss = mean_rho * mean_beta * kwargs['flux_sss']
    dflux_sst = -(mean_rho * mean_alpha * kwargs['flux_sst'])
    dflux = dflux_sst + dflux_sss

    return dflux, dflux_sst, dflux_sss

def calculate_all_fluxes(ds, kwargs):
    """
    calculate_all_fluxes(xarray_dataset ds, dict kwargs)

    Parameters
    ----------
    xarray_dataset_object ds : An xarray Dataset that includes 1 3D (time, lat, lon) variable 
                               as well as auxhiliary information on dimensions.
                               'variable 1', sea surface salinity
                               'variable 2', sea surface temperature 
                               'variable 3', sea surface currents (u)
                               'variable 4', sea surface currents (v)
                               'variable 5', mixed layer depth
                               
    dict kwargs : dictionary of required arguments.
                   -sst, name of the variable to be derived,
                   -sss, name of the variable to be derived,
                   -u, name of the variable to be derived,
                   -v, name of the variable to be derived,
                   -mld, name of the variable to be derived,
                   -lat, name of the latitude dimension,
                   -lon, name of the longitude dimension,
                   -time_res, temporal resolution of variable [dAILY | mONTHLY]
                   -spatial_res_type, type of spatial grid [DEG | KM]

    Returns
    -------
    xarray_dataset_object ds : An xarray Dataset that includes the following variables with
                               the same dimensions as the input ds, although, with one less
                               time dimension (due to derivatives)
                               the variables are:
                       - flux_sss = freshwater flux (i.e. the product between the material 
                       derivative of surface salinity and mld)
                       - flux_sst = heat flux (i.e. the product between the material 
                       derivative of surface temperature and mld)
                       - flux_ssd = density flux (i.e. the product between the material 
                       derivative of surface density and mld)
                       - dflux_sst = thermal density flux  
                       - dflux_sss = haline density flux 
                       - dflux = the sum between the thermal and haline density fluxes (i.e. the 
                       density flux assuming a linear realtion between surface salinity/temperature
                       on surface density)

    Notes
    -----
    ``compute_all_fluxes`` computes freshwater/heat and density fluxes (including all components)
    as the products between the respective material derivatives of the scalars and a mixed layer 
    depth. It uses the FTCS (Forward time-centered space) differncing scheme to compute the 
    the respective derivatives of time, latitude and longitude. The outputs are referenced to 
    a time grid which is 1/2 forward shifted (relative to the input) 
    """
    # adding ssd to the dataset to compute its fluxes alone 
    print('Initialising variables...', end="" , flush=True) 
    ssd = gsw.rho(ds[kwargs['sss']].data, ds[kwargs['sst']].data, 0)
    ds['ssd'] = ([kwargs['time'], kwargs['lat'], kwargs['lon']], ssd)

    ds_original = ds

    material_derivative_ssd = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    material_derivative_sss = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    material_derivative_sst = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    flux_ssd = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    flux_sss = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    flux_sst = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    dflux = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    dflux_sst = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)
    dflux_sss = np.full((ds.sizes[kwargs['time']] - 1, ds.sizes[kwargs['lat']], ds.sizes[kwargs['lon']]), np.nan)

    for timestep in range(ds_original.sizes[kwargs['time']] - 1):
        print("\rProgress:" + ' ' + str(round(timestep/(ds_original.sizes[kwargs['time']] - 2) * 100)) + '%',
              end="", flush=True)
        if kwargs['time_resolution'] == 'M':
            month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            months_in_year = 12
            kwargs['days_in_month'] = month_days[timestep % months_in_year]

        ds = ds_original.isel({'time': [timestep, timestep + 1]})

        kwargs['scalar'] = 'ssd'
        flux_ssd[timestep, :, :], material_derivative_ssd[timestep, :, :] =\
                calculate_flux(ds, kwargs)
        kwargs['scalar'] = kwargs['sss']
        flux_sss[timestep, :, :], material_derivative_sss[timestep, :, :] =\
                calculate_flux(ds, kwargs)
        kwargs['scalar'] = kwargs['sst']
        flux_sst[timestep, :, :], material_derivative_sst[timestep, :, :] =\
                calculate_flux(ds, kwargs)

        kwargs['flux_sss'] = flux_sss[timestep, :, :]
        kwargs['flux_sst'] = flux_sst[timestep, :, :]
        dflux[timestep, :, :], dflux_sst[timestep, :, :], dflux_sss[timestep, :, :] =\
                calculate_density_flux(ds, kwargs)

    first_time_halfshift = (ds_original[kwargs['time']][0] + (ds_original[kwargs['time']].diff('time') / 2))[0]
    rest_time_halfshift = ds_original[kwargs['time']][:-1] + (ds_original[kwargs['time']].diff('time') / 2)
    time_shifted = xr.concat([first_time_halfshift, rest_time_halfshift], dim = 'time')

    ds = xr.Dataset({
        "material_derivative_ssd": ([kwargs['time'], kwargs['lat'], kwargs['lon']], material_derivative_ssd),
        "material_derivative_sss": ([kwargs['time'], kwargs['lat'], kwargs['lon']], material_derivative_sss),
        "material_derivative_sst": ([kwargs['time'], kwargs['lat'], kwargs['lon']], material_derivative_sst),
        "flux_ssd": ([kwargs['time'], kwargs['lat'], kwargs['lon']], flux_ssd),
        "flux_sss": ([kwargs['time'], kwargs['lat'], kwargs['lon']], flux_sss),
        "flux_sst": ([kwargs['time'], kwargs['lat'], kwargs['lon']], flux_sst),
        "dflux": ([kwargs['time'], kwargs['lat'], kwargs['lon']], dflux),
        "dflux_sss": ([kwargs['time'], kwargs['lat'], kwargs['lon']], dflux_sss),
        "dflux_sst": ([kwargs['time'], kwargs['lat'], kwargs['lon']], dflux_sst),
    }, coords={kwargs['time']: time_shifted, kwargs['lat']: ds_original[kwargs['lat']], 
               kwargs['lon']: ds_original[kwargs['lon']]})

    if kwargs['to_netcdf']:
        filename = "fluxes.nc"
        print('\nWriting all fluxes to ' + filename + '...', end="" , flush=True)
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(filename, encoding=encoding)

    print('\nDone', end="" , flush=True)

    return ds


