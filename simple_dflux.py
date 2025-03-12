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
    ``derivative_degrees2meters`` computes first order derivative of an ocean scalar, using
    the FTCS (Forward time-centered space) differncing scheme. Deals with irregular grid sizes by 
    calculating latitudinal distances at each latitude (instead of assigning a constant distance)
    and so when calculating the longitude distances these values are used
    """
    dtime, dlat, dlon = np.gradient(ds[kwargs['scalar']])
    if kwargs['time_resolution'] == 'D':
        dtime = dtime[0] / (86400)
    elif kwargs['time_resolution'] == 'W':
        dtime = dtime[0] / (7 * 86400)
    elif kwargs['time_resolution'] == 'M':
        dtime = dtime[0] / (kwargs['days_in_month'] * 86400)


    if kwargs['spatial_resolution'] == 'DEG':
        earth_radius = 6370 * 1000
        lat_degree_distance = (2 * np.pi * earth_radius) / 360

        res = np.gradient(ds[kwargs['lat']])
        distance_lat = lat_degree_distance * res
        lon_grid, lat_grid = np.meshgrid(ds[kwargs['lon']], ds[kwargs['lat']])
        res_lon_meters = np.abs(np.gradient(lon_grid)[1] * lat_degree_distance)
        distance_lon = res_lon_meters * np.cos(np.radians(lat_grid))

        dlon = np.nanmean(dlon, 0) 
        dlat = np.nanmean(dlat, 0) 
        for lat in range(ds.sizes[kwargs['lat']]):
            dlon[lat, :] = dlon[lat, :] / (2*distance_lon[lat, :])
        for lon in range(ds.sizes[kwargs['lon']]):
            dlat[:, lon] = dlat[:, lon] / (2 * distance_lat)
    elif kwargs['spatial_resolution'] == 'Km':
        res = np.gradient(ds[kwargs['lat']])
        distance_lat = res * 1000
        res = np.gradient(ds[kwargs['lat']]) * 1000
        lat_grid, lon_grid = np.meshgrid(ds[kwargs['lon']], ds[kwargs['lat']])
        res_lon_meters = np.gradient(lon_grid)[0] * 1000
        distance_lon = res_lon_meters * np.cos(np.radians(lat_grid))

        dlon = np.nanmean(dlon, 0) 
        dlat = np.nanmean(dlat, 0) 
        for lat in range(ds.sizes[kwargs['lat']]):
            dlon[lat, :] = dlon[lat, :] / distance_lon[lat, :]
        for lon in range(ds.sizes[kwargs['lon']]):
            dlat[:, lon] = dlat[:, lon] / distance_lat


    return dtime, dlat, dlon

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
TODO                   -spatial_res_type, type of spatial grid [DEG | KM]

    Returns
    -------
    numpy matrix : a 2D numpy matrix with shape of original dataset minus the time dimension
    (dataset should only have 2 timesteps). The timestamp of the dataset will be in between the 2 
    timesteps of the original input data

    Notes
    -----
    ``calculate_material_derivative`` computes the material derivative of an ocean scalar, using
    the FTCS (Forward time-centered space) differncing scheme (via a call to 
    ``derivative_degrees2meters``) 
    """
    dtime, dlat, dlon = derivative_degrees2meters(ds, kwargs)
    temporal_derivative = dtime

    mean_u = np.nanmean(ds[kwargs['u']].data, 0) 
    mean_v = np.nanmean(ds[kwargs['v']].data, 0) 
    advective_derivative = dlon * mean_u + dlat * mean_v
    material_derivative = temporal_derivative + advective_derivative

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
TODO                   -spatial_res_type, type of spatial grid [DEG | KM]
TODO                   -to_netcdf, BOOL TYPE FOR OUTPUT AS A COMPRESSED NETcdf FILE}

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
    xarray Dataset : four new entries into `ds` being, 
                   dflux_sss/sst, haline/thermal component of the density flux
                   dflux, the density flux
                   flux_sss/sst, the freshwater/heat flux 
    file netCDF : optionally saves the populated xarray Dataset as a compressed netCDF

    Notes
    -----
    before calling this function, you should make a prior call to ``calculate_flux``
    density_flux computes the density flux through the sea surface as the product between the 
    mixed layer depth and the material derivative of sea surface density. Additionally, 
    ``calculate_density_flux`` estimates the thermal and haline components of the density flux.
    """
    rho, alpha, beta = gsw.rho_alpha_beta(ds[kwargs['sss']].data, ds[kwargs['sst']].data, 0)

    mean_rho = np.nanmean(rho, 0) 
    mean_alpha = np.nanmean(alpha, 0) 
    mean_beta = np.nanmean(beta, 0) 

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
TODO                   -spatial_res_type, type of spatial grid [DEG | KM]

    Returns
    -------
    5 numpy matrices : 1, freshwater flux
                       2, heat flux
                       3, density flux
                       4, density flux (thermal component)
                       5, density flux (haline component)

    Notes
    -----
    ``compute_all_fluxes`` computes freshwater/heat and density fluxes (including all components)
    as the products between the respective material derivatives of the scalars and a mixed layer 
    depth. It uses the FTCS (Forward time-centered space) differncing scheme to compute the 
    the respective derivatives of time, latitude and longitude. The outputs are referenced to 
    a time grid which is 1/2 forward shifted (relative to the input), although the dimension length
    is the same as the input (using cyclical boundary conditions)
    """
    # adding ssd to the dataset to compute its fluxes alone 
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

        #TODO: do the same for density
        kwargs['scalar'] = 'ssd'
        flux_ssd[timestep, :, :], material_derivative_ssd[timestep, :, :] =\
                calculate_flux(ds, kwargs)
        #TODO: we want mld division done later
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
        "material_derivative_ssd": (["time", "lat", "lon"], material_derivative_ssd),
        "material_derivative_sss": (["time", "lat", "lon"], material_derivative_sss),
        "material_derivative_sst": (["time", "lat", "lon"], material_derivative_sst),
        "flux_ssd": (["time", "lat", "lon"], flux_ssd),
        "flux_sss": (["time", "lat", "lon"], flux_sss),
        "flux_sst": (["time", "lat", "lon"], flux_sst),
        "dflux": (["time", "lat", "lon"], dflux),
        "dflux_sss": (["time", "lat", "lon"], dflux_sss),
        "dflux_sst": (["time", "lat", "lon"], dflux_sst),
    }, coords={"time": time_shifted, "lat": ds_original[kwargs['lat']], 
               "lon": ds_original[kwargs['lon']]})

    if kwargs['to_netcdf']:
        filename = "fluxes.nc"
        print('\nWriting all fluxes to ' + filename + '...', end="" , flush=True)
        comp = dict(zlib=True, complevel=5)
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(filename, encoding=encoding)

    print('\nDone', end="" , flush=True)

    return ds


