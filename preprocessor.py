# data preprocessor should be called by the main function before calculating density flux
# purpose: to take any combination of datasets with irregular resolutions (i.e. each of the 
# datasets have a different spatial and temporal resolution) 
# 
# this fuction then will reinterpolate in space and time to standardise all input data before
# passing on the data to estimate density fluxes
# structure: TODO user will probably have to provide the function with a text file containing
#   the paths to all the datasets
#            TODO also output should be a single xarray object containing all inputs homogenised to
#   desired resolution (space and time) to then pass to estimate density fluxes



# do, users provides four seperate files of salinity temperature mld U.
# function puts them on a sing unified grid and then includes all in a single xarray datasets that 
# can be passed to compute_all_fluxes
