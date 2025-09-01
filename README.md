# Ensemble-Score-Filter
In order to implement Ensemble Score Filter(EnSF) correctly, you have to figure out the input and output.
Input:  
      pseudo_time_step:        the number of time step in reverse time SDE, eg 1000 is enough in most case   
      prior_ensemble:          the prior ensembles, usually it will be a matrix with dimension(ensemble_size, y_dim)
      ensemble_size:           the number of ensembles
      obs:                     observation, usually it will be a vector with dimension(y_dim,)
      sigma:                   std of observation      
      y_dim:                   the dimension of observation
      scalefact:               a constant number, we assume it to be 1
      indx_indxob_linear:      the index of linear observation
      initial_std:             the initial sample std
Output:
      np.array(x_ens)          the posterior ensembles, usually it will be a matrix with dimension(ensemble_size, y_dim)


Also, here is an example of invoking EnSF:
#####
x_ens_subarray = x_ens[:, indxob]
x_ens[:, indxunob] = ((x_ens[:, indxunob] - x_ens[:, indxunob].mean(axis=0)) / x_ens[:, indxunob].std(axis=0)) * initial_std[indxunob] + x_ens[:, indxunob].mean(axis=0)
if use_ensf == True:
      user = REVERSE_SDE(1000, x_ens_subarray, N, y_obs, obs_sigma, nobs, 1., indx_indxob_linear, initial_std[indxob])
      x_ens_analysis_subarray = user.reverse_SDE()
x_ens_analysis[:, indxob] = x_ens_analysis_subarray
x_ens_analysis[:, indxunob] = x_ens[:, indxunob]
#####
