import os
import sys
from astropy.cosmology import Planck15
from scipy.integrate import cumtrapz
"""Import the basics; numpy, pandas, matplotlib et al."""
import pandas as pd
import numpy as np


"""This module hosts the definitions of various SFH priors, as detailed in Leja+2019"""

def continuity_prior_timebins(nbins, tuniv):

    """
    An implementation of the age binning to be used for the continuity prior definition from Leja+2019

    ::param nbins: number of time bins where an observation of SFR is to be made
    ::param tuniv: the age of the universe at the redshift of the galaxy, in Gyr

    Important: The method implements bins in lookback time

    """

    """
    De facto limits imposed on the age binning:
    
    The first lookback time bin is set to be 0-30 Myr,
    The second lookback time bin is set to be 30-100 Myr,
    The final lookback time bin is set to be 0.85*tuniv-tuniv

    The remaining time bins are split logarithmically evenly in time

    """
    """Times in years"""
    min_tbin_minedge = 0.0
    min_tbin_maxedge = 3 * np.power(10, 7)
    min2_tbin_maxedge = np.power(10, 8)

    max_tbin_minedge = 0.85 * tuniv * np.power(10, 9)
    max_tbin_maxedge = tuniv * np.power(10, 9)

    """Split the remaining time range evenly on a log scale"""
    tbins_logeven = np.linspace(start=np.log10(min2_tbin_maxedge), stop=np.log10(max_tbin_minedge), num=nbins - 2)

    age_minedge_log = [min_tbin_minedge, np.log10(min_tbin_maxedge)] + tbins_logeven.tolist() + [np.log10(max_tbin_maxedge)]
    agebins_fsps = np.array([age_minedge_log[:-1], age_minedge_log[1:]])

    return age_minedge_log, agebins_fsps

# def compute_sfr(data_df, num_bins):
#
#     """This function computes the SFR per bin per object"""
#     # ssfr_array = np.zeros(shape=data_df.shape[0])
#     """Initialize a dictionary that will host the SFR's per bin"""
#     norm_sfr_arr = np.zeros(shape=(data_df.shape[0], num_bins))
#     time_grid_arr = np.zeros(shape=(data_df.shape[0], num_bins))
#
#     for obj_ind_, obj_ in enumerate(tqdm(data_df['ID'].values)):
#         """tuniv: Age of the universe in Gyr at redshift z"""
#         tuniv = Planck15.age(data_df.loc[data_df['ID'] == obj_]['z'].item()).value
#         log_time_grid, fsps_grid = continuity_prior_timebins(num_bins, tuniv)
#         time_grid_yr = [np.power(10, log_t_) for log_t_ in log_time_grid[1:]]
#         time_grid_yr.insert(0, 0.0)
#         sfr_ratio_cols = [col_ for col_ in data_df.columns if 'logsfr_ratio' in col_]
#
#         # """Convert N to log10M"""
#         # log10M = (data_df.loc[data_df['ID'] == obj_]['N'].item() - distance_modulus(data_df.loc[data_df['ID'] == obj_]['z'].item())) / (-2.5)
#         # log10M.numpy() converts to flaoting variable
#         nonnorm_sfr = np.ones(num_bins)
#
#         """First, populate an array of non-normalized SFR's"""
#         for sf_bin_ in range(1, num_bins):
#             nonnorm_sfr[sf_bin_] = nonnorm_sfr[sf_bin_ - 1] * np.power(10, data_df.loc[data_df['ID'] == obj_][sfr_ratio_cols].values[0, sf_bin_ - 1])
#
#         """Normalize the non-norm SFR"""
#         norm_sfr = nonnorm_sfr / np.sum(nonnorm_sfr * (np.array(time_grid_yr[1:]) - np.array(time_grid_yr[0:-1])))
#         # """Populate the sfr_dict"""
#         # sfr_dict['ID'].append(obj_)
#         # for sfr_bin_ind_, sfr_bin_ in zip(range(1, num_bins+1), norm_sfr):
#         #     sfr_dict['sfr_bin'+str(sfr_bin_ind_)].append(sfr_bin_)
#         norm_sfr_arr[obj_ind_, :] = norm_sfr
#         time_grid_arr[obj_ind_, :] = time_grid_yr[1:]
#
#     # data_df.insert(loc=, column='log_sSFR', value=pd.Series(ssfr_array.tolist()))
#     # sfr_df = pd.DataFrame.from_dict(sfr_dict)
#
#     return norm_sfr_arr, time_grid_arr


def compute_sfr(data_df, num_bins):

    """This function computes the SFR per bin per object"""
    # ssfr_array = np.zeros(shape=data_df.shape[0])
    """Initialize a dictionary that will host the SFR's per bin"""
    norm_sfr_arr = np.zeros(shape=(data_df.shape[0], num_bins))
    time_grid_arr = np.zeros(shape=(data_df.shape[0], num_bins))

    # for obj_ind_, obj_ in enumerate(tqdm(data_df['ID'].values)):
    """tuniv: Age of the universe in Gyr at redshift z"""
    tuniv = Planck15.age(data_df['z'].item()).value
    log_time_grid, fsps_grid = continuity_prior_timebins(num_bins, tuniv)
    time_grid_yr = [np.power(10, log_t_) for log_t_ in log_time_grid[1:]]
    time_grid_yr.insert(0, 0.0)
    # sfr_ratio_cols = [col_ for col_ in data_df.columns if 'logsfr_ratio' in col_]
    sfr_ratio_cols = ['logsfr_ratio'+str(n) for n in range(1, num_bins)]

    # """Convert N to log10M"""
    # log10M = (data_df.loc[data_df['ID'] == obj_]['N'].item() - distance_modulus(data_df.loc[data_df['ID'] == obj_]['z'].item())) / (-2.5)
    # log10M.numpy() converts to flaoting variable
    nonnorm_sfr = np.ones(num_bins)

    """First, populate an array of non-normalized SFR's"""
    for sf_bin_ in range(1, num_bins):
        nonnorm_sfr[sf_bin_] = nonnorm_sfr[sf_bin_ - 1] * np.power(10, data_df[sfr_ratio_cols].values[sf_bin_ - 1])

    """Normalize the non-norm SFR"""
    norm_sfr = nonnorm_sfr / np.sum(nonnorm_sfr * (np.array(time_grid_yr[1:]) - np.array(time_grid_yr[0:-1])))
    # """Populate the sfr_dict"""
    # sfr_dict['ID'].append(obj_)
    # for sfr_bin_ind_, sfr_bin_ in zip(range(1, num_bins+1), norm_sfr):
    #     sfr_dict['sfr_bin'+str(sfr_bin_ind_)].append(sfr_bin_)
    # norm_sfr_arr[obj_ind_, :] = norm_sfr
    # time_grid_arr[obj_ind_, :] = time_grid_yr[1:]

    # data_df.insert(loc=, column='log_sSFR', value=pd.Series(ssfr_array.tolist()))
    # sfr_df = pd.DataFrame.from_dict(sfr_dict)

    return norm_sfr, time_grid_yr[1:]

def finegrid(arr, num_slices):
    finegr_ = []
    for ind_ in range(np.shape(arr)[0]-1):
        if ind_ == np.shape(arr)[0]-2:
            finegr_.extend(np.linspace(start=arr[ind_], stop=arr[ind_ + 1], endpoint=True, num=num_slices).tolist())
        else:
            finegr_.extend(np.linspace(start=arr[ind_], stop=arr[ind_ + 1], endpoint=False, num=num_slices).tolist())

    return np.array(finegr_)


def flexible_sfh(parameter_df):

    sfr, t_yr = compute_sfr(parameter_df, num_bins=7)
    """convert the time grid to Gyr"""
    t_yr = np.insert(t_yr, 0, 0.0)
    t = np.divide(t_yr, np.power(10, 9))

    slice_num = 100
    t_finegrid = finegrid(t, slice_num)

    # """slice the sfr and t-bins"""
    # t_finegrid_init = np.repeat(t, 100)
    # """need to add a very small value to each entry in t_finegrid to make the following fsps assert statement hold true
    # assert np.all(age[1:] > age[:-1]), 'Ages must be increasing.'
    # """
    # n_ = np.linspace(start=10**-11, stop=10**-10, num=t_finegrid_init.shape[0])
    # t_finegrid = t_finegrid_init + n_
    sfr_finegrid = np.repeat(sfr, slice_num)

    # plt.scatter(t_finegrid, sfr_finegrid)
    # plt.show()
    #
    # compute the metallicity history based on the integral of the SFH
    zt = np.concatenate([np.array([0.]), cumtrapz(x=t_finegrid, y=sfr_finegrid)])
    # re-normalize the metalicity history such that the mean metalicity over the last 10% of star formation = target Z
    zt = zt * parameter_df['Z'] / zt[-1]

    return t_finegrid, sfr_finegrid, zt


def pi_in_contprior(sfr_list, current_iter):
    print(current_iter)
    mult_ = 1
    while current_iter > 0:
        mult_ *= sfr_list[current_iter]
        current_iter -= 1

    return mult_


