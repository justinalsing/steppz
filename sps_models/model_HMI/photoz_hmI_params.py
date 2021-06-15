import time, sys

import numpy as np
from sedpy.observate import load_filters

from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.io import write_results as writer
from astropy.cosmology import WMAP9
from scipy.special import gamma, gammainc

# --------------
# Model Definition
# --------------

def build_model(**extras):
    """Construct a model.  This method defines a number of parameter
    specification dictionaries and uses them to initialize a
    `models.sedmodel.SedModel` object.

    :param add_dust: (optional, default: False)
        Switch to add (fixed) parameters relevant for dust emission.

    """
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, sedmodel

    # --- Get a basic delay-tau SFH parameter set. ---
    # This has 5 free parameters:
    #   "mass", "logzsol", "dust2", "tage", "tau"
    # And two fixed parameters
    #   "zred"=0.1, "sfh"=4
    model_params = TemplateLibrary["parametric_sfh"]

    # First, redshift
    model_params['zred']['isfree'] = True
    model_params['zred']['prior'] = priors.TopHat(mini=0.0,maxi=9.0)

    # Mass
    # note by default this is *total mass formed*, not stellar mass
    model_params['mass']['prior'] = priors.LogUniform(mini=1e6, maxi=1e13)
    model_params['imf_type']['init'] = 1 # assume a Chabrier IMF

    # Metallicity
    # prior simply reflects allowed template range for now.
    model_params['logzsol']['prior'] = priors.TopHat(mini=-1.98,maxi=0.19)
    # Here we specify the metallicity interpolation to use a triangular weighting scheme
    # this avoids the challenges of interpolating over non-linear and non-monotonic
    # effect of stellar metallicity
    model_params['pmetals'] = {'N': 1,
                               'init': -99,
                               'isfree': False}

    # choose a delayed-tau SFH here
    # tau limits set following Wuyts+11
    model_params['sfh']['init'] = 4 # delayed-tau
    model_params["tau"]["prior"] = priors.LogUniform(mini=0.316, maxi=100)

    model_params['peraa'] = {'N': 1,
                     'isfree': False,
                     'init': True}

    # Introduce new variable, `tmax`
    # this ranges from 0 to 1, and allows us to set tage_max = f(z)
    # This assumes a WMAP9 cosmology for the redshift --> age of universe conversion
    def tmax_to_tage(tmax=None,zred=None,**kwargs):
        return WMAP9.age(zred).value*tmax # in Gyr
    model_params['tage']['isfree'] = False
    model_params['tage']['depends_on'] = tmax_to_tage
    model_params['tmax'] = {'N': 1,
                            'isfree': True,
                            'init': 0.5,
                            'prior': priors.TopHat(mini=0.73e-4, maxi=1.0)}
                            # minimum here is set by following requirements:
                            # this *can't* be zero
                            # too large and we're banning very young galaxies
                            # the oldest galaxy this bans is (13.7 Gyr * min) = 1 Myr

    # Dust attenuation. We choose a standard dust screen model here.
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=4.0)

    # Young stars and nebular regions get extra attenuation
    def dust2_to_dust1(dust2=None,**kwargs):
        return dust2
    model_params['dust1'] = {'N': 1,
                             'init': 0.5,
                             'isfree': False,
                             'depends_on': dust2_to_dust1}

    # We let the dust attenuation law vary according to the amount of dust
    def dust2_to_dustindex(dust2=None,**kwargs):
        return -0.095 + 0.111*dust2 - 0.0066*dust2**2
    model_params['dust_type']['init'] = 4   # Calzetti with power-law modification
    model_params['dust_index'] = {'N': 1,
                                  'init': 0.0,
                                  'isfree': False,
                                  'depends_on': dust2_to_dustindex}

    # Nebular emission
    model_params.update(TemplateLibrary["nebular"])

    # Gas metallicity == stellar metallicity
    def logzsol_to_gaslogz(logzsol=None,**kwargs):
        return logzsol
    model_params['gas_logz']['isfree'] = False
    model_params['gas_logz']['depends_on'] = logzsol_to_gaslogz

    # Allow ionization parameter to vary based on sSFR
    def ssfr_to_gaslogu(tmax=None,tau=None,zred=None,**kwargs):
        # calculate sSFR
        tage = tmax_to_tage(tmax=tmax,zred=zred)
        ssfr = (tage/tau**2) * np.exp(-tage/tau) / (gamma(2) * gammainc(2, tage/tau)) * 1e-9

        # above calculation is missing a factor of (stellar mass / total mass formed)
        # this is a pain to estimate and typically is (0.64-1.0)
        # take a rough estimate here to split the difference
        # this is an ok approximation since it typically varies by ~7 orders of magnitude
        ssfr *= 0.82

        # now plug into relationship from Kaasinen+18
        gas_logu = np.log10(ssfr)*0.3125 + 0.9982   

        return np.clip(gas_logu,-4.0,-1.0)  # stay within allowed range
    model_params['gas_logu']['isfree'] = False
    model_params['gas_logu']['depends_on'] = ssfr_to_gaslogu

    # Make sure to add nebular emission to the spectrum
    # this takes extra runtime, but will be important when emulating the spectrum
    model_params['nebemlineinspec']['init'] = True

    # We don't need to produce or emulate the infrared.
    model_params['add_dust_emission'] = {'N': 1, 'init': False, 'isfree': False}

    return sedmodel.SedModel(model_params)

def build_obs(objid=0,**kwargs):
    """Load photometry from an ascii file.  Assumes the following columns:
    `objid`, `filterset`, [`mag0`,....,`magN`] where N >= 11.  The User should
    modify this function (including adding keyword arguments) to read in their
    particular data format and put it in the required dictionary.

    :param objid:
        The object id for the row of the photomotery file to use.  Integer.
        Requires that there be an `objid` column in the ascii file.

    :param luminosity_distance: (optional)
        The Johnson 2013 data are given as AB absolute magnitudes.  They can be
        turned into apparent magnitudes by supplying a luminosity distance.

    :returns obs:
        Dictionary of observational data.
    """
    # Writes your code here to read data.  Can use FITS, h5py, astropy.table,
    # sqlite, whatever.
    # e.g.:
    # import astropy.io.fits as pyfits
    # catalog = pyfits.getdata(phottable)

    from prospect.utils.obsutils import fix_obs

    # Name the filters
    kids = ['omegacam_'+n for n in ['u','g','r','i']]
    viking = ['VISTA_'+n for n in ['Z','Y','J','H','Ks']]
    filternames = kids+viking


    # fake fluxes!
    mags = np.zeros(len(filternames)) + 1e-9

    # Build output dictionary.
    obs = {}
    # This is a list of sedpy filter objects.    See the
    # sedpy.observate.load_filters command for more details on its syntax.
    obs['filters'] = load_filters(filternames)
    # This is a list of maggies, converted from mags.  It should have the same
    # order as `filters` above.
    obs['maggies'] = np.squeeze(10**(-mags/2.5))
    # Hack, should use real flux uncertainties
    obs['maggies_unc'] = obs['maggies'] * 0.07
    # Here we mask out any NaNs or infs
    obs['phot_mask'] = np.isfinite(np.squeeze(mags))
    # We have no spectrum.
    obs['wavelength'] = None
    obs['spectrum'] = None

    # Add unessential 'bonus' info.  This will be stored in output
    obs['objid'] = objid

    # This ensures all required keys are present and adds some extra useful info
    obs = fix_obs(obs)

    return obs

# --------------
# SPS Object
# --------------
def build_sps(zcontinuous=2, compute_vega_mags=False, **extras):
    from prospect.sources import CSPSpecBasis
    sps = CSPSpecBasis(zcontinuous=zcontinuous,
                       compute_vega_mags=compute_vega_mags)
    return sps

# -----------------
# Noise Model
# ------------------

def build_noise(**extras):
    return None, None

# -----------
# Everything
# ------------
def build_all(**kwargs):

    return (build_obs(**kwargs), build_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))


if __name__=='__main__':

    # - Parser with default arguments -
    parser = prospect_args.get_parser()
    # - Add custom arguments -
    parser.add_argument('--object_redshift', type=float, default=0.0,
                        help=("Redshift for the model"))
    parser.add_argument('--objid', type=int, default=0,
                        help="zero-index row number in the table to fit.")

    args = parser.parse_args()
    run_params = vars(args)
    obs, model, sps, noise = build_all(**run_params)

    run_params["sps_libraries"] = sps.ssp.libraries
    run_params["param_file"] = __file__

    print(model)


    #hfile = setup_h5(model=model, obs=obs, **run_params)
    hfile = "{0}_{1}_mcmc.h5".format(args.outfile, int(time.time()))
    output = fit_model(obs, model, sps, noise, **run_params)

    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1],
                      sps=sps)

    try:
        hfile.close()
    except(AttributeError):
        pass

