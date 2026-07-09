import numpy as np
import logging
from galario.double import get_image_size, chi2Profile, deg, arcsec, chi2Image
import pandas as pd

#########################
### ring_1blob with new models and priors discussed with Doug
#########################

def gaussring_prof(peak_ring, sigma_ring, rad_ring, radius, dxy):
    return 10**peak_ring * np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2) * (dxy**2)

def gaussblob(peak, sigma, xx, yy, xoff, yoff, dxy):
    return 10**peak * np.exp((-1/2) * (((xx-xoff)/sigma)**2 + ((yy-yoff)/sigma)**2)) * (dxy**2)

PA_RAD = 18.97 * deg
COS_PA = np.cos(PA_RAD)
SIN_PA = np.sin(PA_RAD)

def new_model_ring_1blob(peak, sigma, rad, inc, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy, cos_pa, sin_pa):

    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    #best guess position angle applied
    xpa_rg = xx*cos_pa + yy*sin_pa
    ypa_rg = -xx*sin_pa + yy*cos_pa

    #ring
    xinc = xpa_rg/np.cos(inc)
    radius_vec = np.hypot(xinc, ypa_rg)

    #blob_1
    #dist_b1 = dist_re_b1 * rad
    
    #xdot_b1 = dist_b1 * np.sin(ang_b1) * np.cos(inc)
    xdot_b1 = dist_b1 * np.sin(ang_b1)
    ydot_b1 = dist_b1 * np.cos(ang_b1)

    r_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    b1_model = gaussblob(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1, dxy)
    
    return r_model + b1_model

def new_model_ring_1blob_setup(pars, args, vis_data):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1 = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec

    inclination *= deg
    #pa_guess = 18.97 * deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    sigma_b1 *= arcsec

    dist_b1 *= arcsec
    ang_b1 *= deg

    model_img = new_model_ring_1blob(peak, sigma, ring_rad, inclination, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy, COS_PA, SIN_PA) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

#########################
### Radial Gaussian Ring
#########################

def radial_gaussian_ring(pars, args, vis_data):

    """
    Calculates chi-squared for a Gaussian ring intensity profile.

    Converts input parameters from log-space and angular units (arcsec/deg) 
    to linear space and radians for the visibility-plane calculation.

    Args:
        pars (np.ndarray):1D array of model parameters to be sampled.
            [peak (log(Jy/sr)), sigma (arcsec), ring_rad (arcsec), 
             inc (deg), PA (deg), dRA (arcsec), dDec (arcsec)]
        args (tuple): Fixed data and constants required for the model.
            (start, step, numsteps, nxy, dxy, u, v, re, im, w)

    Returns:
        float: The chi-squared value calculated via chi2Profile from GALARIO.
    """

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert from log to real space
    peak = 10**peak   

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec
    start *= arcsec
    step *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    #define gaussian profile
    radius = np.linspace(start, start + numsteps*step, numsteps)

    rad_prof = peak*np.exp((-1/2)*((radius-ring_rad)/sigma)**2)

    #compute the chi-square of the model
    chi2 = chi2Profile(rad_prof, start, step, nxy, dxy, u, v, re, im, w, inc=inclination, PA=posangle, dRA=dRA, dDec=dDec)

    return chi2

#########################
### 2D Gaussian Ring
#########################

def gaussring_model(peak, sigma, rad, inc, nxy, dxy):
    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    xinc = xx/np.cos(inc)
    radius_vec = np.hypot(xinc, yy)

    ring_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    
    return ring_model

def twod_gaussring(pars, args, vis_data):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    model_img = gaussring_model(peak, sigma, ring_rad, inclination, nxy, dxy) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

#########################
### 2D Gaussian Ring + 1 blob
#########################

def gaussring_1blob_model(peak, sigma, rad, inc, peak_b, sigma_b, dist, ang, nxy, dxy):
    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    #ring
    xinc = xx/np.cos(inc)
    radius_vec = np.hypot(xinc, yy)

    #blob
    xdot = dist * np.sin(ang) * np.cos(inc)
    ydot = dist * np.cos(ang)

    ring_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    blob_model = gaussblob(peak_b, sigma_b, xx, yy, xdot, ydot, dxy)
    
    return ring_model + blob_model

def twod_gaussring_1blob(pars, args, vis_data):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b, sigma_b, dist, ang = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    sigma_b *= arcsec
    dist *= arcsec

    ang *= deg

    model_img = gaussring_1blob_model(peak, sigma, ring_rad, inclination, peak_b, sigma_b, dist, ang, nxy, dxy) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

#########################
### Fixed 2D Gaussian Ring + 1 blob
#########################

def twod_fixring_1blob(pars, args, vis_data):

    peak_b, sigma_b, dist, ang = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma = 0.94 * arcsec
    ring_rad = 6.97 * arcsec

    inclination = 61.98 * deg
    posangle = 13.84 * deg

    dRA = 0.29 * arcsec
    dDec = 0.38 * arcsec

    sigma_b *= arcsec
    dist *= arcsec

    ang *= deg

    model_img = gaussring_1blob_model(6.54, sigma, ring_rad, inclination, peak_b, sigma_b, dist, ang, nxy, dxy) 
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

#########################
### 2D Gaussian Ring + 2 blob in NE region
#########################

def gaussring_2blob_model(peak, sigma, rad, inc, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2, nxy, dxy):
    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    #ring
    xinc = xx/np.cos(inc)
    radius_vec = np.hypot(xinc, yy)

    #blob_1
    xdot1 = dist1 * np.sin(ang1) * np.cos(inc)
    ydot1 = dist1 * np.cos(ang1)

    #blob_2
    xdot2 = dist2 * np.sin(ang2) * np.cos(inc)
    ydot2 = dist2 * np.cos(ang2)

    ring_model = gaussring_prof(peak, sigma, rad, radius_vec, dxy)
    blob1_model = gaussblob(peak_b1, sigma_b1, xx, yy, xdot1, ydot1, dxy)
    blob2_model = gaussblob(peak_b2, sigma_b2, xx, yy, xdot2, ydot2, dxy)
    
    return ring_model + blob1_model + blob2_model

def twod_gaussring_2blob(pars, args, vis_data):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2 = pars
    start, step, numsteps, nxy, dxy = args
    u, v, re, im, w = vis_data

    # convert to radians
    sigma *= arcsec
    ring_rad *= arcsec

    inclination *= deg
    posangle *= deg

    dRA *= arcsec
    dDec *= arcsec

    sigma_b1 *= arcsec
    dist1 *= arcsec
    ang1 *= deg

    sigma_b2 *= arcsec
    dist2 *= arcsec
    ang2 *= deg

    model_img = gaussring_2blob_model(peak, sigma, ring_rad, inclination, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2, nxy, dxy)
    chi2 = chi2Image(model_img, dxy, u, v, re, im, w, dRA=dRA, dDec=dDec, PA=posangle, origin='lower')

    return chi2

##################################################################################################################

def model_prof(pars, args, vis_data, fittype):

    """
    Routes the parameter evaluation to the appropriate physical model profile.

    Args:
        pars (np.ndarray): 1D array of model parameters to be sampled.
        args (tuple): Fixed data and constants required for the model.
        fittype (str): The model configuration identifier.

    Returns:
        float: The chi-squared value for the selected model.
    """

    if fittype == 'gaussring':
        chi2 = radial_gaussian_ring(pars, args, vis_data)

    elif fittype == 'twodgaussring':
        chi2 = twod_gaussring(pars, args, vis_data)

    elif fittype == 'twodgaussring_blob':
        chi2 = twod_gaussring_1blob(pars, args, vis_data)

    elif fittype == 'fixring_blob':
        chi2 = twod_fixring_1blob(pars, args, vis_data)

    elif fittype == 'twodring_2blob_ne':
        chi2 = twod_gaussring_2blob(pars, args, vis_data)

    elif fittype == 'blob_radex15':
        chi2 = new_model_ring_1blob_setup(pars, args, vis_data)

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return chi2

###################################

m95_ra = (10+43/60+57.73/3600)*15
logging.info(m95_ra, 'deg')
m95_dec = 11+42/60+13.3/3600
logging.info(m95_dec, 'deg')

pc_ra = (10+43/60+57.7330/3600)*15
logging.info(pc_ra, 'deg')
pc_dec = 11+42/60+12.9996/3600
logging.info(pc_dec, 'deg')

def cal_offset(ymc_ra, ymc_dec, mode):

    if mode == 'm95':

        ref_ra = m95_ra
        ref_dec = m95_dec

    elif mode == "phc":

        ref_ra = pc_ra
        ref_dec = pc_dec

    else:

        logging.warning('Pick m95 or phc')

    ###    

    dec_mid = (ymc_dec + ref_dec)/2
    ra_off = (ymc_ra - ref_ra) * np.cos(np.deg2rad(ref_dec)) * 3600

    dec_off = (ymc_dec - ref_dec) * 3600

    return ra_off, dec_off

def radec_to_rang(ymc_ra, ymc_dec):

    ra_off, dec_off = cal_offset(ymc_ra, ymc_dec, 'm95')
    
    logging.info(ra_off, dec_off)

    ang = np.rad2deg(np.arctan2(-ra_off, dec_off)) % 360

    logging.info(ang)

    dist_off = np.sqrt((ra_off**2)+(dec_off**2))

    logging.info(dist_off)

    logging.info(ang, dist_off)

    return ang, dist_off

def prior_guess_1blob(df, ymc_id):

    ang_b1, dist_b1 = radec_to_rang(df['ra (deg)'][ymc_id], df['dec (deg)'][ymc_id])
    
    #pars_jb = [6.43, 1.06, 6.8, 60.85, 15, 3, 3, df['log (peak93 (jy/sr))'][ymc_id], df['sigma (arcsec)'][ymc_id], dist_re_b1, ang_b1]

    return ang_b1, dist_b1

def model_init(fittype):

    """
    Provides starting positions and prior boundaries for a specified model.

    Args:
        fittype (str): The model configuration identifier.

    Returns:
        tuple: A 2-element tuple containing:
            - init_guess (list[float]): The initial guess starting values for walkers.
            - model_fits_ranges (list[list[float]]): The [min, max] prior bounds.
    """

    if fittype == 'gaussring':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'twodgaussring':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'twodgaussring_blob':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0, 6, 1, 7, 15]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5], [-5, 15], [0, 10], [0, 20], [0,90]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b, sigma_b, distance, angle = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'fixring_blob':
        model_fits_initial_guesses = [6, 1, 7, 15]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0,90]]
        logging.info('peak_b, sigma_b, distance, angle = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'twodring_2blob_ne':
        model_fits_initial_guesses = [6, 1, 7, 60, 15, 0, 0, 6, 1, 8, 15, 6, 1, 7, 20]
        model_fits_ranges = [[-5, 15], [0, 10], [0, 20], [0, 90], [0, 180],[-5, 5], [-5, 5], [5, 10], [0, 2], [5, 10], [0,90], [5, 10], [0, 2], [5, 10], [0,90]]
        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, distance1, angle1, peak_b2, sigma_b2, distance2, angle2 = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    elif fittype == 'blob_radex15':

        # Load the CSV file into a DataFrame
        df = pd.read_csv('ymc_prior_full_err.csv')
        
        ang_b1, dist_b1 = prior_guess_1blob(df, 11)

        model_fits_initial_guesses_unround = [6.43, 1.06, 6.8, 60.85, 0, 0, 0, df['log (peak93 (jy/sr))'][11], df['sigma (arcsec)'][11], dist_b1, ang_b1]
        model_fits_ranges_unround = [[0, 10], [0.8, 1.2], [6.4, 7.2], [58.0, 63.0], [-10, 10], [-2, 2], [-2, 2],
                            [df['log (peak93 (jy/sr))'][11]-(20*df['log (peak93 (jy/sr))_err'][11]), df['log (peak93 (jy/sr))'][11]+(20*df['log (peak93 (jy/sr))_err'][11])], 
                            [0, df['sigma (arcsec)'][11]+(20*df['sigma_err (arcsec)'][11])], 
                            [dist_b1-(10*df['sigma (arcsec)'][11]), dist_b1+(10*df['sigma (arcsec)'][11])],
                            [ang_b1-10, ang_b1+10]]

        model_fits_initial_guesses = [round(num, 2) for num in model_fits_initial_guesses_unround]
        model_fits_ranges = [[round(num, 2) for num in sublist] for sublist in model_fits_ranges_unround]

        logging.info('peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1 = pars')
        logging.info('start, step, numsteps, nxy, dxy = args')
        logging.info('u, v, re, im, w = vis_data')

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return model_fits_initial_guesses, model_fits_ranges