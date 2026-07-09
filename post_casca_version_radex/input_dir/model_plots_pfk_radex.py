import numpy as np
import logging
from galario.double import get_image_size, chi2Profile, deg, arcsec, chi2Image

#########################
### ring_1blob with new models and priors discussed with Doug
#########################

def radial_gaussian_ring_plot(peak_ring, sigma_ring, rad_ring, radius):
    return 10**peak_ring*np.exp((-1/2)*((radius-rad_ring)/sigma_ring)**2)

def gauss_blob_plot(peak_blob, sigma_blob, xx, yy, xoff, yoff):
    return 10**peak_blob * np.exp((-1/2) * (((xx-xoff)/sigma_blob)**2 + ((yy-yoff)/sigma_blob)**2))

def new_model_ring_1blob(peak, sigma, rad, inc, pa_guess, pa_gal, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy):

    #Initialize the image plane
    image_size = nxy * dxy
    x = np.linspace(-image_size/2, image_size/2, nxy)
    y = np.linspace(-image_size/2, image_size/2, nxy)
    xx, yy = np.meshgrid(x, y)

    #apply deltaRA and deltaDec
    xx_shifted = xx + (dra) #convert to arcsec, then pix (to match xx system) #move left 
    yy_shifted = yy - (ddec) #move up

    #then apply the pa determined from galario
    xpa = xx_shifted*np.cos(np.deg2rad(pa_gal)) + yy_shifted*np.sin(np.deg2rad(pa_gal))
    ypa = -xx_shifted*np.sin(np.deg2rad(pa_gal)) + yy_shifted*np.cos(np.deg2rad(pa_gal))

    #best guess position angle applied
    xpa_rg = xpa*np.cos(np.deg2rad(pa_guess)) + ypa*np.sin(np.deg2rad(pa_guess))
    ypa_rg = -xpa*np.sin(np.deg2rad(pa_guess)) + ypa*np.cos(np.deg2rad(pa_guess))

    #ring
    xinc = xpa_rg/np.cos(np.deg2rad(inc))
    radius_vec = np.hypot(xinc, ypa_rg)

    #blob_1
    #dist_b1 = dist_re_b1 * rad
    
    #xdot_b1 = dist_b1 * np.sin(ang_b1) * np.cos(inc)
    xdot_b1 = dist_b1 * np.sin(np.deg2rad(ang_b1))
    ydot_b1 = dist_b1 * np.cos(np.deg2rad(ang_b1))
    
    r_model = radial_gaussian_ring_plot(peak, sigma, rad, radius_vec)
    b1_model = gauss_blob_plot(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1)
    model_jysr = r_model+b1_model
    
    return model_jysr

def new_model_ring_1blob_setup_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1 = pars
    nxy, dxy = args

    pa_guess = 18.97

    ring_model = new_model_ring_1blob(peak, sigma, ring_rad, inclination, pa_guess, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, nxy, dxy) 

    return ring_model

#########################
### Gaussian Ring
#########################

def make_model_ring(npix, pixscale, peak, sigma, ringrad, inc, pa, dra, ddec):

    #The inputs that are the product of the emcee fitting will come out in the units in which I put them into the fitting chain.
    #This means that the peak is in Jy/sr, the sigma and ringrad are in arcseconds, and inc, pa, deltaRA and deltaDec are in degrees. 

    image_size = npix * pixscale
    x = np.linspace(-image_size/2, image_size/2, npix)
    y = np.linspace(-image_size/2, image_size/2, npix)
    xx, yy = np.meshgrid(x, y)

    #apply deltaRA and deltaDec
    xx_shifted = xx + (dra) #convert to arcsec, then pix (to match xx system) #move left 
    yy_shifted = yy - (ddec) #move up

    xpa = xx_shifted*np.cos(np.deg2rad(pa)) + yy_shifted*np.sin(np.deg2rad(pa))
    ypa = -xx_shifted*np.sin(np.deg2rad(pa)) + yy_shifted*np.cos(np.deg2rad(pa))
    
    xinc = xpa/np.cos(np.deg2rad(inc))

    rad = np.hypot(xinc, ypa)
    model_jysr = radial_gaussian_ring_plot(peak, sigma, ringrad, rad)
    return model_jysr #leave in Jy/sr here. I will convert Jy/bm to Jy/sr to match later.

def gaussring_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec = pars
    nxy, dxy = args

    ring_model = make_model_ring(nxy, dxy, peak, sigma, ring_rad, inclination, posangle, dRA, dDec)

    return ring_model

#########################
### Gaussian Ring + 1 blob
#########################

def make_model_blob(npix, pixscale, peak, sigma, ringrad, inc, pa, dra, ddec, peak_b, sigma_b, dist, ang):

    #The inputs that are the product of the emcee fitting will come out in the units in which I put them into the fitting chain.
    #This means that the peak is in Jy/sr, the sigma and ringrad are in arcseconds, and inc, pa, deltaRA and deltaDec are in degrees. 

    image_size = npix * pixscale
    x = np.linspace(-image_size/2, image_size/2, npix)
    y = np.linspace(-image_size/2, image_size/2, npix)
    xx, yy = np.meshgrid(x, y)

    #apply deltaRA and deltaDec
    xx_shifted = xx + (dra) #convert to arcsec, then pix (to match xx system) #move left 
    yy_shifted = yy - (ddec) #move up

    xpa = xx_shifted*np.cos(np.deg2rad(pa)) + yy_shifted*np.sin(np.deg2rad(pa))
    ypa = -xx_shifted*np.sin(np.deg2rad(pa)) + yy_shifted*np.cos(np.deg2rad(pa))
    
    #ring
    xinc = xpa/np.cos(np.deg2rad(inc))
    rad = np.hypot(xinc, ypa)

    #blob
    xdot = dist * np.sin(np.deg2rad(ang)) * np.cos(np.deg2rad(inc))
    ydot = dist * np.cos(np.deg2rad(ang))

    ring_model = radial_gaussian_ring_plot(peak, sigma, ringrad, rad)
    blob_model = gauss_blob_plot(peak_b, sigma_b, xpa, ypa, xdot, ydot)
    model_jysr = ring_model+blob_model
    
    return model_jysr #leave in Jy/sr here. I will convert Jy/bm to Jy/sr to match. 

def gaussring_blob_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b, sigma_b, dist, ang = pars
    nxy, dxy = args

    ring_model = make_model_blob(nxy, dxy, peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b, sigma_b, dist, ang)

    return ring_model

#########################
### Fixed Ring + 1 blob
#########################

def fixring_blob_plot(pars, args):

    peak_b, sigma_b, dist, ang = pars
    nxy, dxy = args

    ring_model = make_model_blob(nxy, dxy, 6.54, 0.94, 6.97, 61.98, 13.84, 0.29, 0.38, peak_b, sigma_b, dist, ang)

    return ring_model

#########################
### Gaussian Ring + 2 blob
#########################

def make_model_2blob(npix, pixscale, peak, sigma, ringrad, inc, pa, dra, ddec, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2):

    #The inputs that are the product of the emcee fitting will come out in the units in which I put them into the fitting chain.
    #This means that the peak is in Jy/sr, the sigma and ringrad are in arcseconds, and inc, pa, deltaRA and deltaDec are in degrees. 

    image_size = npix * pixscale
    x = np.linspace(-image_size/2, image_size/2, npix)
    y = np.linspace(-image_size/2, image_size/2, npix)
    xx, yy = np.meshgrid(x, y)

    #apply deltaRA and deltaDec
    xx_shifted = xx + (dra) #convert to arcsec, then pix (to match xx system) #move left 
    yy_shifted = yy - (ddec) #move up

    xpa = xx_shifted*np.cos(np.deg2rad(pa)) + yy_shifted*np.sin(np.deg2rad(pa))
    ypa = -xx_shifted*np.sin(np.deg2rad(pa)) + yy_shifted*np.cos(np.deg2rad(pa))
    
    #ring
    xinc = xpa/np.cos(np.deg2rad(inc))
    rad = np.hypot(xinc, ypa)

    #blob_1
    xdot1 = dist1 * np.sin(np.deg2rad(ang1)) * np.cos(np.deg2rad(inc))
    ydot1 = dist1 * np.cos(np.deg2rad(ang1))

    #blob_2
    xdot2 = dist2 * np.sin(np.deg2rad(ang2)) * np.cos(np.deg2rad(inc))
    ydot2 = dist2 * np.cos(np.deg2rad(ang2))

    ring_model = radial_gaussian_ring_plot(peak, sigma, ringrad, rad)
    blob1_model = gauss_blob_plot(peak_b1, sigma_b1, xpa, ypa, xdot1, ydot1)
    blob2_model = gauss_blob_plot(peak_b2, sigma_b2, xpa, ypa, xdot2, ydot2)
    model_jysr = ring_model+blob1_model+blob2_model
    
    return model_jysr

def gaussring_2blob_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2 = pars
    nxy, dxy = args

    ring_model = make_model_2blob(nxy, dxy, peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist1, ang1, peak_b2, sigma_b2, dist2, ang2)

    return ring_model

##################################################################################################################

def model_plot(pars, args, fittype):
    
    if fittype == 'gaussring':
        ring_model = gaussring_plot(pars, args)

    elif fittype == 'twodgaussring':
        ring_model = gaussring_plot(pars, args)

    elif fittype == 'twodgaussring_blob':
        ring_model = gaussring_blob_plot(pars, args)

    elif fittype == 'fixring_blob':
        ring_model = fixring_blob_plot(pars, args)
    
    elif fittype == 'twodring_2blob_ne':
        ring_model = gaussring_2blob_plot(pars, args)

    elif fittype == 'blob_radex15':
        ring_model = new_model_ring_1blob_setup_plot(pars, args)

    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')

    return ring_model

def model_label(fittype):

    if fittype=='gaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    elif fittype=='twodgaussring':
        #label = ["Peak", "$\sigma$", r"R$_{ring}$", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Ring Rad", "Inc", "PA", "Offset RA", "Offset Dec"]
    elif fittype=='twodgaussring_blob':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B. Peak", "B. Width", "Dist", "Angle"]
    elif fittype=='fixring_blob':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["B. Peak", "B. Width", "Dist", "Angle"]
    elif fittype=='twodring_2blob_ne':
        #label = ["Peak", "Width", "Offset", "Inc", "PA", r"$\Delta$RA", r"$\Delta$Dec"]
        label = ["Peak", "Width", "Offset", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak", "B1. Width", "Dist1", "Angle1", "B2. Peak", "B2. Width", "Dist2", "Angle2"]
    elif fittype == 'blob_radex15':
        label = ["Peak", "Width", "Radius", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak", "B1. Width", "B1. Dist", "B1. Angle"]
    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')
    
    return label