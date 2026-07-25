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
### ring_1blob_axr
#########################

def gaussblob_axr_plot(peak, geo_mean, xx, yy, axr):

    sigmay = geo_mean / np.sqrt(axr)
    sigmax = sigmay * axr
    
    return 10**peak * np.exp((-1/2) * (((xx)/sigmax)**2 + ((yy)/sigmay)**2))

def new_model_ring_1blob_axr(peak, sigma, rad, inc, pa_guess, pa_gal, dra, ddec, peak_b1, geo_mean_b1, axr_b1, pa_b1, dist_b1, ang_b1, nxy, dxy):

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

    xoff_b1 = xpa - xdot_b1
    yoff_b1 = ypa - ydot_b1

    xpab_b1 = xoff_b1*np.cos(np.deg2rad(pa_b1)) + yoff_b1*np.sin(np.deg2rad(pa_b1))
    ypab_b1 = -xoff_b1*np.sin(np.deg2rad(pa_b1)) + yoff_b1*np.cos(np.deg2rad(pa_b1))

    r_model = radial_gaussian_ring_plot(peak, sigma, rad, radius_vec)
    b1_model = gaussblob_axr_plot(peak_b1, geo_mean_b1, xpab_b1, ypab_b1, axr_b1)
    model_jysr = r_model+b1_model
    
    return model_jysr

def new_model_ring_1blob_axr_setup_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, geo_mean_b1, axr_b1, pa_b1, dist_b1, ang_b1 = pars
    nxy, dxy = args

    pa_guess = 18.97

    ring_model = new_model_ring_1blob_axr(peak, sigma, ring_rad, inclination, pa_guess, posangle, dRA, dDec, peak_b1, geo_mean_b1, axr_b1, pa_b1, dist_b1, ang_b1, nxy, dxy) 

    return ring_model


#########################
### ring_3blob
#########################

def new_model_ring_3blob(peak, sigma, rad, inc, pa_guess, pa_gal, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy):

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

    #blob_2
    xdot_b2 = dist_b2 * np.sin(np.deg2rad(ang_b2))
    ydot_b2 = dist_b2 * np.cos(np.deg2rad(ang_b2))

    #blob_3
    xdot_b3 = dist_b3 * np.sin(np.deg2rad(ang_b3))
    ydot_b3 = dist_b3 * np.cos(np.deg2rad(ang_b3))

    r_model = radial_gaussian_ring_plot(peak, sigma, rad, radius_vec)
    b1_model = gauss_blob_plot(peak_b1, sigma_b1, xpa, ypa, xdot_b1, ydot_b1)
    b2_model = gauss_blob_plot(peak_b2, sigma_b2, xpa, ypa, xdot_b2, ydot_b2)
    b3_model = gauss_blob_plot(peak_b3, sigma_b3, xpa, ypa, xdot_b3, ydot_b3)
    model_jysr = r_model + b1_model + b2_model + b3_model

    return model_jysr

def new_model_ring_3blob_setup_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3 = pars
    nxy, dxy = args

    pa_guess = 18.97

    ring_model = new_model_ring_3blob(peak, sigma, ring_rad, inclination, pa_guess, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy) 

    return ring_model

#########################
### ring_3blob - fishing 1 blob
#########################

def new_model_ring_3blob_1fish(peak, sigma, rad, inc, pa_guess, pa_gal, dra, ddec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy):

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

    #blob_2
    xdot_b2 = dist_b2 * np.sin(np.deg2rad(ang_b2))
    ydot_b2 = dist_b2 * np.cos(np.deg2rad(ang_b2))

    #blob_3
    xdot_b3 = dist_b3 * np.sin(np.deg2rad(ang_b3))
    ydot_b3 = dist_b3 * np.cos(np.deg2rad(ang_b3))

    r_model = radial_gaussian_ring_plot(peak, sigma, rad, radius_vec)
    b1_model = gauss_blob_plot(peak_b1, sigma_b1, xx, yy, xdot_b1, ydot_b1)
    b2_model = gauss_blob_plot(peak_b2, sigma_b2, xx, yy, xdot_b2, ydot_b2)
    b3_model = gauss_blob_plot(peak_b3, sigma_b3, xpa, ypa, xdot_b3, ydot_b3)
    model_jysr = r_model + b1_model + b2_model + b3_model

    return model_jysr

def new_model_ring_3blob_1fish_setup_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b1, sigma_b1, peak_b2, sigma_b2, peak_b3, sigma_b3, dist_b3, ang_b3 = pars
    nxy, dxy = args

    pa_guess = 18.97

    dist_b1 = 6.2396616850087865
    ang_b1 = 176.49647334982126

    dist_b2 = 6.541229262283732
    ang_b2 = 161.38982927847752

    ring_model = new_model_ring_3blob_1fish(peak, sigma, ring_rad, inclination, pa_guess, posangle, dRA, dDec, peak_b1, sigma_b1, dist_b1, ang_b1, peak_b2, sigma_b2, dist_b2, ang_b2, peak_b3, sigma_b3, dist_b3, ang_b3, nxy, dxy) 

    return ring_model

#########################
### ring_3blob - 2 peak!
#########################

def gaussblob_2peak_plot(peak1, sigma1, peak2, sigma2, xx, yy, xoff, yoff):

    g1 = 10**peak1 * np.exp((-1/2) * (((xx-xoff)/sigma1)**2 + ((yy-yoff)/sigma1)**2))
    g2 = 10**peak2 * np.exp((-1/2) * (((xx-xoff)/sigma2)**2 + ((yy-yoff)/sigma2)**2))

    full_prof = g1+g2
    
    return full_prof

def new_model_ring_3blob_2peak(peak, sigma, rad, inc, pa_guess, pa_gal, dra, ddec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1, peak_b21, sigma_b21, peak_b22, sigma_b22, dist_b2, ang_b2, peak_b31, sigma_b31, peak_b32, sigma_b32, dist_b3, ang_b3, nxy, dxy):

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

    #blob_2
    xdot_b2 = dist_b2 * np.sin(np.deg2rad(ang_b2))
    ydot_b2 = dist_b2 * np.cos(np.deg2rad(ang_b2))

    #blob_3
    xdot_b3 = dist_b3 * np.sin(np.deg2rad(ang_b3))
    ydot_b3 = dist_b3 * np.cos(np.deg2rad(ang_b3))

    r_model = radial_gaussian_ring_plot(peak, sigma, rad, radius_vec)
    b1_model = gaussblob_2peak_plot(peak_b11, sigma_b11, peak_b12, sigma_b12, xpa, ypa, xdot_b1, ydot_b1, dxy)
    b2_model = gaussblob_2peak_plot(peak_b21, sigma_b21, peak_b22, sigma_b22, xpa, ypa, xdot_b2, ydot_b2, dxy)
    b3_model = gaussblob_2peak_plot(peak_b31, sigma_b31, peak_b32, sigma_b32, xpa, ypa, xdot_b3, ydot_b3, dxy)
    model_jysr = r_model + b1_model + b2_model + b3_model

    return model_jysr

def new_model_ring_3blob_2peak_setup_plot(pars, args):

    peak, sigma, ring_rad, inclination, posangle, dRA, dDec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1, peak_b21, sigma_b21, peak_b22, sigma_b22, dist_b2, ang_b2, peak_b31, sigma_b31, peak_b32, sigma_b32, dist_b3, ang_b3 = pars
    nxy, dxy = args

    pa_guess = 18.97

    ring_model = new_model_ring_3blob_2peak(peak, sigma, ring_rad, inclination, pa_guess, posangle, dRA, dDec, peak_b11, sigma_b11, peak_b12, sigma_b12, dist_b1, ang_b1, peak_b21, sigma_b21, peak_b22, sigma_b22, dist_b2, ang_b2, peak_b31, sigma_b31, peak_b32, sigma_b32, dist_b3, ang_b3, nxy, dxy) 

    return ring_model

######################################################################################################################

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

    elif fittype == 'blob_15axr':
        ring_model = new_model_ring_1blob_axr_setup_plot(pars, args)

    elif fittype == 'ring_3blob':
        ring_model = new_model_ring_3blob_setup_plot(pars, args)

    elif fittype == 'ring_3blob_1fish':
        ring_model = new_model_ring_3blob_1fish_setup_plot(pars, args)
    
    elif fittype == 'ring_3blob_2peak':
        ring_model = new_model_ring_3blob_2peak_setup_plot(pars, args)

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
    elif fittype == 'blob_15axr':
        label = ["Peak", "Width", "Radius", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak", "B1. Width", "B1. AxR", "B1. PA", "B1. Dist", "B1. Angle"]
    elif fittype == 'ring_3blob':
        label = ["Peak", "Width", "Radius", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak", "B1. Width", "B1. Dist", "B1. Angle", "B2. Peak", "B2. Width", "B2. Dist", "B2. Angle", "B3. Peak", "B3. Width", "B3. Dist", "B3. Angle"]
    elif fittype == 'ring_3blob_1fish':
        label = ["Peak", "Width", "Radius", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak", "B1. Width", "B2. Peak", "B2. Width", "B3. Peak", "B3. Width", "B3. Dist", "B3. Angle"]
    elif fittype == 'ring_3blob_2peak':
        label = ["Peak", "Width", "Radius", "Inc", "PA", "Offset RA", "Offset Dec", "B1. Peak 1", "B1. Width 1", "B1. Peak 2", "B1. Width 2", "B1. Dist", "B1. Angle", "B2. Peak 1", "B2. Width 1", "B2. Peak 2", "B2. Width 2", "B2. Dist", "B2. Angle", "B3. Peak 1", "B3. Width 1", "B3. Peak 2", "B3. Width 2", "B3. Dist", "B3. Angle"]
    else:
        logging.warning('Please choose a valid fitting model, or add a new one into the code.')
    
    return label