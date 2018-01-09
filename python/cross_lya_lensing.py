import healpy
from astropy.io import fits
import os
import numpy as N

a = fits.open('kappa_nside512_rpmax20.0_rtmax50.0.fits.gz')[1].data
lya_kappa = a.kappa

#-- masking values where there's no forests
w0 = lya_kappa!=0.
w = (lya_kappa > N.percentile(lya_kappa[w0], 0.5)) & \
    (lya_kappa < N.percentile(lya_kappa[w0], 99.5))
lya_kappa[~w] = healpy.UNSEEN 

#-- smoothing in angular direction
lya_kappa_sm = healpy.smoothing(lya_kappa, fwhm=15./60*N.pi/180)

#-- getting alm and auto power-spectrum 
lya_alm = healpy.map2alm(lya_kappa_sm)
lya_cl = healpy.alm2cl(lya_alm)

#-- reading CMB kappa
cmb_kappa_alm = healpy.read_alm(os.environ['ASTRODATA']+\
        '/Planck/lensing/data/dat_klm.fits')
cmb_map = healpy.alm2map(cmb_kappa_alm, 512)

#-- masking CMB map over BOSS area
cmb_map_masked = cmb_map*1.
cmb_map_masked[~w0] = healpy.UNSEEN

#-- CMB kappa auto-power
cmb_cl = healpy.alm2cl(cmb_kappa_alm)

#-- cross power-spectrum
lya_cmb_cl = healpy.anafast(lya_kappa_sm, cmb_map_masked)
ell = N.arange(lya_cmb_cl.size)


