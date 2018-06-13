import healpy
from astropy.io import fits
import os
import numpy as N
import pylab as P
import sys


def plot_cl(infile):

    a = fits.open(infile)[1].data
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
    #cmb_kappa_alm = healpy.read_alm(os.environ['ASTRODATA']+\
    #        '/Planck/lensing/data/dat_klm.fits')
    #cmb_map = healpy.alm2map(cmb_kappa_alm, 512)

    #-- masking CMB map over BOSS area
    #cmb_map_masked = cmb_map*1.
    #cmb_map_masked[~w0] = healpy.UNSEEN

    #-- CMB kappa auto-power
    #cmb_cl = healpy.alm2cl(cmb_kappa_alm)

    #-- cross power-spectrum
    #lya_cmb_cl = healpy.anafast(lya_kappa_sm, cmb_map_masked)



    ell = N.arange(lya_cl.size)
    weights = 2.*ell+1.

    ellmin = 10
    ellmax = max(ell)
    nell = 100

    w = (ell>=ellmin)&(ell<=ellmax)
    index = N.floor( (ell[w]-ellmin)*1./(ellmax-ellmin)*nell ).astype(int)
    well = N.bincount( index, weights=weights[w])
    sell = N.bincount( index, weights=weights[w]*ell[w])
    scl =  N.bincount( index, weights=weights[w]*lya_cl[w])

    new_ell = sell/well
    new_cl = scl/well

    #P.plot(ell, lya_cmb_cl)
    P.plot(new_ell, new_cl, label=os.path.basename(infile)[:-8])
    #P.ylim(-0.001, 0.001)
    P.xscale('log')
    P.ylabel(r'$\langle \kappa_{\rm Lya} \kappa_{\rm Lya} \rangle$', fontsize=20)
    P.xlabel(r'$\ell$', fontsize=20)
    P.legend(loc=0)
    P.tight_layout()
        
for fi in sys.argv[1:]:
    plot_cl(fi)
P.show()
#plot_cl('kappa-mock-lensed.fits.gz')
#P.legend(['Unlensed', 'Lensed'])
#P.savefig('plots/cl-kappa-mock.pdf', bbox_inches='tight')



