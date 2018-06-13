# Author: Sam Youles
# Modified by Julian Bautista

from astropy.io import fits
import numpy as N
import healpy as hp
import sys
import glob
import os
import fitsio
from SphericalDiff import *

import camb
import camb.model
import pylab as plt
import numpy as np
import sys

def create_cl_kappa(zstar=2.1):

    nz = 100 #number of steps to use for the radial/redshift integration
    kmax=100  #kmax to use
    #First set up parameters as usual
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.8, ombh2=0.022, omch2=0.122)
    pars.InitPower.set_params(ns=0.965)

    # For Limber result, want integration over \chi (comoving radial distance),
    # from 0 to chi_*.
    #so get background results to find chistar, set up arrage in chi, and calculate corresponding redshifts
    results= camb.get_background(pars)

    chistar = results.comoving_radial_distance(zstar)
    #chistar = results.conformal_time(0)- camb.model.tau_maxvis.value
    chis = np.linspace(0,chistar,nz)
    zs=results.redshift_at_comoving_radial_distance(chis)
    #Calculate array of delta_chi, and drop first and last points where things go singular
    dchis = (chis[2:]-chis[:-2])/2
    chis = chis[1:-1]
    zs = zs[1:-1]

    #-- Get the matter power spectrum interpolation object 
    #-- (based on RectBivariateSpline). 
    #-- Here for lensing we want the power spectrum of the Weyl potential.
    PK = camb.get_matter_power_interpolator(pars, nonlinear=True,
        hubble_units=False, k_hunit=False, kmax=kmax, zmax=zs[-1])
        #var1=camb.model.Transfer_Weyl, 
        #var2=camb.model.Transfer_Weyl)

    #win = ((chistar-chis)/(chis**2*chistar))**2
    win = ((chistar-chis)/(chistar/(1+zs)))**2
    #Do integral over chi
    ls = np.arange(2,10000, dtype=np.float64)
    cl_kappa=np.zeros(ls.shape)
    #this is just used to set to zero k values out of range of interpolation
    w = np.ones(chis.shape)
    for i, l in enumerate(ls):
        #k=(l+0.5)/chis
        k=(l)/chis
        w[:]=1
        w[k<1e-4]=0
        w[k>=kmax]=0
        #cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win/k**4)
        cl_kappa[i] = np.dot(dchis, w*PK.P(zs, k, grid=False)*win)

    #convert kappa power to [l(l+1)]^2C_phi/2pi (what cl_camb is)
    #cl_kappa *= (ls*(ls+1))**2 * 4 / (2*np.pi)
    cl_kappa *= 9/4*pars.omegam**2*(0.7/3000)**4

    ell = np.zeros(ls.size+2)
    cl = np.zeros(ls.size+2)
    ell[2:] = ls
    cl[2:] = cl_kappa

    return ell, cl



# input directory name containing delta files
indir = sys.argv[1]
outdir = sys.argv[2]


nside=1024
npix=nside**2*12


ell, cl = create_cl_kappa()
#-- factor to make sensible displacements
factor = ell*(ell+1)/(2*np.pi)

obsk = hp.sphtfunc.synfast(cl*factor, nside=nside, lmax=2*nside-1,\
                 pol=False)
kappa = SphericalMap(obsk) 
kappa.compute_deriv()

# Amend DEC and RA in each of the delta files by the bend angle from alpha map
alldeltas = glob.glob(indir+'/*.fits.gz')
ndel = len(alldeltas)
i=0
for filename in alldeltas:
    #hdus = fits.open(filename)
    hdus = fitsio.FITS(filename)
    print(i, ndel)
    i+=1

    out = fitsio.FITS(outdir+"/"+os.path.basename(filename),'rw',clobber=True)

    for hdu in hdus[1:]:
        header = hdu.read_header()
        ra = header['RA']
        dec = header['DEC']

        # Add bend angles to ra and dec
        theta_lens, phi_lens = kappa.DisplaceObjects(N.pi/2-dec, ra) 
        
        # Rewrite new delta file with new values
        header['RA'] = phi_lens
        header['DEC'] = N.pi/2-theta_lens
        header['RA0'] = ra
        header['DEC0'] = dec
      
        #-- Re-create columns (maybe there's a better way to do this?) 
        ll = hdu['LOGLAM'][:]
        de = hdu['DELTA'][:]
        we = hdu['WEIGHT'][:]
        co = hdu['CONT'][:] 
        cols=[ll, de, we, co]
        names=['LOGLAM','DELTA','WEIGHT','CONT']
        out.write(cols, names=names, header=header, \
                  extname=str(header['THING_ID']))

    out.close()

