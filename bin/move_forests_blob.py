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


# input directory name containing delta files
indir = sys.argv[1]
outdir = sys.argv[2]


nside=512
npix=nside**2*12

kappa = N.zeros(npix)
theta, phi = hp.pix2ang(nside, N.arange(npix))
kappa = SphericalMap(N.exp(-((phi-N.pi)**2+(theta-(1.1))**2)/(2*0.1**2)))
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
        header['RA_LENS'] = phi_lens
        header['DEC_LENS'] = N.pi/2-theta_lens
      
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

