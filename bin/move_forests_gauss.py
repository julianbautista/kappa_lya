# Author: Sam Youles
# Modified by Julian Bautista

import numpy as np
import healpy as hp
import sys
import glob
import os
import fitsio
from kappa_lya import *

# input directory name containing delta files
indir = sys.argv[1]
outdir = sys.argv[2]



#-- Create angular power spectrum of kappa
theory = Theory()
ell, cell = theory.get_cl_kappa(2.1, kmax=100., nz=100, lmax=10000)

nside=1024
npix=nside**2*12
seed=1
np.random.seed(seed)
kappa = create_gaussian(ell, cell, nside=nside, seed=seed)

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
        theta_lens, phi_lens = kappa.displace_objects(np.pi/2-dec, ra) 
        
        # Rewrite new delta file with new values
        header['RA'] = phi_lens
        header['DEC'] = np.pi/2-theta_lens
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

