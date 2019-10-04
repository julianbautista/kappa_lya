# Author: Sam Youles
# Modified by Julian Bautista

import numpy as np
import sys
import glob
import os
import fitsio
import argparse

from kappa_lya import create_gaussian_kappa
from astropy.table import Table

parser = argparse.ArgumentParser()
parser.add_argument('--cl_file', required=True, help='Input convergence power spectrum')
parser.add_argument('--in_cat', required=False, help='Input quasar catalog')
parser.add_argument('--in_dir', required=False, help='Input directory containing deltas')
parser.add_argument('--seed', type=int, required=True, help='Seed for random generator')
parser.add_argument('--nside', type=int, default=2048, help='Healpix nside for lensing map')
parser.add_argument('--out_kappa_map', required=False, 
    help='Fits file where the kappa map will be written')
parser.add_argument('--out_cat', required=False, help='Output lensed quasar catalog')
parser.add_argument('--out_dir', required=False, help='Output directory containing lensed deltas')
args = parser.parse_args()


#-- Input power spectrum as a text file
ell, cell = np.loadtxt(args.cl_file, unpack=1)

#-- Input 
indir = args.in_dir 
incat = args.in_cat
if indir is None and incat is None:
    print('Need at least one: --in_dir or --in_cat')
    sys.exit()

#-- Create kappa map from power spectrum
kappa = create_gaussian_kappa(ell, cell, nside=args.nside, seed=args.seed)
if args.out_kappa_map:
    t = Table()
    t['KAPPA'] = kappa.input_map
    t.meta['NSIDE'] = args.nside
    t.meta['SEED'] = args.seed
    t.meta['CL_FILE'] = args.cl_file
    t.write(args.out_kappa_map, overwrite=True)
    print('Kappa map exported to:', args.out_kappa_map)

#-- Move quasars in catalog
if args.in_cat:
    zcat = Table.read(args.in_cat)
    zcat.rename_column('RA', 'RA0') 
    zcat.rename_column('DEC', 'DEC0') 
    ra  = np.radians(zcat['RA0'].data)
    dec = np.radians(zcat['DEC0'].data)
    theta_lens, phi_lens = kappa.displace_objects(np.pi/2-dec, ra) 
    ra_lens  = np.degrees(phi_lens)
    dec_lens = np.degrees(np.pi/2-theta_lens)
    zcat['RA'] = ra_lens
    zcat['DEC']= dec_lens
    zcat.meta['NSIDE'] = args.nside
    zcat.meta['SEED'] = args.seed
    zcat.meta['CL_FILE'] = args.cl_file
    zcat.write(args.out_cat, overwrite=True)
    print('Done moving ', len(zcat), 'quasars')
    print('Lensed quasar catalog exported to:', args.out_cat)


#-- Move forests in delta files  
if args.in_dir:
    os.makedirs(args.out_dir, exist_ok=True)

    alldeltas = glob.glob(args.in_dir+'/*.fits.gz')
    ndel = len(alldeltas)
    i = 0
    nqso = 0
    for filename in alldeltas:
        hdus = fitsio.FITS(filename)
        print(i, ndel)
        i+=1

        out = fitsio.FITS(args.out_dir+"/"+os.path.basename(filename),'rw', clobber=True)

        for hdu in hdus[1:]:
            header = hdu.read_header()
            
            #-- These are in radians
            ra = header['RA']
            dec = header['DEC']

            #-- Add bend angles to ra and dec
            theta_lens, phi_lens = kappa.displace_objects(np.pi/2-dec, ra) 
            
            #-- Rewrite new delta file with new values
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
            nqso += 1

        out.close()

    print('Done moving ', ndel, 'delta files containing', nqso, 'forests')
    print('Lensed deltas exported to:', args.out_dir)


