#!/usr/bin/env python

import numpy as N
import pylab as P
import scipy as sp
import fitsio
from astropy.io import fits
import argparse
import glob
import healpy
import sys
from scipy import random
import copy

from picca import constants
from picca.data import delta

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value

import configargparse



class kappa:

    nside = 512
    nside_data = 32
    rot = healpy.Rotator(coord=['C', 'G'])
    lambda_abs = 1215.67

    fid_Om=0.31
    cosmo = constants.cosmo(fid_Om)

    rt_min=0.
    rt_max=50.
    rp_min=0.
    rp_max=20.
    nt = 1
    np = 1

    xi2d = None

    angmax = None
    z_min_pix = None

    counter = Value('i',0)
    lock = Lock()

    data={}
    ndata=0

    @staticmethod
    def load_model(modelfile, nbins=20) :

        data_rp, data_rt, xi_dist = N.loadtxt(modelfile, unpack=1)

        kappa.rp_min = data_rp[:50].max()
        kappa.rt_min = data_rt[::50].max()
        rp = N.linspace(kappa.rp_min, kappa.rp_max, nbins)
        rt = N.linspace(kappa.rt_min, kappa.rt_max, nbins)
        xim = sp.interpolate.griddata((data_rt, data_rp), xi_dist, \
                    (rt[:, None], rp[None, :]), method='cubic')

        xi2d = sp.interpolate.RectBivariateSpline(rp, rt, xim)
        r2 = rt[:, None]**2+rp[None, :]**2

        kappa.xi2d = xi2d
        return xi2d

    @staticmethod
    def read_deltas(in_dir, nspec=None):
        data = {}
        ndata = 0
        dels = []

        fi = glob.glob(in_dir+"/*.fits.gz")
        for i,f in enumerate(fi):
            sys.stderr.write("\rread {} of {} {}".format(i,len(fi),ndata))
            hdus = fitsio.FITS(f)
            dels += [delta.from_fitsio(h) for h in hdus[1:]]
            ndata+=len(hdus[1:])
            hdus.close()
            if nspec and ndata > nspec:
                break

        phi = [d.ra for d in dels]
        th = [sp.pi/2.-d.dec for d in dels]
        pix = healpy.ang2pix(kappa.nside_data, th, phi)

        z_min_pix = 10**dels[0].ll[0]/kappa.lambda_abs-1

        for d,p in zip(dels,pix):
            if p not in data:
                data[p]=[]
            data[p].append(d)

            z = 10**d.ll/kappa.lambda_abs-1.
            z_min_pix = sp.amin( sp.append([z_min_pix], z) ) 
            d.z = z
            d.r_comov = kappa.cosmo.r_comoving(z)
            d.we *= ((1.+z)/(1.+2.25))**(2.9-1.)
            d.project()

        kappa.z_min_pix = z_min_pix
        kappa.angmax = 2.*\
                sp.arcsin(kappa.rt_max/(2.*kappa.cosmo.r_comoving(z_min_pix)))
        kappa.ndata = ndata
        kappa.data = data

        return data

    @staticmethod
    def fill_neighs():
        data = kappa.data
        for ipix in data.keys():
            for d1 in data[ipix]:
                npix = healpy.query_disc(kappa.nside_data, \
                        [d1.xcart, d1.ycart, d1.zcart], \
                        kappa.angmax, inclusive = True)
                npix = [p for p in npix if p in data]
                neighs = [d for p in npix for d in data[p]]
                ang = d1^neighs
                w = ang<kappa.angmax
                neighs = sp.array(neighs)[w]
                d1.neighs = [d for d in neighs if d1.ra > d.ra]
        
    @staticmethod
    def get_kappa(pixels):

        ikappa = []
        skappa = {} 
        wkappa = {}

        for ipix in pixels:
            for i,d1 in enumerate(kappa.data[ipix]):
                sys.stderr.write("\rcomputing kappa: {}%".format(\
                                 round(kappa.counter.value*100./kappa.ndata,2)))
                with kappa.lock:
                    kappa.counter.value += 1
                for d2 in d1.neighs:
                    #--  compute the cartesian mid points and convert back 
                    #--  to ra, dec
                    mid_xcart = 0.5*(d1.xcart+d2.xcart)
                    mid_ycart = 0.5*(d1.ycart+d2.ycart)
                    mid_zcart = 0.5*(d1.zcart+d2.zcart)
                    mid_ra = sp.arctan(mid_ycart/mid_xcart) + \
                             sp.pi+sp.pi*(mid_xcart>0)
                    mid_ra -= 2*sp.pi*(mid_ra>2*sp.pi)
                    mid_dec = sp.arcsin(mid_zcart/\
                              sp.sqrt(mid_xcart**2+mid_ycart**2+mid_zcart**2))

                    #-- apply rotation into Galactic coordinates
                    th, phi = kappa.rot(sp.pi/2-mid_dec, mid_ra)

                    #-- check if pair of skewers belong to same spectro
                    same_half_plate = (d1.plate == d2.plate) and\
                            ( (d1.fid<=500 and d2.fid<=500) or \
                            (d1.fid>500 and d2.fid>500) )

                    #-- angle between skewers
                    ang = d1^d2

                    #-- getting pixel in between 
                    mid_pix = healpy.ang2pix(kappa.nside, \
                                  th, phi) 
                    sk, wk = kappa.fast_kappa(\
                            d1.z, d1.r_comov, d1.we, d1.de, \
                            d2.z, d2.r_comov, d2.we, d2.de, \
                            ang, same_half_plate) 
                    if mid_pix in ikappa:
                        skappa[mid_pix]+=sk
                        wkappa[mid_pix]+=wk
                    else:
                        ikappa.append(mid_pix)
                        skappa[mid_pix]=sk
                        wkappa[mid_pix]=wk

                setattr(d1, "neighs", None)
                
        return ikappa, skappa, wkappa

    @staticmethod
    def fast_kappa(z1,r1,w1,d1,z2,r2,w2,d2,ang,same_half_plate):
        wd1 = d1*w1
        wd2 = d2*w2
        rp = abs(r1-r2[:,None])*sp.cos(ang/2)
        rt = (r1+r2[:,None])*sp.sin(ang/2)
        wd12 = wd1*wd2[:,None]
        w12 = w1*w2[:,None]
        z = (z1+z2[:,None])/2

        w = (rp>=kappa.rp_min) & (rp<=kappa.rp_max) & \
            (rt<=kappa.rt_max) & (rt>=kappa.rp_min)

        rp = rp[w]
        rt = rt[w]
        z  = z[w]
        wd12 = wd12[w]
        w12 = w12[w]

        #-- getting model and first derivative
        xi_model = kappa.xi2d(rt, rp, grid=False)
        xip_model = kappa.xi2d(rt, rp, dx=1, grid=False)*rt
       
        ska = sp.sum( (wd12 - xi_model)*xip_model*w12 )
        wka = sp.sum( w12*xip_model**2 ) 

        return ska, wka




def compute_kappa(p):
    tmp = kappa.get_kappa(p)
    return tmp
            

if __name__=='__main__':

    parser = configargparse.ArgParser()

    parser.add('--deltas', required=True, type=str, \
               help='folder containing deltas in pix format')
    parser.add('--model', required=True, \
               help='text file containing model')
    parser.add('--out', required=True, \
               help='output fits file with kappa map')
    parser.add('--nproc', required=False, type=int, default=1, \
               help='number of procs used in calculation')
    args, unknown = parser.parse_known_args()


    kappa.load_model(args.model)
    kappa.read_deltas(args.deltas)
    kappa.fill_neighs()

    cpu_data = {}
    for p in kappa.data.keys():
        cpu_data[p] = [p]

    print(' ', len(kappa.data.keys()), 'pixels with data')
    pool = Pool(processes=args.nproc)
    results = pool.map(compute_kappa, cpu_data.values())
    pool.close()
    print('')

    #-- compiling results from pool
    kap = sp.zeros(12*kappa.nside**2)
    wkap = sp.zeros(12*kappa.nside**2)
    for i, r in enumerate(results):
        print(i, len(results))
        index = r[1].keys()
        values = r[1].values()
        kap[index]  += values
        index = r[2].keys()
        values = r[2].values()
        wkap[index] += values
    
    w = wkap>0
    kap[w]/= wkap[w]
    
    out = fitsio.FITS(args.out% \
            (kappa.nside, kappa.rp_max, kappa.rt_max), \
            'rw', clobber=True)
    head = {}
    head['RPMIN']=kappa.rp_min
    head['RPMAX']=kappa.rp_max
    head['RTMAX']=kappa.rt_max
    head['NT']=kappa.nt
    head['NP']=kappa.np
    head['NSIDE']=kappa.nside
    out.write([kap, wkap], names=['kappa', 'wkappa'], header=head)
    out.close()



