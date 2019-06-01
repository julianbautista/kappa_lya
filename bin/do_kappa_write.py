#!/usr/bin/env python
  
import numpy as np
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
from picca.mydata import delta

from multiprocessing import Pool,Process,Lock,Manager,cpu_count,Value


class kappa:

    nside = 256
    nside_data = 32
    rot = healpy.Rotator(coord=['C', 'G'])
    lambda_abs = 1215.67

    fid_Om=0.31
    cosmo = constants.cosmo(fid_Om)

    rt_min=0.
    rt_max=40.
    rp_min=0.
    rp_max=10.
    nt = 1
    np = 1

    xi2d = None

    angmax = None
    z_min_pix = None

    true_corr = False

    counter = Value('i',0)
    lock = Lock()

    data={}
    ndata=0

    @staticmethod
    def load_model(modelfile, nbins=50) :

        data_rp, data_rt, xi_dist = np.loadtxt(modelfile, unpack=1)

        #-- get the larger value of the first separation bin to make a grid
        rp_min = data_rp.reshape(50, 50)[0].max()
        rp_max = data_rp.reshape(50, 50)[-1].min()
        rt_min = data_rt.reshape(50, 50)[:, 0].max()
        rt_max = data_rt.reshape(50, 50)[:, -1].min()
        #-- create the regular grid for griddata
        rp = np.linspace(rp_min, rp_max, nbins)
        rt = np.linspace(rt_min, rt_max, nbins)
        xim = sp.interpolate.griddata((data_rt, data_rp), xi_dist, \
                    (rt[:, None], rp[None, :]), method='cubic')

        #-- create interpolator object
        xi2d = sp.interpolate.RectBivariateSpline(rt, rp, xim)

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
        print('\n Filling neighbors')
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

        id1 = []
        id2 = []
        skappa = []
        wkappa = []

        for ipix in pixels:
            j=0
            for i,d1 in enumerate(kappa.data[ipix]):
                j+=1
                kcounter = round(kappa.counter.value*100./kappa.ndata,2)
                if (j%5==0):
                    sys.stderr.write("\rcomputing kappa: {}%".format(kcounter))
                with kappa.lock:
                    kappa.counter.value += 1
                for d2 in d1.neighs:
                    #--  compute the cartesian mid points and convert back 
                    #--  to ra, dec
                    #mid_xcart = 0.5*(d1.xcart+d2.xcart)
                    #mid_ycart = 0.5*(d1.ycart+d2.ycart)
                    #mid_zcart = 0.5*(d1.zcart+d2.zcart)
                    #mid_ra, mid_dec = get_radec(\
                    #    np.array([mid_xcart, mid_ycart, mid_zcart]))

                    #-- apply rotation into Galactic coordinates
                    #th, phi = kappa.rot(sp.pi/2-mid_dec, mid_ra)
                    #-- keeping without rotation
                    #th, phi = sp.pi/2-mid_dec, mid_ra

                    #-- check if pair of skewers belong to same spectro
                    #same_half_plate = (d1.plate == d2.plate) and\
                    #        ( (d1.fid<=500 and d2.fid<=500) or \
                    #        (d1.fid>500 and d2.fid>500) )

                    #-- getting pixel in between 
                    #mid_pix = healpy.ang2pix(kappa.nside, \
                    #              th, phi)
                    
                    #-- angle between skewers
                    ang = d1^d2
                    if kappa.true_corr:
                        ang_delensed = d1.delensed_angle(d2)


                    if kappa.true_corr:
                        sk, wk = kappa.fast_kappa_true(\
                                d1.z, d1.r_comov, \
                                d2.z, d2.r_comov, \
                                ang_delensed, ang)
                    else:
                        sk, wk = kappa.fast_kappa(\
                                d1.z, d1.r_comov, d1.we, d1.de, \
                                d2.z, d2.r_comov, d2.we, d2.de, \
                                ang)

                    if wk != 0:
                        id1.append(d1.thid)
                        id2.append(d2.thid)
                        skappa.append(sk)
                        wkappa.append(wk)


                setattr(d1, "neighs", None)

        return id1, id2, skappa, wkappa


    @staticmethod
    def fast_kappa(z1,r1,w1,d1,z2,r2,w2,d2,ang):
        rp = abs(r1-r2[:,None])*sp.cos(ang/2)
        rt = (r1+r2[:,None])*sp.sin(ang/2)
        d12 = d1*d2[:, None]
        w12 = w1*w2[:,None]

        w = (rp>=kappa.rp_min) & (rp<=kappa.rp_max) & \
            (rt<=kappa.rt_max) & (rt>=kappa.rt_min)

        rp = rp[w]
        rt = rt[w]
        w12 = w12[w]
        d12 = d12[w]


        #-- getting model and first derivative
        xi_model = kappa.xi2d(rt, rp, grid=False)
        xip_model = kappa.xi2d(rt, rp, dx=1, grid=False)

        #-- weight of estimator
        R = 1/(xip_model*rt)

        ska = sp.sum( (d12 - xi_model)/R*w12 )
        wka = sp.sum( w12/R**2 )

        return ska, wka

    @staticmethod
    def fast_kappa_true(z1, r1, z2, r2, ang, ang_lens):

        rp      = abs(r1-r2[:,None])*sp.cos(ang/2)
        rt      = (r1+r2[:,None])*sp.sin(ang/2)
        rp_lens = abs(r1-r2[:,None])*sp.cos(ang_lens/2)
        rt_lens = (r1+r2[:,None])*sp.sin(ang_lens/2)

        #z = (z1+z2[:,None])/2

        w = (rp>=kappa.rp_min) & (rp<=kappa.rp_max) & \
            (rt<=kappa.rt_max) & (rt>=kappa.rt_min)

        rp = rp[w]
        rt = rt[w]
        rp_lens = rp_lens[w]
        rt_lens = rt_lens[w]
        #z  = z[w]

        #-- getting model and first derivative
        xi_model  = kappa.xi2d(rt,      rp,       grid=False)
        xi_lens   = kappa.xi2d(rt_lens, rp_lens,  grid=False)
        xip_model = kappa.xi2d(rt,      rp, dx=1, grid=False)
        R = 1/(xip_model*rt)

        ska = sp.sum( (xi_lens - xi_model)/R )
        wka = sp.sum( 1/R**2  )

        return ska, wka

def get_radec(pos):
    ra = np.arctan(pos[1]/pos[0]) + np.pi + np.pi*(pos[0]>0)
    ra -= 2*np.pi*(ra>2*np.pi)
    dec = np.arcsin(pos[2]/np.sqrt(pos[0]**2+pos[1]**2+pos[2]**2))
    return ra, dec



def compute_kappa(p):
    id1, id2, skappa, wkappa = kappa.get_kappa(p)
    return id1, id2, skappa, wkappa

if __name__=='__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--deltas', required=True, type=str, \
               help='folder containing deltas in pix format')
    parser.add_argument('--model', required=True, \
               help='text file containing model') 
    parser.add_argument('--out', required=True, \
               help='output fits file with kappa values')
    parser.add_argument('--nproc', required=False, type=int, default=1, \
               help='number of procs used in calculation')
    parser.add_argument('--nspec', required=False, type=int, default=None, \
               help='number of spectra to process')
    parser.add_argument('--rt_min', required=False, type=float, default=3., \
               help='minimum transverse separation')
    parser.add_argument('--rp_min', required=False, type=float, default=3., \
               help='minimum radial separation')
    parser.add_argument('--rt_max', required=False, type=float, default=40., \
               help='maximum transverse separation')
    parser.add_argument('--rp_max', required=False, type=float, default=10., \
               help='maximum radial separation')
    parser.add_argument('--true_corr', required=False, default=False,\
               action='store_true', help='use actual lensed correlation')
    args = parser.parse_args()

    kappa.true_corr = args.true_corr
    kappa.rt_min = args.rt_min
    kappa.rp_min = args.rp_min
    kappa.rt_max = args.rt_max
    kappa.rp_max = args.rp_max
    kappa.load_model(args.model)
    kappa.read_deltas(args.deltas, nspec=args.nspec)
    kappa.fill_neighs()

    cpu_data = {}
    for p in kappa.data.keys():
        cpu_data[p] = [p]

    print(' ', len(kappa.data.keys()), 'pixels with data')
    pool = Pool(processes=args.nproc)
    results = pool.map(compute_kappa, cpu_data.values())
    pool.close()
    #print('', results)

    id1 = np.empty(0, dtype=int)
    id2 = np.empty(0, dtype=int)
    skappa = np.empty(0)
    wkappa = np.empty(0)
    
    for r in results:
        id1 = np.append(id1, np.array(r[0]).astype(int))
        id2 = np.append(id2, np.array(r[1]).astype(int))
        skappa = np.append(skappa, np.array(r[2]))
        wkappa = np.append(wkappa, np.array(r[3]))


    out = fitsio.FITS(args.out, 'rw', clobber=True)
    head = {}
    head['RPMIN']=kappa.rp_min
    head['RPMAX']=kappa.rp_max
    head['RTMIN']=kappa.rt_min
    head['RTMAX']=kappa.rt_max
    head['NT']=kappa.nt
    head['NP']=kappa.np
    head['NSIDE']=kappa.nside
    out.write([id1, id2, skappa, wkappa],
              names=['THID1', 'THID2', 'SKAPPA', 'WKAPPA'],
              header=head)
    out.close()




