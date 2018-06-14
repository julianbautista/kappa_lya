import healpy as hp
import numpy as np
import pylab as P


def make_maps(a3darraymap):  
    '''a function that makes maps from gamma 1 and gamma 2 shear measurements -
        needs a 3d array of an empty map, a g1 map and a g2map
    '''
    
    #finds nside of a map from just input map
    nside = hp.npix2nside(len(a3darraymap[1])) 
    lmax=2*nside-1
    nell = hp.sphtfunc.Alm.getsize(lmax=lmax)
    
    #puts ell modes in an array, doesnt need to be a loop
    lmode = hp.sphtfunc.Alm.getlm(lmax=lmax, i=np.arange(nell))[0] 

    #-- get alm from input map
    alms = hp.sphtfunc.map2alm(a3darraymap, lmax=lmax, pol=True)

    #ell mode coefficients to go from E_alm to gravitational potential
    lfactor_phi = -2/np.sqrt((lmode+2)*(lmode+1)*(lmode)*(lmode-1)) 
    lfactor_phi[lmode<2]=0

    ##ell mode coefficients to go from E_alm to kappa
    lfactor_kappa = lmode*(lmode+1)/(lmode+2)/np.sqrt(lmode-1) 
    lfactor_kappa[lmode<2]=0

    ## SY 30/5/18 Added minus sign below
    #ell mode coefficients to go from E_alm to alpha, bend angle
    lfactor_alpha = -2/np.sqrt((lmode+2)*(lmode-1)) 
    lfactor_alpha[lmode<2] = 0

    weird_alm = lfactor_kappa*alms[1]
    phi_alm   = lfactor_phi  *alms[1]

    alphaminus_alm = lfactor_alpha*alms[1]
    alphaplus_alm  = lfactor_alpha*alms[2]

    weird_map = hp.sphtfunc.alm2map(kappa_alm, nside=nside, lmax=lmax)
    phi_map   = hp.sphtfunc.alm2map(phi_alm,   nside=nside, lmax=lmax)
    # -- spin1 transform to get the 2 bend angles
    alpha_map = hp.alm2map_spin([alphaplus_alm, alphaminus_alm], 
                    nside=nside, spin=1, lmax=lmax) 
    return weird_map, phi_map, alpha_map


def get_kappa_phi_alpha_ben(cell=None, kappa_map=None, nside=512, seed=1):
    lmax=2*nside-1 
    np.random.seed(seed)

    #-- Using synfast to generate a Gaussian random field using 
    #   the power spectrum,
    #   chose to define this as observed kappa as opposed to observed E mode. 
    #   Pol=False means a scalar field is output only. 
    #   Can play with these to get more types of fields out
    if kappa_map is None and cell is not None:
        kappa_map = hp.sphtfunc.synfast(cell, nside=nside, lmax=lmax, pol=True) 
    lmode , em = hp.sphtfunc.Alm.getlm(lmax=lmax) 

    #-- Kappa map into harmonic space
    alm = hp.sphtfunc.map2alm(kappa_map, lmax=lmax, pol=False) 

    #-- inverse of coefficients that relate the kappa field and 
    #   the "E mode type" measurement from lensing shear. 
    lfac = np.sqrt( (lmode+2)*(lmode-1)/lmode/(lmode+1) )
    lfac[lmode<2]=0

    #-- Getting a E mode in spherical space from this kappa 
    Etruealm = lfac*alm 

    #-- Getting a shear (spin-2 ) field from the E modes
    [kk, g1in, g2in] = hp.sphtfunc.alm2map([alm*0.0, Etruealm, alm*0.0], 
                                           nside=nside, lmax=lmax, pol=True) 

    # SY 13/4/18
    # first array needs to be empty & same dimension as others
    a3darraymap = np.array([g1in*0.0, g1in, g2in]) 
    weird_map, phi_map, alpha_map = make_maps(a3darraymap)
    return kappa_map, weird_map, phi_map, alpha_map 

def read_des_cl():

    #-- Loading in a CAMB C_ls power spectrum file
    ell, cell = np.loadtxt('y1a1_massmapping_cl_nofz_y1a1_spt_mcal_2.txt', 
                           unpack=1) 
    ell_max = int(ell.max())
    cell_new = np.zeros(ell_max)
    cell_new[1:] = np.interp(np.arange(1, ell_max), ell, cell)
    ell_new = np.arange(ell_max)

#hp.fitsfunc.write_map('outputs/kappa_map.fits', kappa_map)
#hp.fitsfunc.write_map('outputs/phi_map.fits', phi_map)
#hp.fitsfunc.write_map('outputs/alpha_map.fits', alpha_map)



