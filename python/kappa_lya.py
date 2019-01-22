import camb
import camb.model
import pylab as plt
import numpy as np
import sys
import healpy as hp
from astropy.io import fits

class Theory:

    def __init__(self, H0=67.8, ombh2=0.022, omch2=0.122):
        #First set up parameters as usual
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2)
        pars.InitPower.set_params(ns=0.965)
        results= camb.get_background(pars)

        self.pars = pars
        self.results = results

    def get_cl_kappa(self, z, kmax=100, nz=100, lmax=10000):
        # For Limber result, want integration over \chi 
        # (comoving radial distance) from 0 to chi_*.
        # so get background results to find chistar, 
        # set up arrage in chi, and calculate corresponding redshifts
        pars = self.pars
        results = self.results

        chistar = results.comoving_radial_distance(z)
        chis = np.linspace(0, chistar, nz)
        zs = results.redshift_at_comoving_radial_distance(chis)
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
        ls = np.arange(2, lmax, dtype=float)
        cl_kappa=np.zeros(ls.shape)
        #this is just used to set to zero k values out of range of interpolation
        w = np.ones(len(chis))
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
        cl_kappa *= 9/4*pars.omegam**2*(pars.H0/299792.458)**4 

        ell  = np.zeros(ls.size+2)
        cell = np.zeros(ls.size+2)
        ell[2:] = ls
        cell[2:] = cl_kappa

        self.PK = PK
        self.ell = ell
        self.cell = cell

        return ell, cell



#-- SphericalMap from Anze Slosar 
#-- modified by Julian Bautista
class SphericalMap:
    
    def __init__ (self, mapin, alm_pack=None):
        if alm_pack is None:
            self.A=mapin
            self.npix = len(mapin)
            self.nside = hp.npix2nside(self.npix) 
            Alm = hp.map2alm(self.A)
            lmax = self.nside*3
            self.Alm = np.zeros((lmax, lmax), np.complex)
            self.ell = np.outer(np.arange(lmax),np.ones(lmax))
            self.em = self.ell.T
            self.theta_pix, self.phi_pix = hp.pix2ang(self.nside, np.arange(self.npix))
            cc=0
            for m in range(lmax):
                for l in range(m, lmax):
                    self.Alm[l, m] = Alm[cc]
                    self.ell[l, m] = l
                    self.em[l, m] = m
                    cc+=1
        else:
            self.Alm, self.ell, self.em, self.nside, self.npix=alm_pack
            lmax=self.nside*3
            ma=[]
            for m in range(lmax):
                for l in range(m, lmax):
                    ma.append(self.Alm[l, m])
            self.A = hp.alm2map(np.array(ma), self.nside) 
            self.theta_pix, self.phi_pix = hp.pix2ang(self.nside, 
                                               np.arange(self.A.size))
            
   
    def compute_deriv(self):
        #-- Cl_kappa = - ell*(ell+1)/2 * Cl_phi
        self.lphi = self.inv_laplace()
        self.lphi.A *= -2  
        self.lphi.Alm *= -2
        self.dtheta_map = self.lphi.dtheta()
        self.dphi_map = self.lphi.dphi()
        dtheta = self.dtheta_map.A
        dphi = self.dphi_map.A/np.sin(self.theta_pix)
        self.displacement = np.sqrt(dtheta**2+dphi**2)

    def _CloneAlm(self,alm):
        almpack = (alm, self.ell, self.em, self.nside, self.npix)
        return SphericalMap(None, almpack)

    def laplace(self):
        return self._CloneAlm (self.Alm*1.0*self.ell*(self.ell+1))

    def inv_laplace(self):
        with np.errstate(divide='ignore'):
            x=self.Alm/(self.ell*(self.ell+1))
        x[self.ell==0]=0.0
        return self._CloneAlm(x)

    def dphi(self):
        return self._CloneAlm(self.Alm*(0+1j)*self.em)

    def dtheta(self):
        almp=np.array(self.Alm)
        almp*=np.sqrt((self.ell+self.em+1)*(self.ell-self.em+1)*
                      (2*self.ell+1)/(2*self.ell+3))
        almp[1:,:]=almp[:-1,:]
        almp[0,:]=0.0
        m2=self._CloneAlm(almp)
        m1=self._CloneAlm(self.Alm*(self.ell+1))
        theta = self.theta_pix
        mtot=-1*m1.A*np.cos(theta)/np.sin(theta)+m2.A*1/np.sin(theta)
        return SphericalMap(mtot)

    def displace_objects(self, theta, phi):
        ipix = hp.ang2pix(self.nside, theta, phi)
        dtheta = self.dtheta_map.A[ipix]
        dphi = self.dphi_map.A[ipix]/np.sin(theta)
        dd = np.sqrt(dtheta**2+dphi**2)
        alpha = np.arctan2(dphi,dtheta)
        ## Equation A15 from 0502469
        thetap = np.arccos(np.cos(dd)*np.cos(theta) - 
                           np.sin(dd)*np.sin(theta)*np.cos(alpha))
        phip = phi+np.arcsin(np.sin(alpha)*np.sin(dd)/np.sin(thetap))
        return thetap, phip

def create_blob_kappa(nside=512, phi0=np.pi/2, theta0=1.1):
    ''' Generatte kappa map with a blob in the center '''
    npix = nside**2*12
    kappa = np.zeros(npix)
    theta, phi = hp.pix2ang(nside,np.arange(npix))
    #phi[phi>np.pi] -= 2*np.pi
    kappa = SphericalMap(np.exp(-((phi-phi0)**2+(theta-theta0)**2)/
                                  (2*0.1**2)))
    kappa.compute_deriv()
    return kappa

def create_gaussian_kappa(ell, cell, nside=1024, seed=1):

    np.random.seed(seed)
    obsk = hp.sphtfunc.synfast(cell, nside=nside, lmax=2*nside-1,
                               pol=False)
    kappa = SphericalMap(obsk)
    kappa.compute_deriv()
    return kappa

def test_disp( disp_x, disp_y,
               rangex=[np.pi/2-1., np.pi/2+1.],
               rangey=[0.5, np.pi/2], ndots=1000, 
               factor=1, seed=1):
    ''' Test displacements by plotting a set of random points in phi, theta'''
    np.random.seed(seed)
    nside = hp.npix2nside(disp_x.size)

    phi   = np.random.uniform(rangex[0], rangex[1], ndots)
    theta = np.random.uniform(rangey[0], rangey[1], ndots)

    ipix = hp.ang2pix(nside, theta, phi)
    dphi = disp_x[ipix]
    dtheta = disp_y[ipix]
   
    ra = phi
    dec = np.pi/2-theta

    dd=np.sqrt(dphi**2+dtheta**2)
    alpha=np.arctan2(dphi, dtheta)
    ## Equation A15 from 0502469
    thetap=np.arccos(np.cos(dd)*np.cos(theta)-
                     np.sin(dd)*np.sin(theta)*np.cos(alpha))
    phip=phi+np.arcsin(np.sin(alpha)*np.sin(dd)/np.sin(thetap))

    rap = phip
    decp = np.pi/2-thetap 

    plt.figure()
    for t,p,t2,p2 in zip(dec, ra, decp, rap):
        plt.arrow(p, t, (p2-p)*factor, (t2-t)*factor, color='k', width=0.0003)
    plt.xlim(rangex[1], rangex[0])
    plt.ylim(np.pi/2-rangey[1], np.pi/2-rangey[0])

def read_kappa(fin, rebin=1, ellmin=None, ellmax=None, nell=100, 
                    smooth=15., cut_outliers=0):
    ''' Read kappa.fits.gz and compute power spectrum
        Input
        -----
        fin: input fits file with kappa map 
        rebin: bool - if True rebins power spectrum with l*(l+1) weights
        smooth: smooth map if != 0 by smooth value in arcmin
        cut_outliers: if True will set 1% extreme pixels to UNSEEN value
    '''

    a = fits.getdata(fin)[1]
    lya_kappa = a.kappa

    #-- masking values where there's no forests
    w0 = a.wkappa != 0.
    
    lya_kappa[~w0] = hp.UNSEEN
    if cut_outliers:
        w = (lya_kappa > np.percentile(lya_kappa[w0], 0.5)) & \
            (lya_kappa < np.percentile(lya_kappa[w0], 99.5))
        lya_kappa[~w] = hp.UNSEEN

    #-- smoothing in angular direction
    if smooth:
        lya_kappa = hp.smoothing(lya_kappa, fwhm=smooth/60*np.pi/180)

    #-- getting alm and auto power-spectrum 
    lya_alm = hp.map2alm(lya_kappa)
    lya_cl = hp.alm2cl(lya_alm)


    ell = np.arange(lya_cl.size)

    if rebin: 
        weights = 2.*ell+1.
        if not ellmin:
            ellmin = min(ell)
        if not ellmax:
            ellmax = max(ell)
        if nell==0:
            nell = ellmax-ellmin

        w = (ell>=ellmin)&(ell<=ellmax)
        index = np.floor( (ell[w]-ellmin)*1./(ellmax-ellmin)*nell ).astype(int)
        well = np.bincount( index, weights=weights[w])
        sell = np.bincount( index, weights=weights[w]*ell[w])
        scl  = np.bincount( index, weights=weights[w]*lya_cl[w])
        ell = sell/well
        lya_cl = scl/well
        
    return ell, lya_cl 



