import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from SphericalDiff import *

plt.ion()


#generatte kappa map with a blob in the center
def create_blob(nside=512):
    npix = nside**2*12
    kappa = np.zeros(npix)
    theta, phi = hp.pix2ang(nside,np.arange(npix))
    #phi[phi>np.pi] -= 2*np.pi
    kappa = SphericalMap(np.exp(-((phi-np.pi)**2+(theta-(1.1))**2)/(2*0.1**2)))
    kappa.compute_deriv()
    return kappa

def test_disp(kappa):
    #generate some points to displace
    ndots = 1000
    theta = np.arccos(np.random.uniform(-np.sin(1), np.sin(1.4), ndots))
    phi = np.random.uniform(-1+np.pi, 1+np.pi, ndots)

    thetap, phip = kappa.DisplaceObjects(theta,phi)

    ## plot displacements, there are issues 
    ## with projection hence weird at the edges
    plt.figure(figsize=(8,8))
    for t,p,t2,p2 in zip(np.cos(theta), phi, np.cos(thetap), phip):
        plt.plot([p,p2],[t,t2],'r-')
    plt.plot(phip, np.cos(thetap), 'b.')


    theta_pix, phi_pix = hp.pix2ang(kappa.Nside, np.arange(kappa.Npix))
    ## plot of displacement size
    dtheta = kappa.dtheta_map.A
    dphi = kappa.dphi_map.A/np.sin(theta_pix)
    hp.mollview(dtheta**2+dphi**2, rot=(180, 0, 0))

    
    


