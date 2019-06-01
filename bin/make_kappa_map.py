from astropy.table import Table, join
import numpy as np
import healpy as hp


def get_radec(thid_in, ra_in, dec_in, thid_out):
    ''' Function to obtain RA, DEC from THING_ID values'''

    win = np.argsort(thid_in)
    wout = np.searchsorted(thid_in[win], thid_out)
    return ra_in[win][wout], dec_in[win][wout]

def radec_to_cart(ra, dec):

    ra_r = np.radians(ra)
    dec_r = np.radians(dec)
    x = np.cos(dec_r)*np.cos(ra_r)
    y = np.cos(dec_r)*np.sin(ra_r)
    z = np.sin(dec_r)
    return x, y, z

def cart_to_radec(x, y, z):

    dist = np.sqrt(x**2+y**2+z**2)
    dec = 90 - np.degrees(np.arccos(z / dist))
    ra = np.degrees(np.arctan2(y, x))
    ra[ra < 0] += 360
    return ra, dec

def get_mid_radec(ra1, dec1, ra2, dec2):

    x1, y1, z1 = radec_to_cart(ra1, dec1)
    x2, y2, z2 = radec_to_cart(ra2, dec2)

    x_mid = 0.5*(x2+x1)
    y_mid = 0.5*(y2+y1)
    z_mid = 0.5*(z2+z1)

    ra_mid, dec_mid = cart_to_radec(x_mid, y_mid, z_mid)
   
    return ra_mid, dec_mid 

def make_map_midpoint(nside, ra_mid, dec_mid, skappa, wkappa):

    th_mid, phi_mid = np.pi/2-np.radians(dec_mid), np.radians(ra_mid)
    pix_mid = hp.ang2pix(nside, th_mid, phi_mid)

    smap = np.bincount(pix_mid, weights=skappa, 
                    minlength=hp.nside2npix(nside))
    wmap = np.bincount(pix_mid, weights=wkappa, 
                    minlength=hp.nside2npix(nside))

    kappa_map = smap*0
    w = wmap !=0
    kappa_map[w] = smap[w]/wmap[w]

    return kappa_map, wmap


#-- read list of kappas
kap = Table.read('/mnt/lustre/bautista/lya/kappa_lya/kappa_lists/kappalist-noiseless-cnstwe-lensed-truecorr-rtmax70-rpmax40.fits')
#kap = Table.read('kappalist-desi-noiseless-fromdeltas-rt20-rp40.fits')

#-- read QSO catalog with THING_ID, RA and DEC information 
drq = Table.read('/mnt/lustre/eboss/lya/mocks/DRQ_mocks.fits')
#drq = Table.read('/mnt/lustre/youless/desi/zcat_desi_drq.fits')

drq.remove_columns(['Z', 'PLATE', 'MJD', 'FIBERID'])

ra1, dec1 = get_radec(drq['THING_ID'].data, drq['RA'].data, drq['DEC'].data,
                      kap['THID1'].data) 
ra2, dec2 = get_radec(drq['THING_ID'].data, drq['RA'].data, drq['DEC'].data,
                      kap['THID2'].data) 
ra_mid, dec_mid = get_mid_radec(ra1, dec1, ra2, dec2)

skappa = kap['SKAPPA'].data
wkappa = kap['WKAPPA'].data

nside = 64
kappamap, wmap = make_map_midpoint(nside, ra_mid, dec_mid, skappa, wkappa)

