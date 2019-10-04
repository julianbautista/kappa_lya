import numpy as np
import pylab as plt
import scipy as sp
import scipy.interpolate
import sys
import fitsio
import h5py

def load_model(modelfile) :

    data_rp, data_rt, xi_dist = np.loadtxt(modelfile, unpack=1)

    nrp = sum(abs(data_rt-data_rt[0])<1)
    nrt = sum(abs(data_rp-data_rp[0])<1)
    print(nrt, 'rt bins')
    print(nrp, 'rp bins')

    #-- get the larger value of the first separation bin to make a grid
    rt_min = data_rt.reshape(nrp, nrt)[:, 0].max()
    rt_max = data_rt.reshape(nrp, nrt)[:, -1].min()
    rp_min = data_rp.reshape(nrp, nrt)[0].max()
    rp_max = data_rp.reshape(nrp, nrt)[-1].min()
    print('rt_min rt_max', rt_min, rt_max)
    print('rp_min rp_max', rp_min, rp_max)

    #-- create the regular grid for griddata
    rt = np.linspace(rt_min, rt_max, nrt)
    rp = np.linspace(rp_min, rp_max, nrp)

    xim =sp.interpolate.griddata((data_rt, data_rp), xi_dist, \
                (rt[:, None], rp[None, :]), method='cubic')

    #-- create interpolator object
    xi2d = sp.interpolate.RectBivariateSpline(rt, rp, xim)
    return rt, rp, xi2d

def load_model_h5(modelfile, xifile):

    #- Read correlation function file to get rp and rt values 
    h = fitsio.FITS(xifile)
    data_rp = h[1]['RP'][:]
    data_rt = h[1]['RT'][:]
    data_co = h[1]['CO'][:]
    var = np.diag(data_co)
    hh = h[1].read_header()
    nrp = hh['NP']
    nrt = hh['NT']
    h.close()

    #-- get the larger value of the first separation bin to make a grid
    rp_min = data_rp.reshape(nrp, nrt)[0].max()
    rp_max = data_rp.reshape(nrp, nrt)[-1].min()
    rt_min = data_rt.reshape(nrp, nrt)[:, 0].max()
    rt_max = data_rt.reshape(nrp, nrt)[:, -1].min()

    ff = h5py.File(modelfile, 'r')
    keys = list(ff.keys())
    for k in keys:
        if k != 'best fit':
            base = k
            break
    fit = ff[base+'/fit'][...]
    ff.close()

    #-- create the regular grid for griddata
    rp = np.linspace(rp_min, rp_max, nrp)
    rt = np.linspace(rt_min, rt_max, nrt)
    xim = sp.interpolate.griddata((data_rt, data_rp), fit, \
            (rt[:, None], rp[None, :]), method='cubic')

    #-- create interpolator object
    xi2d = sp.interpolate.RectBivariateSpline(rt, rp, xim)

    return rt, rp, xi2d, var


modelfile = sys.argv[1]
if modelfile.endswith('.h5'):
    rt, rp, xi2d, var = load_model_h5(sys.argv[1], sys.argv[2]) 
else:
    rt, rp, xi2d = load_model(sys.argv[1])
        

plt.ion()
plt.figure(figsize=(12, 5))

#-- R is the weight in the kappa estimator = ( d xi/ d r_\perp ) * r_\perp
R = xi2d(rt, rp, dx=1)*rt[:, None]
R = np.log10(R**2)

xir2 = xi2d(rt, rp)*(rt[:, None]**2+rp[None, :]**2)

rt_plot = np.linspace(rt[0]-0.5*(rt[1]-rt[0]), rt[-1]+0.5*(rt[-1]-rt[-2]), rt.size+1)
rp_plot = np.linspace(rp[0]-0.5*(rp[1]-rp[0]), rp[-1]+0.5*(rp[-1]-rp[-2]), rp.size+1)

plt.subplot(121)
plt.pcolormesh(rt_plot, rp_plot, xir2.T, 
               vmin=np.percentile(xir2, 1), vmax=np.percentile(xir2, 99))
plt.xlabel(r'$r_{\perp}$ [Mpc/h]')
plt.ylabel(r'$r_{\parallel}$ [Mpc/h]')
plt.title(r'$r^2 \times \xi$')
plt.colorbar()
plt.subplot(122)
plt.pcolormesh(rt_plot, rp_plot, R.T, 
               vmin=np.percentile(R, 0.5), vmax=np.percentile(R, 99.9))
plt.colorbar()
plt.xlabel(r'$r_{\perp}$ [Mpc/h]')
plt.ylabel(r'$r_{\parallel}$ [Mpc/h]')
plt.title(r'$R^2 = (r_{\perp} ({\rm d}\xi/{\rm d}r_{\perp}))^2$')
plt.show()
to = input()

