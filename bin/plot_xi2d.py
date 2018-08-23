import numpy as np
import pylab as plt
import scipy as sp
import scipy.interpolate
import sys

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
    xim =sp.interpolate.griddata((data_rt, data_rp), xi_dist, \
                (rt[:, None], rp[None, :]), method='cubic')

    #-- create interpolator object
    xi2d = sp.interpolate.RectBivariateSpline(rt, rp, xim)
    return xi2d

xi2d = load_model(sys.argv[1]) 
rt = np.linspace(2, 198, 50)
rp = np.linspace(2, 198, 50)

plt.ion()
plt.figure()
R = xi2d(rt, rp, dx=1)*rt[:, None]

rt_plot = np.linspace(0, 200, 51)
rp_plot = np.linspace(0, 200, 51)
plt.pcolormesh(rt_plot, rp_plot, R.T, vmin=-5e-4, vmax=5e-4)
plt.colorbar()
plt.xlabel(r'$r_{\perp}$ [Mpc/h]')
plt.ylabel(r'$r_{\parallel}$ [Mpc/h]')
plt.title(r'$R = r_{\perp} ({\rm d}\xi/{\rm d}r_{\perp})$')
plt.show()
to = input()

