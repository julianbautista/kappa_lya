from astropy.io import fits
from astropy.table import Table, join
import glob
import numpy as np
import pylab as P
import sys

def get_radec(indir, fout, lens=True):
    alld = glob.glob(indir+'/*.fits.gz')
    fout = open(fout, 'w')
    for i, deltafile in enumerate(alld):
        print(i, len(alld))
        hdus = fits.open(deltafile)
        for h in hdus[1:]:
            line = ''
            if lens:
                for key in ['THING_ID', 'PLATE', 'MJD', 'FIBERID', 'RA', 'DEC', 'Z', 'RA0', 'DEC0']:
                    line = line +' '+str(h.header[key])
            else:
                for key in ['THING_ID', 'PLATE', 'MJD', 'FIBERID', 'RA', 'DEC', 'Z']:
                    line = line +' '+str(h.header[key])
            print(line, file=fout)
    fout.close()

def plot_displacements(tab):
    t = Table(np.loadtxt(tab), 
            names=('THING_ID', 'PLATE', 'MJD', 
                   'FIBERID', 'RA', 'DEC', 'Z', 'RA0', 'DEC0'),
            dtype=(int, int, int, int, float, float, float, float, float))
   
    rand_ind = np.random.choice(np.arange(len(t)), \
                    replace=False, size=2000)

    P.figure()
    for i in rand_ind:
        x = t['RA0'][i]
        y = t['DEC0'][i]
        dx = t['RA'][i]-x
        dy = t['DEC'][i]-y 
        P.arrow(x, y, dx, dy, color='k', width=0.0003)
    P.xlim(t['RA'][rand_ind].min(), t['RA'][rand_ind].max())
    P.ylim(t['DEC'][rand_ind].min(), t['DEC'][rand_ind].max())
    P.show()

if sys.argv[1] == 'get_radec':
    get_radec(sys.argv[2], sys.argv[3])
elif sys.argv[1] == 'plot_disp':
    plot_displacements(sys.argv[2]) 
else:
    print('Need get_radec or plot_disp')

