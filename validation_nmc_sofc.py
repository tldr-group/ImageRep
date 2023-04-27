from util import *
import  numpy as np
import matplotlib.pyplot as plt
import tifffile
import os

twod=True
errs_tpc = []
errs_stat = []
for pths, c in zip(['Graphite/Calendering','NMC/Calendering'], ['g', 'b']):
    d = os.listdir(pths)
    for i, d in enumerate(d):
        
        img = tifffile.imread(f'{pths}/{d}')
        print(img.shape, )

        for ph in np.unique(img):
            imph = np.zeros_like(img)
            imph[img==ph]=1
            l = min(np.array(img.shape[1:]))-1
            vf = np.mean(imph)
            err, n, tpc = make_error_prediction(imph[0,:l,:l] if twod else imph[:l,:l,:l], model_error=False, mxtpc=100)
            errs_tpc.append(err)
            x, y, z = img.shape
            vfs = []
            for y0 in range(0, y-l, l):
                for z0 in range(0, z-l, l):
                    if twod:
                        for x0 in range(x):
                            vfs.append(np.mean(imph[x0, y0:y0+l, z0:z0+l]))
                    else:
                        for x0 in range(0, x-l, l):
                            vfs.append(np.mean(imph[x0:x0+l, y0:y0+l, z0:z0+l]))
            std = np.std(vfs)
            print(len(vfs))
            errs_stat.append(100*((1.96*std)/vf))

i=10
plt.scatter(errs_tpc[:40-i], errs_stat[:40-i], c='g')
plt.scatter(errs_tpc[40:80-i], errs_stat[40:80-i], c='b')

plt.plot(np.arange(60), np.arange(60), c='black')

        
