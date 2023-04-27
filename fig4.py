from util import *
import  numpy as np
import matplotlib.pyplot as plt
import tifffile
import os

twod=True
errs_tpc = []
errs_stat = []
imgsizes = []
for pths, c in zip(['Graphite/Calendering','NMC/Calendering'], ['g', 'b']):
    d = os.listdir(pths)
    for i, d in enumerate(d):  
        img = tifffile.imread(f'{pths}/{d}')
        shape = np.array(img.shape)
        mindim = np.argmin(shape)
        img = np.swapaxes(img, mindim, 0)
        print(img.shape)
        for ph in np.unique(img):
            imph = np.zeros_like(img)
            imph[img==ph]=1
            vf = np.mean(imph)
            err, n, tpc = make_error_prediction(imph[0], correction=True, model_error=True, mxtpc=50)
            errs_tpc.append(err*100)
            vfs = np.mean(imph, axis=(1,2))
            std = np.std(vfs)
            errs_stat.append(100*((1.96*std)/vf))
            imgsizes.append(np.array(imph.shape))

i1, i2 =10, 34
cs = np.array([np.prod(i) for i in imgsizes])
fig, ax = plt.subplots(1)
# ax.scatter(errs_stat[:4], errs_tpc[:4], label='Graphite')
# ax.scatter(errs_stat[7:11],errs_tpc[7:11], label='NMC')
ax.scatter(errs_stat[:40], errs_tpc[:40], label='Graphite')
ax.scatter(errs_stat[40:50],errs_tpc[40:50], label='NMC')
ax.set_xlabel('Error from statistical analysis')
ax.set_ylabel('Error from tpc analysis')
ax.set_title(f'Electrode error predictions')
ax.set_aspect(1)
ax.plot(np.arange(50), np.arange(50), c='black')
ax.legend()
plt.savefig('fig4.pdf')
