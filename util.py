import numpy as np
import torch
import slicegan
from torchvision.transforms.functional import rotate
from scipy.optimize import curve_fit
import kneed

def bernouli(vf, ls):
    errs = []
    for l in ls:
        std_theo = (1/l)*(vf*(1-vf))**0.5
        errs.append(100*((1.96*std_theo)/vf))
    return np.array(errs)

def real_image_stats(img, ls, vf, repeats=1000, threed=True):
    errs = []
    for l in ls:
        vfs = []
        for i in range(repeats):
            if not threed:
                xm, ym = img.shape[:2]
                x = torch.randint(0, xm-l, (1,))
                y = torch.randint(0, ym-l, (1,))
                crop = img[x:x+l, y:y+l]
            else:
                bm, xm, ym = img.shape
                x = torch.randint(0, xm-l, (1,))
                y = torch.randint(0, ym-l, (1,))
                # z = torch.randint(0, zm-l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x:x+l, y:y+l]
            vfs.append(torch.mean(crop).cpu())
        std = np.std(vfs)
        errs.append(100*((1.96*std)/vf))
    return errs

def make_sa(img, twod=False):
    if twod:
        img = np.expand_dims(img, 0)
        sa = np.zeros_like(img)

    else:
        sa = torch.zeros_like(img)
    sa[:, 1:] += img[:, 1:] * (1 - img[:, :-1])
    sa[:, :, 1:] += img[:, :, 1:] * (1-img[:, :, :-1])
    sa[:, :-1] += (1-img[:, 1:]) * img[:, :-1]
    sa[:, :, :-1] += (1-img[:, :, 1:]) * img[:, :, :-1]
    if twod:
        sa = sa[0]
    return sa

def real_image_sa(img, ls, sa, repeats=1000, threed=True):
    errs = []
    for l in ls:
        print(l)
        sas = []
        for i in range(repeats):
            if not threed:
                xm, ym = img.shape[:2]
                x = torch.randint(0, xm-l, (1,))
                y = torch.randint(0, ym-l, (1,))
                crop = img[x:x+l, y:y+l]
            else:
                bm, xm, ym = img.shape
                x = torch.randint(0, xm-l, (1,))
                y = torch.randint(0, ym-l, (1,))
                # z = torch.randint(0, zm-l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x:x+l, y:y+l]
            sa1 = torch.mean(crop[:, 1:] * (1-crop[:, :-1])).cpu()
            sa2 = torch.mean(crop[:, :, 1:] * (1-crop[:, :, :-1])).cpu()
            sa3 = torch.mean(1-img[:, 1:] * (img[:, :-1])).cpu()
            sa4 = torch.mean(1-img[:, :, 1:] * (img[:, :, :-1])).cpu()
            sas.append(sa1+sa2+sa3+sa4)
        std = np.std(sas)
        errs.append(100*((1.96*std)/sa))
    return errs

def load_generator(Project_path):
    img_size, img_channels, scale_factor = 64, 1, 1
    z_channels = 16
    lays = 6
    dk, gk = [4] * lays, [4] * lays
    ds, gs = [2] * lays, [2] * lays
    df, gf = [img_channels, 64, 128, 256, 512, 1], [z_channels, 512, 256, 128, 64,
                                                    img_channels]
    dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

    ## Create Networks
    netD, netG = slicegan.networks.slicegan_nets(Project_path, False, 'grayscale', dk, ds,
                                        df, dp, gk, gs, gf, gp)
    netG = netG()
    netG = netG.cuda()
    return netG

def generate_image(netG, Project_path, lf=50):
    
    try:
        netG.load_state_dict(torch.load(Project_path + '_Gen.pt'))
    except:
        return torch.tensor(0)
    netG.eval()
    imgs = []
    for i in range(50):
        noise = torch.randn(1, 16, 4, lf, lf)
        noise = noise.cuda()
        img = netG(noise)
        img = slicegan.util.post_proc(img)
        imgs.append(img[0])
    img = torch.stack(imgs, 0)
    # print(img.shape)
    return img.float()

def fit_fac(err_exp, ls, vf, max_fac=100):
    err_fit = []
    err_exp = np.array(err_exp)
    for i in range(1, max_fac, 1):
        ls_test = [l/i for l in ls]
        err_model = bernouli(vf, ls_test)
        err = np.mean(abs(err_exp - err_model))
        err_fit.append(err)
    fac = np.argmin(err_fit)+1
    err_fit = []
    facs = torch.linspace(fac-1, fac+1, 20)
    for i in facs:
        ls_test = [l/i for l in ls]
        err_model = bernouli(vf, ls_test)
        err = np.mean(abs(err_exp - err_model))
        err_fit.append(err)
    fac = facs[np.argmin(err_fit)].item()
    ls_model = np.linspace(ls[0]/fac, (ls[-1]*2)/fac, 100)
    err_model = bernouli(vf,  ls_model)
    return err_model, [l*fac for l in ls_model], fac

def tpc_radial(img, mx=100, threed=True):
    img = torch.tensor(img, device=torch.device('cuda:0')).float()
    tpc = {i:[] for i in range(1,mx)}
    for x in range(1,mx):
        for y in range(1,mx):
            d = int((x**2 + y**2)**0.5)
            if d < mx:
                tpc[d].append(torch.mean(img[..., :-x, :-y]*img[..., x:, y:]).cpu())
    tpcfin = []
    for key in tpc.keys():
        tpcfin.append(np.mean(tpc[key]).item())
    return np.array(tpcfin)

def cld(img):
    '''
    Calculating the chord length distribution function
    '''
    iterations = 150
    return_length = 150
    sums = torch.zeros(iterations)
    
    # for ang in torch.linspace(0,180, 20):
    sm = []
    # cur_im = rotate(torch.tensor(img), ang.item())
    # cur_im = torch.round(cur_im[0,0,280:-280, 280:-280])
    cur_im = torch.tensor(img, device=torch.device('cuda:0'))
    for i in range(1, iterations):
        sm.append(torch.sum(cur_im))  # sum of all current "live" pixels that are part of an i length chord
        cur_im = cur_im[1:] * cur_im[:-1]  # deleting 1 pixel for each chord, leaving all pixels that are part of an i+1 length chords
    sm.append(torch.sum(cur_im))
    sums += torch.tensor(sm)

    cld = sums.clone()
    cld[-1] = 0  # the assumption is that the last sum is 0
    for i in range(1, iterations):  # calculation of the chord lengths by the sums
        cld[-(i+1)] = (sums[-(i+1)] - sums[-i] - sum(cld[-i:])).cpu().item()
    cld = np.array(cld)
    return cld/np.sum(cld)

def tpc_fit(x, a, b, c):
    return a*np.e**(-b*x) + c

def linear_fit(x, a, b):
    return a*x + b

def tpc_to_fac(x, y):
    bounds = ((-np.inf, 0.01, -np.inf), (np.inf, np.inf, np.inf))
    coefs_poly3d, _ = curve_fit(tpc_fit, x, y, bounds=bounds)
    y_data = tpc_fit(x,*coefs_poly3d)
    kneedle = kneed.KneeLocator(x, y_data, S=1.0, curve="convex", direction="decreasing")
    return kneedle.knee

def make_error_prediction(img, coeffs=None, err_targ=5, z=1.96):
    safety_factor = 2
    vf = img.mean()
    tpc = tpc_radial(img)
    x = np.arange(len(tpc))
    fac = tpc_to_fac(x, tpc)
    err_targ/safety_factor
    l_img = int((img.shape[0]*img.shape[1])**0.5)
    err_for_img = bernouli(vf, [l_img/fac])[0] * safety_factor
    n_for_err_targ = vf*(1-vf)/(err_targ*0.01*vf/z)**2
    l_for_err_targ = int((n_for_err_targ*fac**2)**0.5)
    return err_for_img, l_for_err_targ

