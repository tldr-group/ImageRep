import numpy as np
import torch
import slicegan

def bernouli(vf, ls):
    errs = []
    for l in ls:
        std_theo = (1/l)*(vf*(1-vf))**0.5
        errs.append(100*((1.96*std_theo)/(vf*2**0.5)))
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
        errs.append(100*((1.96*std)/(vf*2**0.5)))
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
    for i in range(1):
        noise = torch.randn(1, 16, 4, lf, lf)
        noise = noise.cuda()
        img = netG(noise)
        img = slicegan.util.post_proc(img)
        imgs.append(img[0])
    img = torch.stack(imgs, 0)
    # print(img.shape)
    return img.float()

def fit_fac(err_exp, ls, vf, max_fac=200):
    err_fit = []
    err_exp = np.array(err_exp)
    for i in range(1, max_fac):
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

def tpc(img):
    tpc = []
    for sh in range(1,150):
        tpc.append(torch.mean(img[:, sh:]*img[:, :-sh]).item())
    return tpc
