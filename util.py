import numpy as np
import torch
import slicegan
from scipy.optimize import curve_fit
import kneed
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm

def load_generator(Project_path):
    img_size, img_channels, scale_factor = 64, 1, 1
    z_channels = 16
    lays = 6
    dk, gk = [4] * lays, [4] * lays
    ds, gs = [2] * lays, [2] * lays
    df, gf = [img_channels, 64, 128, 256, 512, 1], [
        z_channels,
        512,
        256,
        128,
        64,
        img_channels,
    ]
    dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

    ## Create Networks
    netD, netG = slicegan.networks.slicegan_nets(
        Project_path, False, "grayscale", dk, ds, df, dp, gk, gs, gf, gp
    )
    netG = netG()
    netG = netG.cuda()
    return netG


def generate_image(netG, Project_path, lf=50, threed=False, reps=50):
    try:
        netG.load_state_dict(torch.load(Project_path + "_Gen.pt"))
    except:  # if the image is greayscale it's excepting because there's only 1 channel
        return torch.tensor(0)
    netG.eval()
    imgs = []
    for i in range(reps):
        noise = torch.randn(1, 16, lf if threed else 4, lf, lf)
        noise = noise.cuda()
        img = netG(noise, threed)
        img = slicegan.util.post_proc(img)
        imgs.append(img.cpu() if threed else img[0])
    img = torch.stack(imgs, 0)
    return img.float()


def make_sa(img, batch=True):
    if not batch:
        img = np.expand_dims(img, 0)
        sa = np.zeros_like(img)
    else:
        sa = torch.zeros_like(img)
    dims = len(img.shape)
    sa[:, 1:] += img[:, 1:] * (1 - img[:, :-1])
    sa[:, :, 1:] += img[:, :, 1:] * (1 - img[:, :, :-1])
    sa[:, :-1] += (1 - img[:, 1:]) * img[:, :-1]
    sa[:, :, :-1] += (1 - img[:, :, 1:]) * img[:, :, :-1]
    if dims == 4:
        sa[:, :, :, 1:] += img[:, :, :, 1:] * (1 - img[:, :, :, :-1])
        sa[:, :, :, :-1] += (1 - img[:, :, :, 1:]) * img[:, :, :, :-1]
    if not batch:
        sa = sa[0]
    sa[sa>0] = 1
    return sa


def make_sa_old(img, batch=True):
    if not batch:
        img = np.expand_dims(img, 0)
        sa = np.zeros_like(img)
    else:
        sa = torch.zeros_like(img)
    dims = len(img.shape)
    sa[:, 1:] += img[:, 1:] * (1 - img[:, :-1])
    sa[:, :, 1:] += img[:, :, 1:] * (1 - img[:, :, :-1])
    sa[:, :-1] += (1 - img[:, 1:]) * img[:, :-1]
    sa[:, :, :-1] += (1 - img[:, :, 1:]) * img[:, :, :-1]
    if dims == 4:
        sa[:, :, :, 1:] += img[:, :, :, 1:] * (1 - img[:, :, :, :-1])
        sa[:, :, :, :-1] += (1 - img[:, :, :, 1:]) * img[:, :, :, :-1]
    if not batch:
        sa = sa[0]
    return sa


def conjunction_img_for_tpc(img, x, y, z, threed):
    if threed:
        if x == 0:
            if y == 0:
                if z == 0:
                    con_img = img * img
                else:
                    con_img = img[..., :-z] * img[..., z:]
            else:
                if z == 0:
                    con_img = img[..., :-y, :] * img[..., y:, :]
                else:
                    con_img = img[..., :-y, :-z] * img[..., y:, z:]
        else:
            if y == 0:
                if z == 0:
                    con_img = img[..., :-x, :, :] * img[..., x:, :, :]
                else:
                    con_img = img[..., :-x, :, :-z] * img[..., x:, :, z:]
            else:
                if z == 0:
                    con_img = img[..., :-x, :-y, :] * img[..., x:, y:, :]
                else:
                    con_img = img[..., :-x, :-y, :-z] * img[..., x:, y:, z:]
    else:
        if x == 0:
            if y == 0:
                con_img = img * img
            else:
                con_img = img[..., :-y] * img[..., y:]
        else:
            if y == 0:
                con_img = img[..., :-x, :] * img[..., x:, :]
            else:
                con_img = img[..., :-x, :-y] * img[..., x:, y:]
    return con_img


def tpc_radial(img, mx=100, threed=False):
    img = torch.tensor(img, device=torch.device("cuda:0")).float()
    tpc = {i:[0,0] for i in range(mx+1)}
    for x in range(0, mx):
        if (x%10) == 0:
            print(f'{x}% complete')
        for y in range(0, mx):
            for z in range(0, mx if threed else 1):
                d = (x**2 + y**2 + z**2) ** 0.5
                if d < mx:
                    remainder = d%1
                    con_img = conjunction_img_for_tpc(img, x, y, z, threed)
                    con_img_tpc = torch.mean(con_img).cpu()
                    weight_floor = 1-remainder
                    weight_ceil = remainder
                    tpc[int(d)][0] += weight_floor 
                    tpc[int(d)][1] += con_img_tpc*weight_floor
                    tpc[int(d)+1][0] += weight_ceil 
                    tpc[int(d)+1][1] += con_img_tpc*weight_ceil
    tpc_weights = [tpc[key][0] for key in tpc.keys()]
    tpcfin = [tpc[key][1]/tpc[key][0] for key in tpc.keys()]
    tpcfin = np.array(tpcfin, dtype=np.float64)
    return np.arange(mx+1, dtype=np.float64), tpc_weights, tpcfin  


def old_tpc_radial(img, mx=100, threed=False):
    img = torch.tensor(img, device=torch.device("cuda:0")).float()
    tpc = {i: [] for i in range(1, mx)}
    for x in range(1, mx):
        for y in range(1, mx):
            for z in range(1, mx if threed else 2):
                d = int((x**2 + y**2 + z**2) ** 0.5)
                if d < mx:
                    if threed:
                        tpc[d].append(
                            torch.mean(
                                img[..., :-x, :-y, :-z] * img[..., x:, y:, z:]
                            ).cpu()
                        )
                    else:
                        tpc[d].append(
                            torch.mean(img[..., :-x, :-y] * img[..., x:, y:]).cpu()
                        )

    tpcfin = []
    for key in tpc.keys():
        tpcfin.append(np.mean(tpc[key]).item())
    tpcfin = np.array(tpcfin, dtype=np.float64)
    return tpcfin  


def stat_analysis_error(img, edge_lengths, img_dims, vf, threed=False, conf=0.95):
    err_exp_vf = real_image_stats(img, edge_lengths, vf, threed=threed)
    err_model_vf, fac_vf = fit_fac(err_exp_vf, img_dims, vf)
    shape = [np.array(img.size()[-3:] if threed else img.size()[-2:])]
    return bernouli(vf, ns_from_dims(shape, fac_vf), conf=conf)


def real_image_stats(img, ls, vf, repeats=1000, threed=False, z_score=1.96):
    errs = []
    for l in ls:
        if (l%50) == 0:
            print(f'length = {l}')
        vfs = []
        if not threed:
            for i in range(repeats):
                bm, xm, ym = img.shape
                x = torch.randint(0, xm - l, (1,))
                y = torch.randint(0, ym - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l, y : y + l]
                vfs.append(torch.mean(crop).cpu())
        else:
            repeats = 500 - l.item()
            for i in range(repeats):
                bm, xm, ym, zm = img.shape
                x = torch.randint(0, xm - l, (1,))
                y = torch.randint(0, ym - l, (1,))
                z = torch.randint(0, zm - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l, y : y + l, z : z + l]
                vfs.append(torch.mean(crop).cpu())
        vfs = np.array(vfs)
        std = np.std(vfs)
        errs.append(100 * ((z_score * std) / vf))
    print(f'error = {errs}')
    return errs


def bernouli(vf, ns, conf=0.95):
    errs = []
    for n in ns:
        std_theo = ((1 / n) * (vf * (1 - vf))) ** 0.5
        errs.append(100 * (stats.norm.interval(conf, scale=std_theo)[1] / vf))
    return np.array(errs, dtype=np.float64)


def fit_fac(err_exp, img_dims, vf, max_fac=100):
    err_exp = np.array(err_exp)
    fac = test_fac_set(err_exp, vf, np.arange(1, max_fac, 1), img_dims)
    fac = test_fac_set(err_exp, vf, np.linspace(fac - 1, fac + 1, 20), img_dims)
    err_model = bernouli(vf, ns_from_dims(img_dims, fac))
    return err_model, fac


def ns_from_dims(img_dims, ir):
    den = ir ** (len(img_dims[0]))
    return [np.prod(i + ir - 1) / den for i in img_dims]

def dims_from_n(n, shape, fac, dims):
    den = fac ** dims
    if shape=='equal':
        return (n*den)**(1/dims)-fac+1
    else:
        if dims==len(shape):
            raise ValueError('cannot define all the dimensions')
        if len(shape)==1:
            return ((n*den)/(shape[0]+fac-1))**(1/(dims-1))-fac+1
        else:
            return ((n*den)/((shape[0]+fac-1) * (shape[1]+fac-1)))-fac+1


def test_fac_set(err_exp, vf, facs, img_dims):
    err_fit = []
    for fac in facs:
        ns = ns_from_dims(img_dims, fac)
        err_model = bernouli(vf, ns)
        err = np.mean(abs(err_exp - err_model))
        err_fit.append(err)
    fac = facs[np.argmin(err_fit)].item()
    return fac


def tpc_fit(x, a, b, c):
    return a * np.e ** (-b * x) + c


def percentage_error(y_true, y_pred):  
    return (y_true - y_pred) / y_true


def mape(y_true, y_pred):  # mean absolute percentage error
    return np.mean(np.abs(percentage_error(y_true, y_pred))) 


def mape_linear_objective(params, y_pred, y_true):
    y_pred_new = linear_fit(y_pred, *params)
    return mape(y_true, y_pred_new) 


def linear_fit(x, m, b):
    return m * x + b


def tpc_to_fac(tpc_dist, tpc):
    tpc, tpc_dist = np.array(tpc), np.array(tpc_dist)
    vf = tpc[0]
    pred_fac = (1/(vf-vf*vf))*np.trapz(tpc - (vf*vf), x=tpc_dist)
    return pred_fac  


def old_tpc_to_fac(x, y):
    bounds = ((-np.inf, 0.01, -np.inf), (np.inf, np.inf, np.inf))
    coefs_poly3d, _ = curve_fit(tpc_fit, x, y, bounds=bounds)
    y_data = tpc_fit(x, *coefs_poly3d)
    kneedle = kneed.KneeLocator(
        x, y_data, S=1.0, curve="convex", direction="decreasing"
    )
    return kneedle.knee


def make_error_prediction(img, conf=0.95, err_targ=0.05,  model_error=True, correction=True, mxtpc=100, shape='equal', met='vf'):
    vf = img.mean()
    dims = len(img.shape)
    print(f'starting tpc radial')
    tpc_dist, tpc_weights, tpc = tpc_radial(img, threed=dims == 3, mx=mxtpc)
    print(f'starting tpc to fac')
    fac = tpc_to_fac(tpc_dist, tpc_weights, tpc)
    print(f'pred fac = {fac}')
    n = ns_from_dims([np.array(img.shape)], fac)
    # print(n, fac)
    std_bern = ((1 / n[0]) * (vf * (1 - vf))) ** 0.5
    std_model, slope, intercept = get_model_params(f'{dims}d{met}') 
    if not correction:
        slope, intercept = 1, 0
    if model_error:
        # print(std_bern)
        bounds = [(conf*1.001, 1)]
        args = (conf, std_bern, std_model, vf, slope, intercept)
        err_for_img = minimize(optimize_error_conf_pred, conf**0.5, args, bounds=bounds).fun
        args = (conf, std_model, vf, slope, intercept, err_targ)
        n_for_err_targ = minimize(optimize_error_n_pred, conf**0.5, args, bounds=bounds).fun
        # print(n, n_for_err_targ, fac)
    else:
        z = stats.norm.interval(conf)[1]
        err_for_img = (z*std_bern/vf)*slope+intercept
        # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)
        n_for_err_targ = vf * (1 - vf) * (z/ ((err_targ -intercept)/slope * vf)) ** 2

        # print(n_for_err_targ, n, fac)
    l_for_err_targ = dims_from_n(n_for_err_targ, shape, fac, dims)
    return err_for_img, l_for_err_targ, tpc_dist, tpc


def make_error_prediction_old(img, conf=0.95, err_targ=0.05,  model_error=True, correction=True, mxtpc=100, shape='equal', met='vf'):
    vf = img.mean()
    dims = len(img.shape)
    tpc = tpc_radial(img, threed=dims == 3, mx=mxtpc)
    cut = max(20, np.argmin(tpc))
    tpc = tpc[:cut]
    x = np.arange(len(tpc))
    fac = tpc_to_fac(x, tpc)
    n = ns_from_dims([np.array(img.shape)], fac)
    # print(n, fac)
    std_bern = ((1 / n[0]) * (vf * (1 - vf))) ** 0.5
    std_model, slope, intercept = get_model_params(f'{dims}d{met}') 
    if not correction:
        slope, intercept = 1, 0
    if model_error:
        # print(std_bern)
        bounds = [(conf*1.001, 1)]
        args = (conf, std_bern, std_model, vf, slope, intercept)
        err_for_img = minimize(optimize_error_conf_pred, conf**0.5, args, bounds=bounds).fun
        args = (conf, std_model, vf, slope, intercept, err_targ)
        n_for_err_targ = minimize(optimize_error_n_pred, conf**0.5, args, bounds=bounds).fun
        # print(n, n_for_err_targ, fac)
    else:
        z = stats.norm.interval(conf)[1]
        err_for_img = (z*std_bern/vf)*slope+intercept
        # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)
        n_for_err_targ = vf * (1 - vf) * (z/ ((err_targ -intercept)/slope * vf)) ** 2

        # print(n_for_err_targ, n, fac)
    l_for_err_targ = dims_from_n(n_for_err_targ, shape, fac, dims)
    return err_for_img, l_for_err_targ, tpc

def optimize_error_conf_pred(bern_conf, total_conf, std_bern, std_model, vf, slope, intercept):
    model_conf = total_conf/bern_conf
    err_bern = ((stats.norm.interval(bern_conf, scale=std_bern)[1]*slope)/vf)+intercept
    err_model = stats.norm.interval(model_conf, scale=std_model)[1]
    # print(stats.norm.interval(bern_conf, scale=std_bern)[1], slope, intercept, vf , err_model, err_bern)
    # print(err_bern, err_model, err_bern * (1 + err_model))
    # print(err_bern * (1 + err_model))
    return err_bern * (1 + err_model)

def optimize_error_n_pred(bern_conf, total_conf, std_model, vf, slope, intercept, err_targ):
    # print(bern_conf)
    model_conf = total_conf/bern_conf
    z1 = stats.norm.interval(bern_conf)[1]
    err_model = stats.norm.interval(model_conf, scale=std_model)[1]
    # print(err_model)
    num = -(err_model+1)**2 * slope**2 * (vf-1) * z1**2
    den = (-intercept * err_model + err_targ - intercept)**2 * vf
    return num/den


def get_model_params(imtype):  # see model_param.py for the appropriate code that was used.
    params= {'2dvf':[0.3863185623920709,2.565789407003636, 0.0003318955996696201],
             '2dsa':[0.3697788866716134,0.8522276248407129, 0.007904077381387118],
             '3dvf':[0.5761622825137038,1.588087103916383, 0.0036686523118274283],
             '3dsa':[0.47390094290400503,0.8922479454738727, 0.007302588617491829]}
    return params[imtype]


def cld(img):
    """
    Calculating the chord length distribution function
    """
    iterations = 150
    return_length = 150
    sums = torch.zeros(iterations)

    # for ang in torch.linspace(0,180, 20):
    sm = []
    # cur_im = rotate(torch.tensor(img), ang.item())
    # cur_im = torch.round(cur_im[0,0,280:-280, 280:-280])
    cur_im = torch.tensor(img, device=torch.device("cuda:0"))
    for i in range(1, iterations):
        sm.append(
            torch.sum(cur_im)
        )  # sum of all current "live" pixels that are part of an i length chord
        cur_im = (
            cur_im[1:] * cur_im[:-1]
        )  # deleting 1 pixel for each chord, leaving all pixels that are part of an i+1 length chords
    sm.append(torch.sum(cur_im))
    sums += torch.tensor(sm)

    cld = sums.clone()
    cld[-1] = 0  # the assumption is that the last sum is 0
    for i in range(1, iterations):  # calculation of the chord lengths by the sums
        cld[-(i + 1)] = (sums[-(i + 1)] - sums[-i] - sum(cld[-i:])).cpu().item()
    cld = np.array(cld)
    return cld / np.sum(cld)
