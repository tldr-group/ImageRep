import numpy as np
import torch
import slicegan
from scipy.optimize import curve_fit
import kneed
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm
from matplotlib import pyplot as plt
import time


def generate_image(netG, Project_path, slice_dim, lf=50, threed=False, reps=50):
    try:
        netG.load_state_dict(torch.load(Project_path + "_Gen.pt"))
    except:  # if the image is greayscale it's excepting because there's only 1 channel
        return torch.tensor(0)
    netG.eval()
    imgs = []
    z_channels = 32
    plot_profiles = []
    img_size = [450, 450, 450] if threed else [64, 1500, 1500]
    for i in range(reps):
        img_step_size = 64
        lfs = np.array([(l-3) // img_step_size + 7 for l in img_size])
        noise = torch.randn(1, z_channels, *lfs)
        noise = torch.permute(noise, (0,1) + tuple(((torch.arange(3) - slice_dim) % 3).numpy() + 2)) 
        noise = noise.cuda()
        img = netG(noise, threed, slice_dim)
        img = slicegan.util.post_proc(img)
        imgs.append(img.cpu())
        # time1 = time.time()
        # fft_calc = p2_crosscorrelation(imgs[0], imgs[0])
        # print(f'fft calc time = {time.time() - time1}')
    # plot_profiles = np.stack(plot_profiles)
    # profile_stds = np.std(plot_profiles, axis=0)
    # print(f'mean stds = {np.mean(profile_stds)}, std stds = {np.std(profile_stds)}')
    # profile_means = np.mean(plot_profiles, axis=0)
    # print(f'std means = {np.std(profile_means)}')
    # plt.errorbar(np.arange(64), profile_means, profile_stds, linestyle='None', marker='*')
    # plt.title(f'volume fraction mean and std for each of the 64 slices.')
    # plt.xlabel(f'first dimension index (slice)')
    # plt.ylabel(f'mean volume fraction (with std)')
    # plt.show()
    img = torch.stack(imgs, 0)
    return img.float()


def make_sas(img, batch=True):  # TODO check this works
    if not batch:
        img = torch.unsqueeze(img, 0)
    sa_lr = torch.zeros_like(img)
    sa_ud = torch.zeros_like(img)
    dims = len(img.shape)
    sa_lr[:, :, :-1] = (img[:, :, 1:] + img[:, :, :-1]) % 2
    sa_ud[:, :-1] = (img[:, 1:] + img[:, :-1]) % 2
    sas = [sa_lr, sa_ud]
    if dims == 4:  # 3d
        sa_z = torch.zeros_like(img)
        sa_z[:, :, :, :-1] += (img[:, :, :, 1:] + img[:, :, :, :-1]) % 2
        sas.append(sa_z)
    if not batch:
        sas = [sa[0] for sa in sas]
    return sas


def sa_map_from_sas(sa_images):
    # For memory issues, this calculation is the same as:
    # return torch.stack(sa_images, dim=0).sum(dim=0)
    sa_maps = [[sa_images[map_idx][k, ...] for map_idx in range(len(sa_images))] for k in range(sa_images[0].size()[0])]
    return torch.stack([torch.stack(maps, dim=0).sum(dim=0) for maps in sa_maps], dim=0)


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


def tpc_radial(img_list, mx=100, threed=False):
    mxs = [mx] * 3 if threed else [mx] * 2
    tpcfin_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        img = torch.tensor(img, device=torch.device("cuda:0")).float()
        tpc = {i:[0,0] for i in range(mx+1)}
        for dim_along in range(0, 3 if threed else 1):
            mxs_cur = mxs.copy()
            mxs_cur[dim_along] = 1 if threed else mx
            for x in range(0, mxs_cur[0]):
                for y in range(0, mxs_cur[1]):
                    for z in range(0, mxs_cur[2] if threed else 1):
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
    
        tpcfin = [tpc[key][1]/tpc[key][0] for key in tpc.keys()]
        tpcfin = np.array(tpcfin, dtype=np.float64)
        tpcfin_list.append(tpcfin)
    return np.arange(mx+1, dtype=np.float64), tpcfin_list  


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

def tpc_horizontal(img_list, mx=100, threed=False):
    tpcfin_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        img = torch.tensor(img, device=torch.device("cuda:0")).float()
        tpc = [torch.mean(img).cpu()]  # vf is tpc[0]
        for d in range(1, mx):
            tpc_1 = torch.mean(img[..., :-d] * img[..., d:]).cpu()
            tpc_2 = torch.mean(img[..., :-d, :] * img[..., d:, :]).cpu()
            if threed:
                tpc_3 = torch.mean(img[..., :-d, :, :] * img[..., d:, :, :]).cpu()
                tpc.append((tpc_1 + tpc_2 + tpc_3) / 3)
            else:
                tpc.append((tpc_1 + tpc_2) / 2)
        tpcfin_list.append(np.array(tpc, dtype=np.float64))
    return np.arange(mx, dtype=np.float64), tpcfin_list  

def stat_analysis_error(img, edge_lengths, img_dims, compared_shape, conf=0.95):  # TODO see if to delete this or not
    vf = torch.mean(img).cpu().item()
    err_exp = real_image_stats(img, edge_lengths, vf)
    real_ir = fit_ir(err_exp, img_dims, vf)
    # TODO different size image 1000 vs 1500
    return bernouli(vf, ns_from_dims(compared_shape, real_ir), conf=conf), real_ir


def real_image_stats(img, ls, vf, repeats=4000, z_score=1.96):  
    dims = len(img.shape) - 1
    print(f'repeats = {repeats}')
    errs = []
    for l in ls:
        vfs = []
        if dims == 1:
            for _ in range(repeats):
                bm, xm = img.shape
                x = torch.randint(0, xm - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l]
                vfs.append(torch.mean(crop).cpu())
        elif dims == 2:
            for _ in range(repeats):
                bm, xm, ym = img.shape
                x = torch.randint(0, xm - l, (1,))
                y = torch.randint(0, ym - l, (1,))
                b = torch.randint(0, bm, (1,))
                crop = img[b, x : x + l, y : y + l]
                vfs.append(torch.mean(crop).cpu())
        else:  # 3D
            cur_repeats = repeats - l*10
            print(cur_repeats)
            for _ in range(cur_repeats):
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
    return errs


def bernouli(vf, ns, conf=0.95):
    errs = []
    for n in ns:
        std_theo = ((1 / n) * (vf * (1 - vf))) ** 0.5
        errs.append(100 * (stats.norm.interval(conf, scale=std_theo)[1] / vf))
    return np.array(errs, dtype=np.float64)


def fit_ir(err_exp, img_dims, vf, max_ir=150):
    err_exp = np.array(err_exp)
    ir = test_ir_set(err_exp, vf, np.arange(1, max_ir, 1), img_dims)
    ir = test_ir_set(err_exp, vf, np.linspace(ir - 1, ir + 1, 50), img_dims)
    print(f'real ir = {ir}')
    return ir


def ns_from_dims(img_dims, ir):
    n_dims = len(img_dims[0])
    den = ir ** n_dims
    return [np.prod(i) / den for i in img_dims]
    # if n_dims == 3:  # 2ir length
    #     return [np.prod(i + 2*(ir - 1)) / den for i in img_dims]
    # else:  # n_dims == 2
    #     return [np.prod(i + 2*(ir - 1)) * (1 + 2*(ir - 1)) / (den * ir) for i in img_dims]

def dims_from_n(n, shape, ir, dims):
    den = ir ** dims
    if shape=='equal':
        return (n*den)**(1/dims)-ir+1
    else:
        if dims==len(shape):
            raise ValueError('cannot define all the dimensions')
        if len(shape)==1:
            return ((n*den)/(shape[0]+ir-1))**(1/(dims-1))-ir+1
        else:
            return ((n*den)/((shape[0]+ir-1) * (shape[1]+ir-1)))-ir+1


def test_ir_set(err_exp, vf, irs, img_dims):
    err_fit = []
    for ir in irs:
        ns = ns_from_dims(img_dims, ir)
        err_model = bernouli(vf, ns)
        err = np.mean(abs(err_exp - err_model))
        err_fit.append(err)
    ir = irs[np.argmin(err_fit)].item()
    return ir


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


def tpc_to_ir(tpc_dist, tpc_list, threed=False):
    pred_irs = []
    for tpc in tpc_list:
        tpc, tpc_dist = np.array(tpc), np.array(tpc_dist)
        vf = tpc[0]
        print(f'vf squared = {vf**2}')
        print(f'end of tpc = {np.mean(tpc[-10:])}')
        vf_squared = (vf**2 + np.mean(tpc[-10:]))/2  # mean of vf**2 and the end of the tpc because the volume fraction is not exact.
        omega_n = 1
        pred_irs.append(omega_n/(vf-vf_squared)*np.trapz(tpc - vf_squared, x=tpc_dist))
    print(f'pred irs = {pred_irs}')
    print(f'sum of pred irs = {np.sum(pred_irs)}')
    return np.sum(pred_irs)  


def old_tpc_to_ir(x, y):
    bounds = ((-np.inf, 0.01, -np.inf), (np.inf, np.inf, np.inf))
    coefs_poly3d, _ = curve_fit(tpc_fit, x, y, bounds=bounds)
    y_data = tpc_fit(x, *coefs_poly3d)
    kneedle = kneed.KneeLocator(
        x, y_data, S=1.0, curve="convex", direction="decreasing"
    )
    return kneedle.knee


def make_error_prediction(images, conf=0.95, err_targ=0.05,  model_error=True, correction=True, mxtpc=100, shape='equal', met='vf'):
    vf = np.mean([torch.mean(i).cpu().item() for i in images])
    dims = len(images[0].shape)
    print(f'starting tpc radial')
    tpc_dist, tpc_list = tpc_radial(images, threed=dims == 3, mx=mxtpc)
    print(f'starting tpc to ir')
    ir = tpc_to_ir(tpc_dist, tpc_list, threed=dims==3)
    print(f'pred ir = {ir}')
    n = ns_from_dims([np.array(images[0].shape)], ir)
    # print(n, ir)
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
        # print(n, n_for_err_targ, ir)
    else:
        z = stats.norm.interval(conf)[1]
        err_for_img = (z*std_bern/vf)*slope+intercept
        # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)
        n_for_err_targ = vf * (1 - vf) * (z/ ((err_targ -intercept)/slope * vf)) ** 2

        # print(n_for_err_targ, n, ir)
    l_for_err_targ = dims_from_n(n_for_err_targ, shape, ir, dims)
    return err_for_img, l_for_err_targ, tpc_dist, tpc_list, ir


def make_error_prediction_old(img, conf=0.95, err_targ=0.05,  model_error=True, correction=True, mxtpc=100, shape='equal', met='vf'):
    vf = img.mean()
    dims = len(img.shape)
    tpc = tpc_radial(img, threed=dims == 3, mx=mxtpc)
    cut = max(20, np.argmin(tpc))
    tpc = tpc[:cut]
    x = np.arange(len(tpc))
    ir = tpc_to_ir(x, tpc)
    n = ns_from_dims([np.array(img.shape)], ir)
    # print(n, ir)
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
        # print(n, n_for_err_targ, ir)
    else:
        z = stats.norm.interval(conf)[1]
        err_for_img = (z*std_bern/vf)*slope+intercept
        # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)
        n_for_err_targ = vf * (1 - vf) * (z/ ((err_targ -intercept)/slope * vf)) ** 2

        # print(n_for_err_targ, n, ir)
    l_for_err_targ = dims_from_n(n_for_err_targ, shape, ir, dims)
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
    params= {'2dvf':[0.34014036731289116, 1.61, 0],
             '2dsa':[0.2795574424176512, 1.61, 0],
             '3dvf':[0.5584825884176943, 6, 0.4217],  # TODO needs to be changed according to fig3.py
             '3dsa':[0.4256621550262103, 17.7, 0]}  # TODO needs to be changed according to fig3.py
    return params[imtype]


def get_model_params_old(imtype):  # see model_param.py for the appropriate code that was used.
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

def p2_crosscorrelation(arr1, arr2):
    """
    defines the crosscorrelation between arr1 and arr2:
    :param arr1:
    :param arr2:
    :return:
    """
    ax = list(range(0, len(arr1.shape)))
    arr1_FFT = np.fft.rfftn(arr1, axes=ax)
    arr2_FFT = np.fft.rfftn(arr2, axes=ax)
    return np.fft.irfftn(arr1_FFT.conjugate() * arr2_FFT, s=arr1.shape, axes=ax).real / np.product(
        arr1.shape)