import numpy as np
import torch
import slicegan
from scipy.optimize import minimize
from scipy import stats
from matplotlib import pyplot as plt
from itertools import product
import time
from scipy import ndimage

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

def generate_image(netG, Project_path, slice_dim, lf=50, threed=False, reps=50):
    try:
        netG.load_state_dict(torch.load(Project_path + "_Gen.pt"))
    except:  # if the image is greayscale it's excepting because there's only 1 channel
        return torch.tensor(0)
    netG.eval()
    imgs = []
    # z_channels = 16
    # plot_profiles = []
    # img_size = [450, 450, 450] if threed else [64, 1500, 1500]
    for i in range(reps):
        noise = torch.randn(1, 16, lf if threed else 4, lf, lf)
        noise.transpose_(2, slice_dim+2)
        noise = noise.cuda()
        img = netG(noise, threed, slice_dim)
        img = slicegan.util.post_proc(img)
        img.transpose_(0, slice_dim)
        if not threed:
            # img = angular_img(img)
            imgs.append(img[0])
        else:
            imgs.append(img.cpu())
        
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

def angular_img(img):
    base_len, l = img.shape[0:2]
    img = img.cpu().numpy()
    plt.imshow(img[0, :100, :100])
    plt.show()
    img_rot = ndimage.rotate(img, base_len/l*90, axes=(1, 0), reshape=False)
    for i in range(img_rot.shape[0]):
        print(f'slice {i}')
        plt.imshow(img_rot[i, :100, :100])
        plt.show()
        plt.imshow(img_rot[i, -100:, -100:])
        plt.show()
    return img_rot


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

# def tpc_radial_fft(img_list, mx=100, threed=False):
#     """Calculates the radial tpc using fft"""
#     tpcfin_list = []
#     for i in range(len(img_list)):  
#         img = img_list[i]
#         tpc_radial = p2_crosscorrelation(img, img)
#         if threed:
#             tpc_radial = tpc_radial[:mx, :mx, :mx]
#         else:   
#             tpc_radial = tpc_radial[:mx, :mx]
#         tpcfin_list.append(np.array(tpc_radial, dtype=np.float64))
#     return tpcfin_list


def tpc_radial(img_list, mx=100, w_fft=True, threed=False):
    tpcfin_list = []
    for i in range(len(img_list)):
        img = img_list[i]
        if w_fft:
            tpc_radial = two_point_correlation(img, desired_length=mx, periodic=True, threed=threed)
            tpcfin_list.append(tpc_radial)
            break
        else:
            img = torch.tensor(img, device=torch.device("cuda:0")).float()
        tpc = {i:[0,0] for i in range(mx+1)}
        for x in range(0, mx):
            for y in range(0, mx):
                for z in range(0, mx if threed else 1):
                    d = (x**2 + y**2 + z**2) ** 0.5
                    if d < mx:
                        remainder = d%1
                        if w_fft:
                            cur_tpc = tpc_radial[x,y,z] if threed else tpc_radial[x,y]
                        else:
                            con_img = conjunction_img_for_tpc(img, x, y, z, threed)
                            cur_tpc = torch.mean(con_img).cpu()
                        weight_floor = 1-remainder
                        weight_ceil = remainder
                        tpc[int(d)][0] += weight_floor 
                        tpc[int(d)][1] += cur_tpc*weight_floor
                        tpc[int(d)+1][0] += weight_ceil 
                        tpc[int(d)+1][1] += cur_tpc*weight_ceil
    
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

# def tpc_to_ir_from_fft(tpc_list, threed=False):
#     pred_irs = []
#     for tpc in tpc_list:
#         if threed:
#             vf = tpc[0,0,0]
#             vf_squared = np.mean([np.mean(tpc[-10:, ...]),np.mean(tpc[:, -10:, :]), np.mean(tpc[..., -10:])])
#             omega_n = 1
#         else:
#             vf = tpc[0,0]
#             vf_squared = np.mean([np.mean(tpc[-10:, :]),np.mean(tpc[:, -10:])])
#             omega_n = 1
#         pred_irs.append(omega_n/(vf-vf_squared)*np.sum(tpc - vf_squared))
#     print(f'pred irs = {pred_irs}')
#     print(f'sum of pred irs = {np.sum(pred_irs)}')
#     return np.sum(pred_irs)  

def tpc_to_ir(tpc_dist, tpc_list, threed=False):
    pred_irs = []
    for tpc in tpc_list:
        tpc, tpc_dist = np.array(tpc), np.array(tpc_dist)
        vf = tpc[0,0,0] if threed else tpc[0,0]
        print(f'vf squared = {vf**2}')
        dist_arr = np.indices(tpc.shape)
        dist_arr = np.sqrt(np.sum(dist_arr**2, axis=0))
        vf_squared = np.mean(tpc[(dist_arr>=90) & (dist_arr<=100)])
        print(f'end of tpc = {vf_squared}')
        omega_n = 4
        vf_squared = (vf_squared + vf**2)/2
        if threed:
            omega_n = 8
        pred_ir = omega_n/(vf-vf_squared)*np.sum((tpc[dist_arr<=100] - vf_squared))
        if pred_ir < 1:
            print(f'pred ir = {pred_ir} CHANGING TPC TO POSITIVE VALUES')
            negatives = np.where(tpc - vf_squared < 0)
            tpc[negatives] += (vf_squared - tpc[negatives])/2
            pred_ir = omega_n/(vf-vf_squared)*np.sum((tpc[dist_arr<=100] - vf_squared))
        pred_ir = pred_ir**(1/3) if threed else pred_ir**(1/2)
        pred_irs.append(pred_ir)
        # else:  
            # pred_irs.append(omega_n/(vf-vf_squared)*np.trapz(tpc - vf_squared, x=tpc_dist))
    print(f'pred irs = {pred_irs}')
    print(f'sum of pred irs = {np.sum(pred_irs)}')
    return np.sum(pred_irs)  


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


def calc_autocorrelation_orthant(img, numel_large, dims, desired_length=100):
    """Calculates the autocorrelation function of an image using the FFT method
    Calculates over a single orthant, in all directions right and down from the origin."""
    ax = list(range(0, len(img.shape)))
    img_FFT = np.fft.rfftn(img, axes=ax)
    tpc = np.fft.irfftn(img_FFT.conjugate() * img_FFT, s=img.shape, axes=ax).real / numel_large
    return tpc[tuple(map(slice, (desired_length,)*dims))]


def two_point_correlation_orthant(img, dims, desired_length=100, periodic=True):
    """
    Calculates the two point correlation function of an image along an orthant.
    If the image is not periodic, it pads the image with desired_length number of zeros, before
    before calculating the 2PC function using the FFT method. After the FFT calculation, it 
    normalises the result by the number of possible occurences of the 2PC function.
    """
    if not periodic:  # padding the image with zeros, then calculates the normaliser.
        indices_img = np.indices(img.shape) + 1
        normaliser = np.flip(np.prod(indices_img, axis=0))
        img = np.pad(img, [(0, desired_length) for _ in range(dims)], 'constant')
    numel_large = np.product(img.shape)
    tpc_desired = calc_autocorrelation_orthant(img, numel_large, dims, desired_length)
    if not periodic:  # normalising the result
        normaliser = normaliser[tuple(map(slice, tpc_desired.shape))]
        normaliser = numel_large/normaliser
        return normaliser*tpc_desired
    else:
        return tpc_desired  
    

def two_point_correlation(img, desired_length=100, periodic=True, threed=False):
    """ 
    Calculates the two point correlation function of an image using the FFT method. 
    The crosscorrelation function is calculated in all directions, where the middle of the output
    is the origin f(0,0,0) (or f(0,0) in 2D). 
    :param img: the image to calculate the 2PC function of.
    :param desired_length: the length of the 2PC function to calculate. Preferably even.
    :param periodic: whether the image is periodic or not, and whether the FFT calculation is done
    with the periodic assumption.
    :param threed: whether the image is 3D or not.
    """
    img = np.array(img)
    dims = 3 if threed else 2
    orthants = {}
    for axis in product((1, 0), repeat=dims-1):
        flip_list = np.arange(dims-1)[~np.array(axis, dtype=bool)]
        flipped_img = np.flip(img, flip_list)
        tpc_orthant = two_point_correlation_orthant(flipped_img, dims, desired_length+1, periodic)
        backflip_orthant = np.flip(tpc_orthant, flip_list)
        orthants[axis + (1,)] = backflip_orthant
        opposite_axis = tuple(1 - np.array(axis)) + (0,)
        orthants[opposite_axis] = np.flip(backflip_orthant)  # flipping the orthant to the opposite side
    res = np.zeros((desired_length*2+1,)*dims)
    for axis in orthants.keys():
        axis_idx = np.array(axis)*desired_length
        slice_to_input = tuple(map(slice, axis_idx, axis_idx+desired_length+1))
        res[slice_to_input] = orthants[axis]
    return res