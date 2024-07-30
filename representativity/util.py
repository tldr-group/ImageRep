import numpy as np

from representativity import core

try:
    import torch
    from representativity import slicegan
except ImportError:
    print("Couldn't import torch, slicegan related code will not work!")

from scipy import stats, ndimage
from matplotlib import pyplot as plt


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


def generate_image(netG, slice_dim=0, lf=50, threed=False, reps=50):

    netG.eval()
    imgs = []

    for _ in range(reps):
        if (_ % 50) == 0 and _ != 0:
            print(f"generating image {_}")
        noise = torch.randn(1, 16, lf if threed else 4, lf, lf)
        noise.transpose_(2, slice_dim + 2)
        noise = noise.cuda()
        img = netG(noise, threed, slice_dim)
        img = slicegan.util.post_proc(img)
        img.transpose_(0, slice_dim)
        if not threed:
            imgs.append(img[0])
        else:
            imgs.append(img.cpu())
    img = torch.stack(imgs, 0)
    return img.float()


def angular_img(img):
    base_len, l = img.shape[0:2]
    img = img.cpu().numpy()
    plt.imshow(img[0, :100, :100])
    plt.show()
    img_rot = ndimage.rotate(img, base_len / l * 90, axes=(1, 0), reshape=False)
    for i in range(img_rot.shape[0]):
        print(f"slice {i}")
        plt.imshow(img_rot[i, :100, :100])
        plt.show()
        plt.imshow(img_rot[i, -100:, -100:])
        plt.show()
    return img_rot


def stat_analysis_error(img, pf, edge_lengths):  # TODO see if to delete this or not
    img_dims = [np.array((l,) * (len(img.shape) - 1)) for l in edge_lengths]
    err_exp = real_image_stats(img, edge_lengths, pf)
    real_cls = core.fit_statisical_cls_from_errors(err_exp, img_dims, pf)
    # TODO different size image 1000 vs 1500
    return real_cls


def real_image_stats(img, ls, pf, repeats=4000, conf=0.95):
    """Calculates the error of the stat. analysis for different edge lengths.
    The error is calculated by the std of the mean of the subimages divided by the pf.
    params:
    img: the image to calculate the error for (Should be a stack of images).
    ls: the edge lengths to calculate the error for.
    pf: the phase fraction of the image.
    repeats: the number of repeats for each edge length.
    conf: the confidence level for the error."""
    dims = len(img[0].shape)
    errs = []
    for l in ls:
        pfs = []
        n_pos_ims = int(np.prod(img.shape) / l**dims)
        repeats = n_pos_ims * 2
        # print(f'one im repeats = {repeats} for l = {l}')
        if dims == 1:
            for _ in range(repeats):
                bm, xm = img.shape
                x = np.random.randint(
                    0,
                    xm - l,
                )
                b = np.random.randint(0, bm)
                crop = img[b, x : x + l]
                pfs.append(np.mean(crop))
        elif dims == 2:
            for _ in range(repeats):
                bm, xm, ym = img.shape
                x = np.random.randint(0, xm - l)
                y = np.random.randint(0, ym - l)
                b = np.random.randint(0, bm)
                crop = img[b, x : x + l, y : y + l]
                pfs.append(np.mean(crop))
        else:  # 3D
            for _ in range(repeats):
                bm, xm, ym, zm = img.shape
                x = np.random.randint(0, xm - l)
                y = np.random.randint(0, ym - l)
                z = np.random.randint(0, zm - l)
                b = np.random.randint(0, bm)
                crop = img[b, x : x + l, y : y + l, z : z + l]
                pfs.append(np.mean(crop))
        pfs = np.array(pfs)
        ddof = 1  # for unbiased std
        std = np.std(pfs, ddof=ddof)
        errs.append(100 * (stats.norm.interval(conf, scale=std)[1] / pf))
    return errs


def bernouli_from_cls(cls, pf, img_size, conf=0.95):
    ns = core.n_samples_from_dims([np.array(img_size)], cls)
    return core.bernouli(pf, ns, conf)


# fit_cls now fit_statistical_cls_from_errors

# ns_from_dims now n_samples_from_dims

# test_cls_set now test_all_cls_in_range


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


def optimize_error_conf_pred(bern_conf, total_conf, std_bern, std_model, pf):
    model_conf = total_conf / bern_conf
    err_bern = stats.norm.interval(bern_conf, scale=std_bern)[1]
    one_side_error_model = model_conf * 2 - 1
    err_model = stats.norm.interval(one_side_error_model, scale=std_model)[1]
    return err_bern * (1 + err_model)


def optimize_error_n_pred(bern_conf, total_conf, std_model, pf, err_targ):
    model_conf = total_conf / bern_conf
    z1 = stats.norm.interval(bern_conf)[1]
    one_side_error_model = model_conf * 2 - 1
    err_model = stats.norm.interval(one_side_error_model, scale=std_model)[1]
    num = (err_model + 1) ** 2 * (1 - pf) * z1**2 * pf
    den = (err_targ) ** 2  # TODO go over the calcs and see if this is right
    return num / den


# renamed calc_autocorrelation_orthant -> autocorrelation_orthant

# renamed one_img_stat_analysis_error -> stat_analysis_error_classic
