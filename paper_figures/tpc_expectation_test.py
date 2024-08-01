from representativity import core
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
import time
from scipy.stats import norm


def tpc_by_radius(tpc):
    tpc = np.array(tpc)
    middle_idx = np.array(tpc.shape) // 2
    pf = tpc[tuple(map(slice, middle_idx, middle_idx + 1))].item()
    # print(f'pf squared = {np.round(pf**2, 5)}')
    dist_arr = np.indices(tpc.shape)
    dist_arr = np.abs((dist_arr.T - middle_idx.T).T)
    img_volume = np.prod(middle_idx + 1)
    vec_arr = np.prod(middle_idx[0] + 1 - dist_arr, axis=0) / img_volume
    dist_arr = np.sqrt(np.sum(dist_arr**2, axis=0))
    end_dist = int(
        np.max(dist_arr)
    )  # TODO this needs to be changed to the maximum distanse from the center 200*sqrt(2)
    sum_circle = np.sum(vec_arr[dist_arr <= end_dist])
    n_bins = 81
    jump = sum_circle / n_bins
    dist_indices = [0]
    tpc_res = [pf]  # tpc_res is the tpc result of the tpc by growing radiuses
    tpc_vec = vec_arr * tpc
    for i in range(0, end_dist + 1, 1):
        # for i in range(len(dist_indices)-1):
        # dist_bool = (dist_arr>dist_indices[i]) & (dist_arr<=dist_indices[i+1])
        dist_bool = (dist_arr >= dist_indices[-1]) & (dist_arr < i)
        if np.sum(vec_arr[dist_bool]) > jump or i == end_dist:
            dist_indices.append(i)
            tpc_res.append(np.sum(tpc_vec[dist_bool]) / np.sum(vec_arr[dist_bool]))
    return pf, pf**2, tpc_res, dist_indices


def tpcs_radius(gen_func, test_runs, args):
    tpc_results = []
    tpcs = []
    pf_squares = []
    pfs = []
    for _ in [args[0]] * test_runs:
        img_tpc = gen_func(*args)
        tpcs.append(img_tpc)
        pf, pf_square, tpc_res, distances = tpc_by_radius(img_tpc)
        pfs.append(pf)
        tpc_results.append(np.array(tpc_res))
        pf_squares.append(pf_square)
        if (len(pf_squares) % 10) == 0:
            print(f"{len(pf_squares)} / {test_runs} done.")
    return tpc_results, pfs, pf_squares, distances


def make_tpc(img):
    dims = len(img.shape)
    tpc = core.radial_tpc(img, volumetric=dims == 3, periodic=False)
    return tpc


def make_circles_tpc(imsize, circle_radius, pf):
    img = make_circles_2D(imsize, circle_radius, pf)
    return make_tpc(img)


def make_circles_2D(imsize, circle_radius, pf):
    """
    This function is used to create an image with circles of the same size,
    which are randomly placed in the image. The pf is the volume fraction of
    the image, in expectation.
    """
    img = np.zeros([imsize + 2 * circle_radius] * 2)
    circle_area = np.pi * circle_radius**2
    # the probability of a pixel being in a circle (1 minus not appearing in any
    # circle around it):
    p_of_center_circle = 1 - (1 - pf) ** (1 / circle_area)
    circle_centers = np.random.rand(*img.shape) < p_of_center_circle
    circle_centers = np.array(np.where(circle_centers))
    time_before_circles = time.time()
    fill_img_with_circles(img, circle_radius, circle_centers)
    return img[circle_radius:-circle_radius, circle_radius:-circle_radius]


def fill_img_with_circles(img, circle_radius, circle_centers):
    """Fills the image with circles of the same size given by the cicle_radius,
    with the centers given in circle_centers."""
    dist_arr = np.indices(img.shape)
    dist_arr_T = dist_arr.T
    dist_arr_reshape = dist_arr_T.reshape(
        np.prod(dist_arr_T.shape[:2]), dist_arr_T.shape[-1]
    )
    distances = cdist(dist_arr_reshape, circle_centers.T)
    if distances.size == 0:
        return img
    min_distances = np.min(distances, axis=1).reshape(img.shape)
    img[min_distances <= circle_radius] = 1
    return img


if __name__ == "__main__":
    # tpc_check()
    pfs = []
    imsize = 100
    circle_size = 20
    pf = 0.5
    args = (imsize, circle_size, pf)
    run_tests = 10000

    fig, axs = plt.subplot_mosaic(
        [
            ["circle0", "circle1", "TPC figure", "TPC figure"],
            ["circle2", "pf_hist", "TPC figure", "TPC figure"],
        ]
    )
    fig.set_size_inches(16, 16 * 7 / 16)
    # axs['circle1'].set_aspect('equal')
    # tpc_fig.set_aspect('equal')

    tpc_fig = axs["TPC figure"]
    plt.figtext(
        0.31,
        0.9,
        # f"4 random {imsize}$^2$ images with $E[\Phi]$ = {pf} and circle diameter = {circle_size*2}",
        f"(a)",
        ha="center",
        va="center",
        fontsize=12,
    )

    # plt.suptitle(
    #     f"Visual presentation of the proof of Theorem 2 in the simple case of random circles."
    # )
    n_circle_im = 3
    circle_ims = [make_circles_2D(imsize, circle_size, pf) for _ in range(n_circle_im)]
    one_im_tpc = make_tpc(circle_ims[0])
    one_im_tpc_by_radius = tpc_by_radius(one_im_tpc)[2]
    circles_pfs = []
    for i in range(n_circle_im):
        axs[f"circle{i}"].imshow(circle_ims[i], cmap="gray", interpolation='nearest')
        circle_pf = np.round(circle_ims[i].mean(), 3)
        circles_pfs.append(circle_pf)
        axs[f"circle{i}"].set_xlabel(f"$\Phi(\omega_{i})={circle_pf}$")
        axs[f"circle{i}"].set_ylabel(f"$\omega_{i}$")
        axs[f"circle{i}"].set_xticks([])
        axs[f"circle{i}"].set_yticks([])

    # im0_tpc = make_tpc(circle_ims[0])
    # im0_tpc_by_radius = tpc_by_radius(im0_tpc)[2]
    # tpc_fig.plot(im0_tpc_by_radius, label='TPC of $\omega_0$')

    tpc_results, pfs, pf_squares, dist_len = tpcs_radius(
        make_circles_tpc, run_tests, args=args
    )
    bins = 30
    axs["pf_hist"].hist(pfs, bins=bins, density=True)
    axs["pf_hist"].set_xlim(0, 1)
    axs["pf_hist"].set_xlabel("Phase Fraction")
    axs["pf_hist"].set_ylabel("PDF")
    # add a 'best fit' line
    mu, sigma = norm.fit(pfs)
    print(f"mu = {mu}, sigma = {sigma}")
    y = norm.pdf(np.linspace(0, 1, 100), mu, sigma)
    axs["pf_hist"].plot(np.linspace(0, 1, 100), y, linewidth=2, label="Phase fraction\ndistribution")
    for i in range(len(circles_pfs)):
        if i == 0:
            axs["pf_hist"].axvline(circles_pfs[i], label=", ".join([f"$\Phi(\omega_{j})$" for j in range(len(circles_pfs))]), ls='--', color="tab:pink")
        else:
            axs["pf_hist"].axvline(circles_pfs[i], ls='--', color="tab:pink")
    axs["pf_hist"].legend(loc='upper left')
    dist_len = np.array(dist_len)
    mean_tpc = np.mean(tpc_results, axis=0)
    tpc_fig.plot(one_im_tpc_by_radius, ls='--', color='tab:pink', label="TPC of $\omega_0$")
    tpc_fig.plot(mean_tpc, label="Mean TPC $E[T_r]$")
    len_tpc = len(tpc_results[0])
    pf_squared = np.mean(pf_squares)
    label_pf_squared = f"$E[\Phi^2]$ = {np.round(pf_squared, 3)}"
    tpc_fig.plot([pf_squared] * len_tpc, label=label_pf_squared)
    # print(f'pf squared = {np.round(pf_squared, 7)}')

    true_pf_squared = np.mean(pfs) ** 2
    label_true_pf_squared = f"$E[\Phi]^2$ = {np.round(true_pf_squared, 3)}"
    tpc_fig.plot([true_pf_squared] * len_tpc, label=label_true_pf_squared)
    # print(f'true pf squared = {np.round(true_pf_squared, 7)}')

    tpc_fig.axvline(
        x=np.where(dist_len == circle_size * 2)[0][0],
        color="black",
        linestyle="--",
        label="Circle Diameter",
    )

    len_tpc = len(tpc_results[0])
    fill_1 = tpc_fig.fill_between(
        np.arange(len_tpc),
        mean_tpc,
        [pf_squared] * len_tpc,
        alpha=0.2,
        where=mean_tpc >= pf_squared,
        label=f"Area $A_1$",
    )
    fill_2 = tpc_fig.fill_between(
        np.arange(len_tpc),
        [pf_squared] * len_tpc,
        mean_tpc,
        alpha=0.2,
        where=np.logical_and(
            dist_len <= circle_size * 2, np.array(mean_tpc < pf_squared)
        ),
        label=f"Area $A_2$",
    )
    fill_3 = tpc_fig.fill_between(
        np.arange(len_tpc),
        [pf_squared] * len_tpc,
        mean_tpc,
        alpha=0.2,
        where=dist_len >= circle_size * 2,
        label=f"Area B",
    )
    text_jump = 0.017
    tpc_fig.text(3, pf-(pf-pf**2) * (5/7) , "$A_1$", fontsize=12)
    tpc_fig.text(16.8, (pf_squared + true_pf_squared) / 2, "$A_2$", fontsize=12)
    tpc_fig.text(40, (pf_squared + true_pf_squared) / 2 - 0.0005, "B", fontsize=12)
    # tpc_fig.text(10, 0.079, '$\Phi$ calculates phase fraction.', fontsize=12)
    # tpc_fig.text(22, 0.068, 'The variance of $\Phi$ is:', fontsize=12)
    tpc_fig.text(
        22,
        pf_squared + text_jump * 7,
        r"How the model predicts the variance of $\Phi$:",
        fontsize=12,
    )
    tpc_fig.text(
        22, pf_squared + text_jump * 6, "$Var[\Phi]=E[\Phi^2]-E[\Phi]^2$", fontsize=12
    )
    tpc_fig.text(
        22,
        pf_squared + text_jump * 5,
        r"$Var[\Phi]=\frac{1}{C_{40}}\cdot B$ (Normalization of B's width)",
        fontsize=12,
    )
    tpc_fig.text(
        22,
        pf_squared + text_jump * 4,
        r"$\sum_r{(E[T_r]-E[\Phi^2])}=0$, So",
        fontsize=12,
    )
    tpc_fig.text(22, pf_squared + text_jump * 3, r"$A_1-A_2=B$", fontsize=12)
    tpc_fig.text(22, pf_squared + text_jump * 2, "Which results in:", fontsize=12)
    tpc_fig.text(
        22,
        pf_squared + text_jump * 1,
        r"$Var[\Phi]=\frac{1}{C_{40}}\cdot (A_1-A_2)=E[\Psi]$",
        fontsize=12,
    )

    tpc_fig.set_title(
        # f"Mean TPC of {run_tests} of these random {imsize}$^2$ circle images"
        f"(b)"
    )

    tpc_fig.set_ylim(true_pf_squared - 0.005, pf + 0.01)
    dist_ticks = list(dist_len[:-5:5]) + [dist_len[-1]]
    x_ticks_labels = list(np.arange(0, len_tpc - 5, 5)) + [len_tpc - 1]
    tpc_fig.set_xticks(x_ticks_labels, dist_ticks)
    tpc_fig.set_xlabel("TPC distance r")
    tpc_fig.set_ylabel(f"TPC function", labelpad=-20)
    # tpc_fig.set_yscale('log')
    tpc_fig.set_yticks(
        [pf**2, np.round(pf_squared, 3), pf], [pf**2, np.round(pf_squared, 3), pf]
    )
    tpc_fig.legend(loc='upper right')
    plt.savefig(f"tpc_results/circles_tpc_visual_proof.pdf", format="pdf")
