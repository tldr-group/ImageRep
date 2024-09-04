import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time
from representativity import core
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from os import getcwd, path

CWD = getcwd()
DATA_PATH = f"{CWD}/paper_figures/"


def plot_likelihood_of_phi(image_pf, pred_std, std_dist_std):
    fig, ax = plt.subplots(2, 3)
    fig.set_size_inches(15, 9)

    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    im = np.load(f"{DATA_PATH}im_for_prediction_interval_fig.npy")
    ax[0, 0].imshow(im[:400, :400], cmap="gray", interpolation="nearest")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title("(a)")
    # ax[0,0].set_ylabel('Sample image')
    # ax[0,0].axis('off')
    std_dist_divisions = 1001
    num_stds = min(pred_std / std_dist_std - pred_std / std_dist_std / 10, 6)
    # first make the "weights distribution", the normal distribution of the stds
    # where the prediction std is the mean of this distribution.
    x_std_dist_bounds = [
        pred_std - num_stds * std_dist_std,
        pred_std + num_stds * std_dist_std,
    ]
    x_std_dist = np.linspace(*x_std_dist_bounds, std_dist_divisions)  # type: ignore
    std_dist = core.normal_dist(x_std_dist, mean=pred_std, std=std_dist_std)
    ax[0, 1].plot(x_std_dist, std_dist, c="green", label="Likelihood of true std")
    ax[0, 1].vlines(
        pred_std,
        0,
        np.max(std_dist),
        linestyle="--",
        color="orange",
        label="Model's predicted std",
    )
    ax[0, 1].set_title("(b)")
    ax[0, 1].set_xlabel("Standard deviation")
    ax[0, 1].set_yticks([0, 80])
    ax[0, 1].set_ylabel("Probability density")
    ax[0, 1].legend()
    integral_std_dist = np.trapz(std_dist, x_std_dist)
    print(integral_std_dist)
    # Next, make the pf distributions, each row correspond to a different std, with
    # the same mean (observed pf) but different stds (x_std_dist), multiplied by the
    # weights distribution (std_dist).
    pf_locs = np.ones((std_dist_divisions, std_dist_divisions)) * image_pf
    pf_x_bounds = [image_pf - num_stds * pred_std, image_pf + num_stds * pred_std]
    pf_x_1d = np.linspace(*pf_x_bounds, std_dist_divisions)  # type: ignore
    pf_mesh, std_mesh = np.meshgrid(pf_x_1d, x_std_dist)
    # before normalising by weight:
    pf_dist_before_norm = core.normal_dist(pf_mesh, mean=pf_locs, std=std_mesh)
    mid_std_dist = pf_dist_before_norm[std_dist_divisions // 2, :]
    ax[0, 2].plot(
        pf_x_1d,
        mid_std_dist,
        c="orange",
        label="Likelihood of the material\nphase fraction, trusting\nstd prediction as true std",
    )
    ax[0, 2].vlines(
        image_pf,
        0,
        np.max(mid_std_dist),
        linestyle="--",
        color="g",
        label="Observed phase fraction $\Phi(\omega)$",
    )
    ax[0, 2].set_title("(c)")
    ax[0, 2].set_xlabel("Phase fraction")
    ax[0, 2].set_yticks([0, 20])
    ax[0, 2].set_ylabel("Probability density")
    ax[0, 2].legend()

    c1 = ax[1, 0].contourf(
        pf_mesh, std_mesh, pf_dist_before_norm, levels=100, cmap="plasma"
    )
    for c in c1.collections:
        c.set_edgecolor("face")
    # ax[1,0].vlines(image_pf, x_std_dist_bounds[0], x_std_dist_bounds[1], colors='g', label = 'Observed $\Phi(\omega)$')
    ax[1, 0].hlines(
        pred_std,
        pf_x_bounds[0],
        pf_x_bounds[1],
        colors="orange",
        label="(c) - Likelihood of the material\nphase fraction using predicted\nstd",
    )
    ax[1, 0].set_title("(d)")
    ax[1, 0].set_xlabel("Phase fraction")
    ax[1, 0].set_ylabel("Standard deviation")
    ax[1, 0].set_yticks(ax[1, 0].get_yticks()[1:-1:2])
    ax[1, 0].legend()
    # normalise by weight:
    pf_dist = (pf_dist_before_norm.T * std_dist).T
    c2 = ax[1, 1].contourf(pf_mesh, std_mesh, pf_dist, levels=100, cmap="plasma")
    for c in c2.collections:
        c.set_edgecolor("face")
    ax[1, 1].set_title("(e)")
    ax[1, 1].set_xlabel("Phase fraction")
    ax[1, 1].set_ylabel("Standard deviation")
    ax[1, 1].vlines(
        image_pf,
        x_std_dist_bounds[0],
        x_std_dist_bounds[1],
        colors="g",
        label="Multiplying each row by std\nlikelihood weights of (b)",
    )
    ax[1, 1].hlines(
        pred_std,
        pf_x_bounds[0],
        pf_x_bounds[1],
        colors="orange",
        label="(c) after multiplied by the\ncenter weight of (b)",
    )
    ax[1, 1].set_yticks(ax[1, 1].get_yticks()[1:-1:2])
    ax[1, 1].legend()
    pf_dist_integral_col = np.trapz(pf_dist, x_std_dist, axis=0)
    pf_dist_integral = np.trapz(pf_dist_integral_col, pf_x_1d)
    
    axins = inset_axes(ax[1,1],
                    width="5%",  
                    height="100%",
                    loc='right',
                    borderpad=-2
                   )
    cbar = fig.colorbar(c2, cax=axins, orientation="vertical")
    cbar_ticks = [0, 2000]
    cbar.set_label(f'Probability density', labelpad=-17)
    cbar.ax.set_yticks(cbar_ticks)
    # fig.colorbar(contour, ax=[ax[1,0],ax[1,1]],orientation='vertical')
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(pf_mesh, std_mesh, pf_dist, cmap='viridis',\
    #                 edgecolor='green')
    # plt.show()
    # print(pf_dist_integral)

    sum_dist_norm = np.sum(pf_dist, axis=0) * np.diff(x_std_dist)[0]
    # mu, std = norm.fit(sum_dist_norm)
    # p = norm.pdf(pf_x_1d, mu, std)
    # print(mu, std)
    # both need a bit of normalization for symmetric bounds (they're both very close to 1)
    sum_dist_norm /= np.trapz(sum_dist_norm, pf_x_1d)
    mid_std_dist /= np.trapz(mid_std_dist, pf_x_1d)

    cum_sum_sum_dist_norm = np.cumsum(sum_dist_norm * np.diff(pf_x_1d)[0])
    # cum_sum_sum_dist_norm /= np.trapz(sum_dist_norm, pf_x_1d)
    alpha = 0.975
    alpha_sum_dist_norm_end = np.where(cum_sum_sum_dist_norm > alpha)[0][0]
    alpha_sum_dist_norm_beginning = np.where(cum_sum_sum_dist_norm > 1 - alpha)[0][0]
    cum_sum_mid_std_dist = np.cumsum(mid_std_dist * np.diff(pf_x_1d)[0])
    # cum_sum_mid_std_dist /= np.trapz(mid_std_dist, pf_x_1d)
    alpha_mid_std_dist_end = np.where(cum_sum_mid_std_dist > alpha)[0][0]
    alpha_mid_std_dist_beginning = np.where(cum_sum_mid_std_dist > 1 - alpha)[0][0]

    ax[1, 2].plot(
        pf_x_1d,
        sum_dist_norm,
        c="blue",
        label="Likelihood of the material\nphase fraction by integrating\nthe rows of (e)",
    )
    # new_std_dist = normal_dist(pf_x_1d, mean=image_pf, std=0.02775)
    # ax[1,2].plot(pf_x_1d, new_std_dist, c='green', label = 'new std dist')
    ax[1, 2].vlines(
        pf_x_1d[alpha_sum_dist_norm_end],
        0,
        np.max(sum_dist_norm),
        linestyle="--",
        color="blue",
        label=r"95% confidence bounds",
    )
    ax[1, 2].vlines(
        pf_x_1d[alpha_sum_dist_norm_beginning],
        0,
        np.max(sum_dist_norm),
        linestyle="--",
        color="blue",
    )
    ax[1, 2].vlines(
        image_pf,
        0,
        np.max(sum_dist_norm),
        linestyle="--",
        color="g",
        label="Observed phase frcation $\Phi(\omega)$",
    )
    ax[1, 2].plot(pf_x_1d, mid_std_dist, c="orange", label="(c)")
    ax[1, 2].vlines(
        pf_x_1d[alpha_mid_std_dist_beginning],
        0,
        np.max(sum_dist_norm),
        linestyle="--",
        color="orange",
    )
    ax[1, 2].vlines(
        pf_x_1d[alpha_mid_std_dist_end],
        0,
        np.max(sum_dist_norm),
        linestyle="--",
        color="orange",
        label=r"(c) 95% confidence interval",
    )
    ax[1, 2].set_xlabel("Phase fraction")
    ax[1, 2].set_title("(f)")
    ax[1, 2].set_yticks([0, 20])
    ax[1, 2].set_ylabel("Probability density")
    ax[1, 2].legend()
    # ax[1,2].yaxis.set_label_position("right")
    # ax[1,2].yaxis.tick_right()
    print(np.trapz(sum_dist_norm, pf_x_1d))
    print(np.trapz(mid_std_dist, pf_x_1d))
    
    plt.savefig("prediction_interval.pdf", format="pdf")


if __name__ == "__main__":
    
    im = np.load(f"{DATA_PATH}im_for_prediction_interval_fig.npy")
    image_pf = im.mean()
    out = core.make_error_prediction(
                            im,
                            confidence=0.95,
                            model_error=False,
                        )
    im_len = im.shape[0]
    cls, std_model = out["integral_range"], out["std_model"]
    pred_std = (cls**2/im_len**2*image_pf*(1-image_pf))**0.5
    std_dist_std = pred_std * std_model
    time_bf = time.time()
    plot_likelihood_of_phi(image_pf, pred_std, std_dist_std)
    print(f"time taken = {time.time()-time_bf}")
    print(core.get_prediction_interval(image_pf, pred_std, std_dist_std)[0])

    print(f'image phase fraction: {image_pf}')
    print(f'predicted std: {pred_std}')
    print(f'std model: {std_model}')
    print(f'std dist std: {std_dist_std}')
    