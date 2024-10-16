from representativity.correction_fitting import prediction_error
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np

sigma_color = "tab:pink"
model_color = "tab:cyan"
oi_color = "tab:orange"
dashes = [2.5, 5]


def scatter_plot(ax, res, xlabel, ylabel):

    pred_data, fit_data, oi_data = res
    max_val = np.max([np.max(fit_data), np.max(pred_data)])
    x = np.linspace(0, max_val, 100)
    ax.plot(x, x, label="Ideal predictions", color="black")

    ax.scatter(
        fit_data, oi_data, alpha=0.7, s=0.3, c=oi_color, label="Classical predictions"
    )
    oi_errs = (fit_data - oi_data) / oi_data
    oi_mean = np.mean(oi_errs)
    ax.plot(
        x,
        x * (1 - oi_mean),
        color=oi_color,
        linestyle="--",
        dashes=[2.5, 5],
        label="Mean predictions",
    )

    ax.scatter(
        fit_data,
        pred_data,
        alpha=0.5,
        s=0.3,
        c=model_color,
        label="Our model's predictions",
    )
    model_errs = (fit_data - pred_data) / pred_data
    model_mean = np.mean(model_errs)
    ax.plot(
        x,
        x * (1 - model_mean),
        color=model_color,
        linestyle="--",
        dashes=[2.5, 5],
        label="Mean predictions",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    model_errs = np.sort(model_errs)
    std = np.std(model_errs)

    ax.set_aspect("equal", adjustable="box")


with_cls = False
fig, axs = plt.subplots(2, 3 + with_cls)
fig.set_size_inches(4 * (3 + with_cls), 8)
dims = ["2D", "3D"]
edge_length = ["1536", "448"]
for i, dim in enumerate(dims):
    pred_cls_all, fit_cls_all, oi_cls_all, _, vfs = (
        prediction_error.pred_vs_fit_all_data(
            dim, edge_length[i], num_runs=9, std_not_cls=False
        )
    )
    cls_results = [pred_cls_all, fit_cls_all, oi_cls_all]
    # calculate the standard deviation instead of the cls:
    vfs = np.array(vfs)
    vfs_one_minus_vfs = vfs * (1 - vfs)
    dim_int = int(dim[0])
    cur_edge_length = int(edge_length[i])
    std_results = [
        ((cls_res / cur_edge_length) ** dim_int * vfs_one_minus_vfs) ** 0.5
        for cls_res in cls_results
    ]
    dim_str = dim[0]
    x_labels = [
        f"True CLS $a_{int(dim[0])}$",
        f"True Phase Fraction std, $\sigma_{int(dim[0])}$",
    ]
    cls_math = r"\tilde{a}_{2}" if dim == "2D" else r"\tilde{a}_{3}"
    std_math = r"\tilde{\sigma}_{2}" if dim == "2D" else r"\tilde{\sigma}_{3}"
    y_labels = [
        "Predicted CLS $%s$" % cls_math,
        "Predicted Phase Fraction std, $%s$" % std_math,
    ]
    # title_suffix = r'Image size $%s^%s$' %(edge_length[i], dim_str)
    # titles = [f'{dim} CLS comparison, '+title_suffix, f'{dim} std comparison, '+title_suffix]

    for j, res in enumerate([cls_results, std_results]):
        ax_idx = j
        if not with_cls:
            if j == 0:
                continue
            else:
                ax_idx = 0

        ax = axs[i, ax_idx]

        ax_xlabel = x_labels[j]
        ax_ylabel = y_labels[j]

        scatter_plot(ax, res, ax_xlabel, ax_ylabel)

        if j == 0 or not with_cls and i == 0:
            ax.legend(loc="upper left")

    # Fit a normal distribution to the data:
    pred_std_all, fit_std_all = std_results[:2]
    errs = (fit_std_all - pred_std_all) / pred_std_all * 100
    mu, std = stats.norm.fit(errs)

    ax_idx = 1 + with_cls
    ax2 = axs[i, ax_idx]

    # Plot the histogram.
    counts, bins = np.histogram(errs)

    max_val = np.max([np.max(errs), -np.min(errs)])
    y, x, _ = ax2.hist(
        errs,
        range=[-max_val, max_val],
        bins=50,
        alpha=0.6,
        color=model_color,
        density=True,
        label="Our model's predictions",
    )
    mean_percentage_error = np.mean(errs)

    # Plot the PDF.
    xmin, xmax = x.min(), x.max()
    max_abs = max(np.abs(np.array([xmin, xmax])))
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)

    ax2.set_xlabel(
        r"Prediction Percentage Error, $\frac{\sigma_{%s}-\tilde{\sigma_{%s}}}{\tilde{\sigma_{%s}}}$"
        % (dim_str, dim_str, dim_str)
    )
    x_ticks = ax2.get_xticks()[1:-1]
    ax2.set_xticks(x_ticks, [f"{int(i)}%" for i in x_ticks])
    ax2.set_yticks([ax2.get_yticks()[0],ax2.get_yticks()[-2]],[round(ax2.get_yticks()[0],2),round(ax2.get_yticks()[-2],2)])
    # if i == 0:
    #     ax2.set_ylim(0, ax2.get_yticks()[-1]*1.6)
    ax2.set_ylabel("Probability density")
    ax2.vlines(0, ymin=0, ymax=y.max(), color="black", label="Ideal predictions")
    ax2.vlines(
        mean_percentage_error,
        ymin=0,
        ymax=y.max(),
        linestyles=[(0, (2.5, 5))],
        color=model_color,
        label=r"Mean predictions",
    )
    ax2.plot(x, p, linewidth=2, label=f"Fitted normal distribution")
    ax2.vlines(
        std,
        ymin=0,
        ymax=y.max(),
        ls="--",
        color=sigma_color,
        label=r"Standard deviation $\sigma_{mod}$",
    )

    if i == 0:
        ax2.legend(loc="upper left")

    # Plot the std error by size of edge length:
    ax3 = axs[i, ax_idx + 1]
    run_data, _ = prediction_error.data_micros(dim)
    edge_lengths = run_data["edge_lengths_pred"]
    start_idx = 2
    pred_error, stds = prediction_error.std_error_by_size(
        dim, edge_lengths, num_runs=10, start_idx=start_idx, std_not_cls=True
    )
    stds = stds * 100  # convert to percentage error
    edge_lengths = edge_lengths[start_idx:]
    ax3.scatter(edge_lengths, stds, color="black")
    edge_length_pos = edge_lengths.index(cur_edge_length)
    ax3.scatter(
        edge_lengths[edge_length_pos],
        stds[edge_length_pos],
        color=sigma_color,
        label=r"$\sigma_{\it{mod}}$ for image size $%s^%s$" % (edge_length[i], dim_str),
    )
    ax3.plot(edge_lengths, pred_error * 100, label="Prediction error fit")
    ax3.set_xlabel("Image size")
    ax3.set_xticks(
        edge_lengths[::2], [r"$%s^%s$" % (i, dim_str) for i in edge_lengths[::2]]
    )
    
    ax3.set_ylabel(r"Model percentage error std, $\sigma_{\it{mod}}$ / %")
    if i == 0:
        ax3.legend(loc="upper right")


if not with_cls:
    titles_1st_row = ["(a)", "(b)", "(c)"]
    [ax.set_title(titles_1st_row[i]) for i, ax in enumerate(axs[0])]
    titles_2nd_row = ["(d)", "(e)", "(f)"]
    [ax.set_title(titles_2nd_row[i]) for i, ax in enumerate(axs[1])]

plt.tight_layout()
fig.savefig("paper_figures/output/fig_model_error.pdf", format="pdf", dpi=300)
