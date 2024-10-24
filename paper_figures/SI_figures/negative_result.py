import numpy as np
import matplotlib.pyplot as plt
import tifffile
from paper_figures.model_accuracy import get_prediction_interval_stats

if __name__ == '__main__':

    # Create a figure with 2 subplots, one to show microstructure 368 (left),
    # And one to show the lower quantile and upper quantile negative result
    # of the un-representativeness of the image.
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Load the png microstructure image:
    micro_368 = tifffile.imread("paper_figures/figure_data/microstructure368_inpainted.tif")
    # Convert the microstructure image to binary:
    micro_368[micro_368 == 255] = 1
    # Switch between 0 and 1:
    micro_368 = 1 - micro_368

    # Plot the microstructure image:

    micro_ax = axs[0]
    micro_ax.imshow(micro_368, cmap='gray', interpolation='nearest')
    

    # Create the prediction interval plot:
    pred_interval_ax = axs[1]
    conf_bounds, pf_1d, cum_sum_sum = get_prediction_interval_stats(micro_368)
    # cumulative sum to the original data:
    original_data = np.diff(cum_sum_sum)
    pred_interval_ax.plot(pf_1d[:-1], original_data, label="ImageRep likelihood of $\phi$\ngiven only inset image")

    pred_interval_ax.set_xlabel('Phase fraction')
    # No y-ticks:
    # pred_interval_ax.set_yticks([])
    # Find the 25th and 75th percentile of the cumulative sum:
    # First, find the 25th percentile:
    conf_start = np.percentile(cum_sum_sum, 25)
    # Find the 75th percentile:
    conf_end = np.percentile(cum_sum_sum, 75)

    # Fill between confidence bounds:
    conf_start, conf_end = conf_bounds
    # Fill between the confidence bounds under the curve:
    pred_interval_ax.fill_between(
        pf_1d[:-1], 
        original_data, 
        where=(pf_1d[:-1] >= 0) & (pf_1d[:-1] <= conf_start), 
        alpha=0.3
        )
    # Plot in dashed vertical lines the materials phase fraction and the inset phase fraction:
    phi = micro_368.mean()
    pred_interval_ax.vlines(
        phi,
        0,
        np.max(original_data),
        linestyle="--",
        label="$\phi$",
    )
    
    # No y-ticks:
    # pred_interval_ax.set_yticks([])
    # pred_interval_ax.set_title("(d)")

    # pred_interval_ax.set_ylim([0, pred_interval_ax.get_ylim()[1]])
    # inset_pf = sofc_small_im.mean()
    # xerr = inset_pf - conf_start
    # pred_interval_ax.errorbar(
    #     sofc_small_im.mean(), 0.0002, xerr=xerr, fmt='o', capsize=6, color=COLOR_INSET, label="95% confidence interval", linewidth=LINE_W)
    # pred_interval_ax.legend(loc='upper left')
