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
    micro_ax.set_title("(a)")
    micro_ax.set_xticks([])
    micro_ax.set_yticks([])
    micro_ax.imshow(micro_368, cmap='gray', interpolation='nearest')
    

    # Create the prediction interval plot:
    pred_interval_ax = axs[1]
    conf_bounds, pf_1d, cum_sum_sum = get_prediction_interval_stats(micro_368, conf_level=0.5, n_divisions=1001)

    print("Confidence bounds:", conf_bounds)
    print(f"percentage from image phase fraction: {(conf_bounds - micro_368.mean())/micro_368.mean()}")
    # cumulative sum to the original data:
    original_data = np.diff(cum_sum_sum)
    val = np.diff(pf_1d)[0]
    original_data = original_data/val
    pred_interval_ax.plot(pf_1d[1:], original_data, label="ImageRep likelihood of graphite phase fraction")

    pred_interval_ax.set_xlabel('Phase fraction')

    # Fill between confidence bounds:
    conf_start, conf_end = conf_bounds
    # Fill between the confidence bounds under the curve:
    pred_interval_ax.fill_between(
        pf_1d[:-1], 
        original_data, 
        where=(pf_1d[1:] >= 0) & (pf_1d[1:] <= conf_start), 
        alpha=0.3,
        color='blue',
        label="50% (negative) confidence interval(s)"
        )
    
    pred_interval_ax.fill_between(
        pf_1d[:-1], 
        original_data, 
        where=(pf_1d[:-1] >= conf_end) & (pf_1d[1:] <= 1), 
        alpha=0.3,
        color='blue'
        )
    # Plot in dashed vertical lines the materials phase fraction and the inset phase fraction:
    phi = micro_368.mean()
    pred_interval_ax.vlines(
        phi,
        0,
        np.max(original_data),
        linestyle="--",
        label="Graphite phase fraction in image",
    )
    
    pred_interval_ax.set_title("(b)")
    # have a legend:
    pred_interval_ax.legend(loc='upper right')

    # Make the distribution smaller so that the legened won't disturb it in the plot:
    pred_interval_ax.set_ylim(0, pred_interval_ax.get_ylim()[1]*1.4)
    pred_interval_ax.set_xlim(phi-0.10, phi+0.10)

    # Save the figure:
    plt.tight_layout()
    plt.savefig("paper_figures/output/SI_negative_result.pdf", dpi=300)
