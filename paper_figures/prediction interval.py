import numpy as np
# from scipy.stats import norm
import matplotlib.pyplot as plt

def likelihood_of_phi(image_pf, pred_std, std_dist_std):
    std_dist_divisions = 101
    num_stds = 5
    # first make the "weights distribution", the normal distribution of the stds
    # where the prediction std is the mean of this distribution.
    x_std_dist_bounds = [pred_std - num_stds*std_dist_std, pred_std + num_stds*std_dist_std]
    x_std_dist = np.linspace(*x_std_dist_bounds, std_dist_divisions)
    std_dist = normal_dist(x_std_dist, mean=pred_std, std=std_dist_std)
    plt.plot(x_std_dist, std_dist)
    plt.show()
    integral_std_dist = np.trapz(std_dist, x_std_dist)
    print(integral_std_dist)
    # Next, make the pf distributions, each row correspond to a different std, with 
    # the same mean (observed pf) but different stds (x_std_dist), multiplied by the
    # weights distribution (std_dist).
    pf_locs = np.ones((std_dist_divisions,std_dist_divisions))*image_pf
    pf_x_bounds = [image_pf - num_stds*pred_std, image_pf + num_stds*pred_std]
    pf_x_1d = np.linspace(*pf_x_bounds, std_dist_divisions)
    pf_mesh, std_mesh = np.meshgrid(pf_x_1d, x_std_dist)
    # before normalising by weight:
    pf_dist_before_norm = normal_dist(pf_mesh, mean=pf_locs, std=std_mesh)
    plt.contourf(pf_mesh, std_mesh, pf_dist_before_norm, levels=100, cmap = 'jet') 
    plt.show()
    # normalise by weight:
    pf_dist = (pf_dist_before_norm.T * std_dist).T
    plt.contourf(pf_mesh, std_mesh, pf_dist, levels=100, cmap = 'jet') 
    plt.title('Likelihood of $\phi$')
    plt.xlabel('Phase fraction')
    plt.ylabel('Standard deviation')
    plt.vlines(image_pf, x_std_dist_bounds[0], x_std_dist_bounds[1], colors='g', label = 'Observed $\Phi(\omega)$')
    plt.hlines(pred_std, pf_x_bounds[0], pf_x_bounds[1], colors='orange', label = r'Predicted std $\tilde{\sigma}$')
    plt.legend()
    pf_dist_integral_col = np.trapz(pf_dist, x_std_dist, axis=0)
    pf_dist_integral = np.trapz(pf_dist_integral_col, pf_x_1d)
    plt.show()
    
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(pf_mesh, std_mesh, pf_dist, cmap='viridis',\
    #                 edgecolor='green')
    # plt.show()
    # print(pf_dist_integral)
    
    sum_dist = np.sum(pf_dist, axis=0)*np.diff(pf_x_1d)[0]*np.diff(x_std_dist)[0]
    print(sum_dist)
    print(np.sum(sum_dist))


def normal_dist(x, mean, std):
    return (1.0 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

if __name__ == "__main__":
    image_pf = 0.5
    pred_std = 0.04
    std_dist_std = 0.005
    likelihood_of_phi(image_pf, pred_std, std_dist_std)