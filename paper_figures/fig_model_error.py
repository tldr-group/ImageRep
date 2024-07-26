from representativity import prediction_error
import matplotlib.pyplot as plt
from scipy import stats 
import math
import numpy as np

# fill_color = [1, 0.49803922, 0.05490196, 0.2]
sigma_color = [0.4, 0.9, 0.0, 1]
model_color = "tab:cyan"
oi_color = "tab:orange"
# fill_color_dark = fill_color.copy()
# fill_color_dark[-1] = 0.5

def scatter_plot(ax, res, title, xlabel, ylabel):
    
    pred_data, fit_data, oi_data = res
    max_val = (np.max([np.max(fit_data), np.max(pred_data)]))
    x = np.linspace(0,max_val,100)
    ax.plot(x, x, label = 'Ideal predictions', color='black')

    ax.scatter(fit_data, oi_data, alpha=0.7, s=0.3, c=oi_color, label='Classical predictions')
    oi_errs = (fit_data-oi_data)/oi_data
    oi_mean = np.mean(oi_errs)
    ax.plot(x, x*(1-oi_mean), color=oi_color, linestyle='--', dashes=[2.5, 5], label="Mean")
    
    ax.scatter(fit_data, pred_data, alpha=0.5, s=0.3, c=model_color, label="Our model's predictions")
    model_errs = (fit_data-pred_data)/pred_data 
    model_mean = np.mean(model_errs)
    ax.plot(x, x*(1-model_mean), color=model_color, linestyle='--', dashes=[2.5, 5], label="Mean")

    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    model_errs = np.sort(model_errs) 
    std = np.std(model_errs) 
    
    # z = stats.norm.interval(0.9)[1]
    # err = std*z
    # print(f'std = {std}')
    # print(f'mean = {np.mean(errs)}')
    # ax.plot(x ,x/(1+err), c=fill_color_dark, ls='--', linewidth=1)
    # fill_1 = ax.fill_between(x, np.ones(x.shape[0])*(max_val),x/(1+err), alpha=0.2, label = f'95% confidence range', color=fill_color)
    ax.set_aspect('equal', adjustable='box')

with_cls = False
fig, axs = plt.subplots(2,3+with_cls)
fig.set_size_inches(4*(3+with_cls),8)
dims = ['2D', '3D']
edge_length = ['1536', '448']
for i, dim in enumerate(dims):
    pred_cls_all, fit_cls_all, oi_cls_all, _, vfs = prediction_error.pred_vs_fit_all_data(dim, edge_length[i], num_runs=9, std_not_cls=False)    
    cls_results = [pred_cls_all, fit_cls_all, oi_cls_all]
    # calculate the standard deviation instead of the cls:
    vfs = np.array(vfs)
    vfs_one_minus_vfs = vfs*(1-vfs)  
    dim_int = int(dim[0])
    cur_edge_length = int(edge_length[i])
    std_results = [((cls_res/cur_edge_length)**dim_int*vfs_one_minus_vfs)**0.5 for cls_res in cls_results]
    # pred_std_all = ((pred_cls_all/cur_edge_length)**dim_int*vfs_one_minus_vfs)**0.5
    # fit_std_all = ((fit_cls_all/cur_edge_length)**dim_int*vfs_one_minus_vfs)**0.5
    # std_results = [pred_std_all, fit_std_all]
    dim_str = dim[0]
    x_labels = [f'True CLS $a_{int(dim[0])}$', f'True Phase Fraction std $\sigma_{int(dim[0])}$']
    cls_math = r'\tilde{a}_{2}' if dim == '2D' else r'\tilde{a}_{3}'
    std_math = r'\tilde{\sigma}_{2}' if dim == '2D' else r'\tilde{\sigma}_{3}'
    y_labels = ['Predicted CLS $%s$' %cls_math, 'Predicted Phase Fraction std $%s$' %std_math]
    title_suffix = r'img size $%s^%s$' %(edge_length[i], dim_str)
    titles = [f'{dim} CLS comparison, '+title_suffix, f'{dim} std comparison, '+title_suffix]
    # sa_results = [err_exp_sa[pred_err_sa!=math.isnan], pred_err_sa[pred_err_sa!=math.nan]]
    
    for j, res in enumerate([cls_results, std_results]):
        ax_idx = j
        if not with_cls:
            if j == 0:
                continue
            else:
                ax_idx = 0
 
        ax = axs[i, ax_idx]
        
        # print(f'slope = {slope} and intercept = {intercept}')
        ax_xlabel = x_labels[j]
        ax_ylabel = y_labels[j]
        ax_title = titles[j]
        
        scatter_plot(ax, res, ax_title, ax_xlabel, ax_ylabel)
        
        if (j == 0 or not with_cls) :
            ax.legend(loc='upper left')        

    # Fit a normal distribution to the data:
    pred_std_all, fit_std_all = std_results[:2]
    errs = (fit_std_all-pred_std_all)/pred_std_all*100
    mu, std = stats.norm.fit(errs)

    ax_idx = 1 + with_cls
    ax2 = axs[i, ax_idx]
    
    # Plot the histogram.
    counts, bins = np.histogram(errs)
    
    max_val = np.max([np.max(errs), -np.min(errs)])
    y, x, _ = ax2.hist(errs, range=[-max_val, max_val], bins=50, alpha=0.6, color=model_color, density=True)

    # Plot the PDF.
    xmin, xmax = x.min(), x.max()
    max_abs = max(np.abs(np.array([xmin, xmax])))
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
        
    ax2.plot(x, p, linewidth=2, label=f'Fitted normal distribution')
    title = f'{dim} std PE distribution, {title_suffix}' 
    ax2.set_title(title)
    ax2.set_xlabel(r'Prediction Percentage Error ($\frac{\sigma_{%s}-\tilde{\sigma_{%s}}}{\tilde{\sigma_{%s}}}\cdot100$)' %(dim_str, dim_str, dim_str))
    # err = std*stats.norm.interval(0.9)[1]
    ax2.vlines(0, ymin=0, ymax=y.max(), color='black')
    ax2.vlines(std, ymin=0, ymax=y.max(), ls='--', color=sigma_color, label = r'$\sigma_{mod}$')
    # ax2.vlines(err, ymin=0, ymax=y.max(), ls='--', color=fill_color_dark, label = r'$\sigma_{mod}\cdot Z_{90\%}$')
    # trunc_x = x[x<err]
    # ax2.fill_between(trunc_x, p[x<err],np.zeros(trunc_x.shape[0]), alpha=0.2, color=fill_color)
    if i == 0:
        ax2.legend()

    # Plot the std error by size of edge length:
    ax3 = axs[i, ax_idx+1]
    run_data, _ = prediction_error.data_micros(dim)
    edge_lengths = run_data['edge_lengths_pred']
    start_idx = 2
    pred_error, stds = prediction_error.std_error_by_size(dim, edge_lengths, num_runs=10, start_idx=start_idx, std_not_cls=True)
    stds = stds*100  # convert to percentage error
    edge_lengths = edge_lengths[start_idx:]
    ax3.scatter(edge_lengths, stds, color='black')
    edge_length_pos = edge_lengths.index(cur_edge_length)
    ax3.scatter(edge_lengths[edge_length_pos], stds[edge_length_pos], color=sigma_color, label=r'$\sigma_{mod}$ for img size $%s^%s$'%(edge_length[i], dim_str))
    ax3.plot(edge_lengths, pred_error*100, label='Prediction error fit')
    ax3.set_xlabel('Img size')
    ax3.set_xticks(edge_lengths[::2], [r'$%s^%s$'%(i,dim_str) for i in edge_lengths[::2]])
    ax3.set_ylabel(r'Model percentage error std $\sigma_{mod}$ [%]')
    ax3.set_title(f'Error by Edge Length {dim}')
    if i == 0:
        ax3.legend(loc='upper right')

plt.tight_layout()
fig.savefig('fig_model_error.pdf', format='pdf')         




