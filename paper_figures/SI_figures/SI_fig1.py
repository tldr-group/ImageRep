from representativity import prediction_error
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
import numpy as np

def scatter_plot(ax, res, title, xlabel, ylabel):
    
    pred_data, fit_data = res
    
    # color = [[vfs[i],0,1] for i in range(len(vfs))]
    ax.scatter(fit_data, pred_data, s=0.2, label='Predictions')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    errs = (fit_data-pred_data)/pred_data 
    max_val = (np.max([np.max(fit_data), np.max(pred_data)]))
    print(max_val)
    x = np.linspace(0,max_val,100)
    ax.plot(x, x, label = 'Ideal predictions', color='k')
    # ax.set_yticks(np.arange(0, max_val, max_val/5))
    # ax.set_xticks(np.arange(0, max_val, max_val/5))
    # ax.set_xlim(right=max_val)
    # ax.set_ylim(top=max_val)
    errs = np.sort(errs) 
    std = np.std(errs) 
    
    z = norm.interval(0.9)[1]
    err = std*z
    print(f'std = {std}')
    print(f'mean = {np.mean(errs)}')
    # print(f'error = {err}')
    # ax.plot(np.arange(0, max_val, 0.001), np.arange(0, max_val, 0.001), c='k')
    ax.plot(x ,x/(1+err), c='orange', ls='--', linewidth=1)
    fill_1 = ax.fill_between(x, np.ones(x.shape[0])*(max_val),x/(1+err), alpha=0.2, label = f'95% confidence range')
    ax.set_aspect('equal', adjustable='box')
    return errs, err


fig, axs = plt.subplots(2,4)
fig.set_size_inches(16,8)
dims = ['2D', '3D']
edge_length = ['1536', '448'] 
for i, dim in enumerate(dims):
    # data.pop('microstructure054')
    dim_data, micro_names = prediction_error.data_micros(dim)
    pred_cls_all, fit_cls_all, _, vfs = prediction_error.pred_vs_fit_all_data(dim, edge_length[i], num_runs=9)    
    cls_results = [pred_cls_all, fit_cls_all]
    # calculate the standard deviation instead of the cls:
    vfs = np.array(vfs)
    vfs_one_minus_vfs = vfs*(1-vfs)  
    dim_int = int(dim[0])
    cur_edge_length = int(edge_length[i])
    pred_std_all = ((pred_cls_all/cur_edge_length)**dim_int*vfs_one_minus_vfs)**0.5
    fit_std_all = ((fit_cls_all/cur_edge_length)**dim_int*vfs_one_minus_vfs)**0.5
    std_results = [pred_std_all, fit_std_all]
    dim_str = dim[0]
    x_labels = [f'True CLS $a_{int(dim[0])}$', f'True Phase Fraction std $\sigma_{int(dim[0])}$']
    cls_math = r'\tilde{a}_{2}' if dim == '2D' else r'\tilde{a}_{3}'
    std_math = r'\tilde{\sigma}_{2}' if dim == '2D' else r'\tilde{\sigma}_{3}'
    y_labels = ['Predicted CLS $%s$' %cls_math, 'Predicted Phase Fraction std $%s$' %std_math]
    title_suffix = r'img size $%s^%s$' %(edge_length[i], dim_str)
    titles = [f'{dim} CLS comparison, '+title_suffix, f'{dim} std comparison, '+title_suffix]
    # sa_results = [err_exp_sa[pred_err_sa!=math.isnan], pred_err_sa[pred_err_sa!=math.nan]]
    for j, res in enumerate([cls_results, std_results]):

        ax = axs[i, j]
        
        # print(f'slope = {slope} and intercept = {intercept}')
        ax_xlabel = x_labels[j]
        ax_ylabel = y_labels[j]
        ax_title = titles[j]
        
        _ = scatter_plot(ax, res, ax_title, ax_xlabel, ax_ylabel)
        
        if j == 0 and dim == '2D':
            ax.legend(loc='upper left')        

    # Fit a normal distribution to the data:
    errs = (fit_cls_all-pred_cls_all)/pred_cls_all
    mu, std = norm.fit(errs)

    ax2 = axs[i, 2]
    
    # Plot the histogram.
    counts, bins = np.histogram(errs)
    
    max_val = np.max([np.max(errs), -np.min(errs)])
    y, x, _ = ax2.hist(errs, range=[-max_val, max_val], bins=50, alpha=0.6, density=True)

    # Plot the PDF.
    xmin, xmax = x.min(), x.max()
    print(xmin, xmax)
    max_abs = max(np.abs(np.array([xmin, xmax])))
    x = np.linspace(xmin, xmax, 100)
    # ax2.set_xlim(1.6, -1.6)
    p = norm.pdf(x, mu, std)
    fill_color = [1, 0.49803922, 0.05490196, 0.2]
    print(fill_color)
    fill_color_dark = fill_color.copy()
    fill_color_dark[-1] = 0.5
    ax2.plot(x, p, 'k', linewidth=2, color=fill_color_dark, label=f'Fitted normal distribution')
    title = f'{dim} std PE distribution, {title_suffix}' 
    ax2.set_title(title)
    # ax2.set_ylabel(f'Histogram')
    ax2.set_xlabel(r'$\frac{\sigma-\tilde{\sigma}}{\tilde{\sigma}}$')
    # ax2.set_aspect(2*x.max()/y.max())
    # ax2.set_yticks([])
    err = std*norm.interval(0.90)[1]
    ax2.vlines(0, ymin=0, ymax=y.max(), color='black')
    ax2.vlines(err, ymin=0, ymax=y.max(), ls='--', color='orange')
    trunc_x = x[x<err]
    ax2.fill_between(trunc_x, p[x<err],np.zeros(trunc_x.shape[0]), alpha=0.2, color=fill_color)
    if i == 0:
        ax2.legend(loc='upper left')

    # Plot the std error by size of edge length:
    ax3 = axs[i, 3]
    run_data, _ = prediction_error.data_micros(dim)
    edge_lengths = run_data['edge_lengths_pred']
    pred_error, stds = prediction_error.std_error_by_size(dim, edge_lengths, num_runs=10, std_not_cld=True)
    ax3.scatter(edge_lengths, stds*100, label='Prediction error std')
    ax3.plot(edge_lengths, pred_error, label='Prediction error fit')
    ax3.set_xlabel('Edge Length')
    ax3.set_ylabel('MPE std (%)')
    ax3.set_title(f'Error by Edge Length {dim}')
    
plt.tight_layout()
fig.savefig('SI_fig1.pdf', format='pdf')         




