import numpy as np
import util
import json
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


with open("data.json", "r") as fp:
    datafin = json.load(fp)

predicted_IR_err_vf = []
stat_analysis_IR_err_vf = []  # TODO add sa and 3D

# data = '2D'
# data = datafin[f'validation_data{data}']
# statistical_err_vf = np.array([data[n]['err_exp_vf'] for n in data.keys()])
# statistical_err_sa = np.array([data[n]['err_exp_sa'] for n in data.keys()])
# pred_err_vf = np.array([data[n]['pred_err_vf'] for n in data.keys()])
# pred_err_sa = np.array([data[n]['pred_err_sa'] for n in data.keys()])

gen_data = datafin["generated_data"]
conf=0.95

for micro_n in gen_data:
    cur_micro = gen_data[micro_n]
    tpc_vf_dist, tpc_vf = cur_micro["tpc_vf_dist"], cur_micro["tpc_vf"]
    pred_IR = util.tpc_to_fac(tpc_vf_dist, tpc_vf)
    
    k = 1000
    m_pred = k/pred_IR
    m_statistical = k/cur_micro["fac_vf"]
    vf = cur_micro["vf"]
    z = norm.interval(conf)[1]
    pred_err = z*((vf*(1-vf))**0.5)/m_pred/vf*100
    statistical_err = z*((vf*(1-vf))**0.5)/m_statistical/vf*100
    predicted_IR_err_vf.append(pred_err)
    stat_analysis_IR_err_vf.append(statistical_err)
    # print(stats.norm.interval(conf, scale=std_bern)[1], std_bern)

predicted_IR_err_vf, stat_analysis_IR_err_vf = np.array(predicted_IR_err_vf), np.array(stat_analysis_IR_err_vf)

def scatter_plot_IR(stat_IR, pred_IR, std):
    plt.scatter(stat_IR, pred_IR, s=4, label='Predictions')
    x = np.arange(0,20)
    plt.plot(x,x,label = 'Ideal fit', c='orange')
    plt.xlabel('VF error from statistical analysis')
    plt.ylabel('VF error from tpc analysis')
    plt.locator_params(nbins=4)
    plt.xticks()
    ax = plt.gca()
    # err = std*norm.interval(conf)[1]/np.sqrt(78)
    # print(err)
    # ax.plot(x - x*err, x, c='black', ls='--', linewidth=1)
    # ax.plot(x ,x -x*err, label = f'95% confidence ', c='black', ls='--', linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    plt.legend(loc='lower right')
    plt.savefig('fac_pred.png')
    plt.show()

# it's not clear where does this 3 is coming from
initial_guess = np.array([3, 0])  # slope and intercept initial guess
minimise_args = (predicted_IR_err_vf, stat_analysis_IR_err_vf)
bounds = [(0,3.5),(-10,10)]
best_slope_and_intercept = minimize(util.mape_linear_objective, initial_guess, args=minimise_args, bounds=bounds)
print(best_slope_and_intercept)

# scatter_plot_IR(stat_analysis_IR_vf, predicted_IR_vf)

slope, intercept = best_slope_and_intercept.x
new_pred_err_vf = util.linear_fit(predicted_IR_err_vf, slope, intercept)

model_errors = (stat_analysis_IR_err_vf - new_pred_err_vf)/stat_analysis_IR_err_vf

mu, std = norm.fit(model_errors)

scatter_plot_IR(stat_analysis_IR_err_vf, new_pred_err_vf, std)

