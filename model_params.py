import numpy as np
import util
import json
from scipy.stats import norm
from scipy.optimize import minimize

with open("data.json", "r") as fp:
    datafin = json.load(fp)

predicted_IR_vf = []
stat_analysis_IR_vf = []  # TODO add sa and 3D


gen_data = datafin["generated_data"]

for micro_n in gen_data:
    cur_micro = gen_data[micro_n]
    tpc_vf_dist, tpc_vf = cur_micro["tpc_vf_dist"], cur_micro["tpc_vf"]
    pred_IR = util.tpc_to_fac(tpc_vf_dist, tpc_vf)
    predicted_IR_vf.append(pred_IR)
    stat_analysis_IR_vf.append(cur_micro["fac_vf"])

predicted_IR_vf, stat_analysis_IR_vf = np.array(predicted_IR_vf), np.array(stat_analysis_IR_vf)

model_errors = (stat_analysis_IR_vf - predicted_IR_vf)/stat_analysis_IR_vf

mu, std = norm.fit(model_errors)