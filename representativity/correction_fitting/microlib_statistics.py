import os
from representativity.old import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
import time
from scipy.stats import norm


"""
File: microlib_statistics.py

Description: This script is used to generate the statistics for the representativity of 
microstructures from the microlib dataset. 
The statistics generated are then saved in a json file, to be analyzed and used for 
representativity prediction of a new microstructure or micrograph.
"""


def insert_v_names_in_all_data(all_data, mode, n, v_names, run_v_names, run_number=0):
    """
    This function is used to insert the names of the variables that will be used to store the statistics
    """
    dim_data = all_data[f"data_gen_{mode}"]
    if n not in dim_data:
        dim_data[n] = {}
    for v_name in v_names:
        if v_name not in dim_data[n]:
            dim_data[n][v_name] = {}
    micro_data = dim_data[n]
    if f"run_{run_number}" not in micro_data:
        micro_data[f"run_{run_number}"] = {}
    run_data = micro_data[f"run_{run_number}"]
    for run_v_name in run_v_names:
        if run_v_name not in run_data:
            run_data[run_v_name] = {}


def run_ir_prediction(netG, n, mode, imsize, all_data, conf, run_number=0):
    """
    This function is used to predict the integral range of the microstructure.
    """
    run_data = all_data[f"data_gen_{mode}"][n][f"run_{run_number}"]
    lf = imsize // 32 + 2  # the size of G's input
    single_img = util.generate_image(netG, lf=lf, threed=mode == "3D", reps=1)
    if single_img.any():
        single_img = single_img.cpu()[0]
        pred_err_vf, _, pred_ir_vf = util.make_error_prediction(
            single_img, conf=conf, model_error=False, correction=False, mxtpc=200
        )
        print(f"pred ir {imsize} = {np.round(pred_ir_vf, 3)}")
        im_vf = single_img.mean().cpu().item()
        one_im_stat_pred = util.one_img_stat_analysis_error(single_img, im_vf)
        print(f"one im stat pred ir = {one_im_stat_pred}")
        if "pred_ir_one_im_fit_vf" not in run_data:
            run_data["pred_ir_one_im_fit_vf"] = {}
        run_data["pred_ir_one_im_fit_vf"][str(imsize)] = one_im_stat_pred
        run_data["pred_ir_vf"][str(imsize)] = pred_ir_vf
        run_data["pred_err_vf"][str(imsize)] = pred_err_vf


def run_statistical_fit_analysis(netG, n, mode, edge_lengths_fit, all_data):
    """
    This function is used to run the statistical fit analysis on the microstructure,
    and find the "true" integral range of the microstructure.
    """
    imsize = 512 if mode == "3D" else 1600
    lf = imsize // 32 + 2  # the size of G's input
    reps = 50 if mode == "3D" else 150
    many_imgs = util.generate_image(netG, lf=lf, threed=mode == "3D", reps=reps)
    vf = torch.mean(many_imgs).cpu().item()
    print(f"{n} vf = {vf}")
    all_data[f"data_gen_{mode}"][n]["vf"] = vf
    fit_ir_vf = util.stat_analysis_error(many_imgs, vf, edge_lengths_fit)
    cur_fit_ir_vf = all_data[f"data_gen_{mode}"][n]["fit_ir_vf"]
    print(f"{n} cur fit ir vf = {cur_fit_ir_vf}")
    print(f"{n} fit ir vf = {fit_ir_vf}")
    # fit_ir_vf_oi = util.stat_analysis_error(many_imgs[0].unsqueeze(0), vf, edge_lengths_fit)
    fit_ir_vf_classic = util.stat_analysis_error_classic(many_imgs, vf)
    print(f"{n} fit ir vf classic = {fit_ir_vf_classic}")
    all_data[f"data_gen_{mode}"][n]["fit_ir_vf"] = fit_ir_vf
    return fit_ir_vf, vf


def compare_statistical_fit_error(
    n, mode, edge_lengths_pred, fit_ir_vf, vf, all_data, conf
):
    """
    This function is used to compare the statistical fit error to the prediction error.
    """
    n_dims = 2 if mode == "2D" else 3
    img_sizes = [(l,) * n_dims for l in edge_lengths_pred]
    fit_errs_vf = util.bernouli(vf, util.ns_from_dims(img_sizes, fit_ir_vf), conf=conf)
    for i in range(len(edge_lengths_pred)):
        imsize = edge_lengths_pred[i]
        all_data[f"data_gen_{mode}"][n]["fit_err_vf"][imsize] = fit_errs_vf[i]


def json_preprocessing():
    """
    This function is used to load the data from the microlib dataset, and to prepare the json file
    """

    # Load the statistics file
    with open("microlib_statistics_periodic.json", "r") as fp:
        all_data = json.load(fp)

    # Dataset path and list of subfolders
    with open("micro_names.json", "r") as fp:
        micro_names = json.load(fp)
    micros = [f"/home/amir/microlibDataset/{p}/{p}" for p in micro_names]
    # Load placeholder generator
    netG = util.load_generator(micros[0])

    v_names = ["vf", "fit_err_vf"]
    run_v_names = ["pred_ir_vf", "pred_err_vf", "pred_ir_one_im_fit_vf"]

    modes = ["2D", "3D"]
    for mode in modes:
        if f"data_gen_{mode}" not in all_data:
            all_data[f"data_gen_{mode}"] = {}

    # Edge lengths for the experimental statistical analysis:
    all_data["data_gen_2D"]["edge_lengths_fit"] = list(
        range(500, 1000, 20)
    )  # TODO change back to 500
    all_data["data_gen_3D"]["edge_lengths_fit"] = list(range(350, 450, 10))

    # Edge lengths for the predicted integral range:
    all_data["data_gen_2D"]["edge_lengths_pred"] = list(range(8 * 32, 65 * 32, 4 * 32))
    all_data["data_gen_3D"]["edge_lengths_pred"] = list(range(4 * 32, 19 * 32, 1 * 32))

    return all_data, micros, netG, v_names, run_v_names


def run_microlib_statistics(
    cur_modes=["2D", "3D"], run_s=False, run_p=True, run_number=0
):
    """
    This function is used to run the statistical analysis on the microlib dataset.
    It will generate the statistics for each microstructure in the dataset, and save it in a json file.
    param cur_modes: list of modes to run the statistical analysis on.
    param run_s: if True, run the statistical fit analysis.
    param run_p: if True, run the prediction of the integral range.
    param run_number: the number of the run for the integral range prediction.
    """

    all_data, micros, netG, v_names, run_v_names = json_preprocessing()

    total_time_0 = time.time()
    # run the statistical analysis on the microlib dataset
    for _, p in enumerate(micros):

        try:
            netG.load_state_dict(torch.load(p + "_Gen.pt"))
        except:  # if the image is greayscale it's excepting because there's only 1 channel
            continue

        t_micro = time.time()
        for mode in cur_modes:

            print(f"{mode} mode")

            edge_lengths_fit = all_data[f"data_gen_{mode}"]["edge_lengths_fit"]
            edge_lengths_pred = all_data[f"data_gen_{mode}"]["edge_lengths_pred"]

            n = p.split("/")[-1]
            insert_v_names_in_all_data(
                all_data, mode, n, v_names, run_v_names, run_number
            )  # insert var names in all_data

            conf = 0.95  # confidence level for the prediction and stat. fit error

            if run_p:
                print(f"{n} starting prediction")
                for imsize in edge_lengths_pred:
                    run_ir_prediction(netG, n, mode, imsize, all_data, conf, run_number)

            if run_s:
                print(f"{n} starting statistical fit")
                fit_ir_vf, vf = run_statistical_fit_analysis(
                    netG, n, mode, edge_lengths_fit, all_data
                )
                compare_statistical_fit_error(
                    n, mode, edge_lengths_pred, fit_ir_vf, vf, all_data, conf
                )

            with open(f"microlib_statistics_periodic.json", "w") as fp:
                json.dump(all_data, fp)

        print(
            f"\ntime for micro {n} {cur_modes} = {np.round((time.time()-t_micro)/60, 2)} minutes\n"
        )

        print(f"{_+1}/{len(micros)} microstructures done")
    print(f"total time = {np.round((time.time()-total_time_0)/60, 2)} minutes")


def main_run_microlib_statistics(
    cur_modes=["2D", "3D"], run_s=False, run_p=True, num_runs=5
):
    # first run the statistical analysis on the microlib dataset
    if run_s:
        print(f"Running statistical analysis fit on {cur_modes} mode(s)")
        run_microlib_statistics(cur_modes=cur_modes, run_s=True, run_p=False)
    # then run the integral range prediction multiple times
    if run_p:
        print(f"Running integral range prediction on {cur_modes} mode(s)")
        for run_number in range(num_runs, 10):
            print(f"Run number {run_number}\n")
            run_microlib_statistics(
                cur_modes, run_s=False, run_p=True, run_number=run_number
            )


if __name__ == "__main__":
    main_run_microlib_statistics(cur_modes=["3D"], run_s=False, run_p=True, num_runs=9)
