from representativity import util
import torch
import matplotlib.pyplot as plt
import json
import numpy as np
from representativity import microlib_statistics as ms
from scipy.spatial.distance import cdist
import time

def generate_sg_tpc(netG, mode, imsize):
    '''
    This function is used to predict the integral range of the microstructure.
    '''
    lf = imsize//32 + 2  # the size of G's input
    single_img = util.generate_image(netG, lf=lf, threed=mode=='3D', reps=1)
    if single_img.any():
        single_img = single_img.cpu()[0]
        dims = len(single_img.shape)
        tpc = util.tpc_radial(single_img, threed=dims == 3)
        return tpc

def tpc_by_radius(tpc):
    tpc = np.array(tpc)
    middle_idx = np.array(tpc.shape)//2
    vf = tpc[tuple(map(slice, middle_idx, middle_idx+1))].item()
    # print(f'vf squared = {np.round(vf**2, 5)}')
    dist_arr = np.indices(tpc.shape)
    dist_arr = np.abs((dist_arr.T - middle_idx.T).T)
    img_volume = np.prod(middle_idx+1)
    vec_arr = np.prod(middle_idx[0]+1 - dist_arr, axis=0)/img_volume
    dist_arr = np.sqrt(np.sum(dist_arr**2, axis=0))
    end_dist = middle_idx[0]
    sum_circle = np.sum(vec_arr[dist_arr<=end_dist])
    n_bins = 101
    jump = sum_circle/n_bins
    # dist_indices = [0, 88, 126, 156, 182, 205, 226, 246, 264, 282, 299, 315, 331, 346, 361, 375, 389, 403, 417, 430, 443, 456, 469, 482, 494, 506, 518, 530, 542, 554, 566, 578, 590, 602, 614, 625, 636, 647, 658, 669, 680, 691, 702, 713, 724, 735, 746, 757, 768, 779, 790, 801, 812, 823, 834, 845, 856, 868, 880, 892, 904, 916, 928, 940, 952, 964, 976, 988, 1000, 1013, 1026, 1039, 1052, 1065, 1078, 1091, 1105, 1119, 1133, 1147, 1162, 1177, 1192, 1208, 1224, 1241, 1258, 1276, 1295, 1314, 1334, 1356, 1379, 1404, 1431, 1462, 1499]
    dist_indices = [0]
    tpc_res = [vf]  # tpc_res is the tpc result of the tpc by growing radiuses
    tpc_vec = vec_arr*tpc
    for i in range(0, end_dist, 1):
    # for i in range(len(dist_indices)-1):
        # dist_bool = (dist_arr>dist_indices[i]) & (dist_arr<=dist_indices[i+1])
        dist_bool = (dist_arr>=dist_indices[-1]) & (dist_arr<i) 
        if np.sum(vec_arr[dist_bool]) > jump:
            dist_indices.append(i)
            tpc_res.append(np.sum(tpc_vec[dist_bool])/np.sum(vec_arr[dist_bool]))
    return vf, vf**2, tpc_res, dist_indices

def tpc_check():
    all_data, micros, netG, v_names, run_v_names = ms.json_preprocessing()
    
    edge_lengths_pred = all_data[f'data_gen_2D']['edge_lengths_pred']
    for j, p in enumerate(micros):

        try:
            netG.load_state_dict(torch.load(p + "_Gen.pt"))
        except:  # if the image is greayscale it's excepting because there's only 1 channel
            continue
        n = p.split('/')[-1]
        args = (edge_lengths_pred[10], netG, '2D')
        tpc_results, vfs, vf_squares = tpcs_radius(generate_sg_tpc, args)
            # print(f'{len(vf_squares)} / {test_runs} done for {n}')
        mean_tpc_results = np.mean(np.stack(tpc_results,axis=0), axis=0)
        plt.plot(mean_tpc_results, label='mean tpc')
        real_vf_squared = np.mean(vfs)**2
        # print(f'real vf squared = {np.round(real_vf_squared, 6)}')
        # print(f'vf squared = {np.round(np.mean(vf_squares), 6)}')
        plt.plot([real_vf_squared]*len(mean_tpc_results), label='real vf squared')
        plt.plot([np.mean(vf_squares)]*len(mean_tpc_results), label='vf squared')
        plt.xlabel('Growing Radius')
        plt.ylabel('TPC')
        plt.legend()
        plt.savefig(f'tpc_results/{n}_tpc.png')
        plt.close()
        # print(f'end tpc = {np.round(np.mean(end_tpc_results), 6)}')
        # print(f'end tpc std = {np.round(np.std(end_tpc_results), 6)}\n')
        print(f'{p} done\n')

def tpcs_radius(gen_func, test_runs, args):
    tpc_results = []
    tpcs = []
    vf_squares = []
    vfs = []
    for _ in [args[0]]*test_runs:
        img_tpc = gen_func(*args)
        tpcs.append(img_tpc)
        vf, vf_square, tpc_res, distances = tpc_by_radius(img_tpc)
        vfs.append(vf)
        tpc_results.append(np.array(tpc_res))
        vf_squares.append(vf_square)
        if (len(vf_squares) % 10) == 0:
            print(f'{len(vf_squares)} / {test_runs} done.') 
    return tpc_results, vfs, vf_squares, distances

def make_tpc(img):
    dims = len(img.shape)
    tpc = util.tpc_radial(img, threed=dims == 3)
    return tpc

def make_circles_tpc(imsize, circle_radius, vf):
    img = make_circles_2D(imsize, circle_radius, vf)
    return make_tpc(img)

def make_circles_2D(imsize, circle_radius, vf):
    '''
    This function is used to create an image with circles of the same size, 
    which are randomly placed in the image. The vf is the volume fraction of
    the image, in expectation.
    '''
    img = np.zeros([imsize+2*circle_radius]*2)
    circle_area = np.pi*circle_radius**2
    # the probability of a pixel being in a circle (1 minus not appearing in any
    # circle around it):
    p_of_center_circle = 1 - (1-vf)**(1/circle_area)
    circle_centers = np.random.rand(*img.shape) < p_of_center_circle
    circle_centers = np.array(np.where(circle_centers))
    time_before_circles = time.time()
    fill_img_with_circles(img, circle_radius, circle_centers)
    return img[circle_radius:-circle_radius, circle_radius:-circle_radius]

def fill_img_with_circles(img, circle_radius, circle_centers):
    '''Fills the image with circles of the same size given by the cicle_radius,
    with the centers given in circle_centers.'''
    dist_arr = np.indices(img.shape)
    dist_arr_T = dist_arr.T
    dist_arr_reshape = dist_arr_T.reshape(np.product(dist_arr_T.shape[:2]), dist_arr_T.shape[-1])
    distances = cdist(dist_arr_reshape, circle_centers.T)
    if distances.size == 0:
        return img
    min_distances = np.min(distances, axis=1).reshape(img.shape)
    img[min_distances<=circle_radius] = 1
    return img

if __name__ == '__main__':
    # tpc_check()
    vfs =[]
    imsize = 200
    circle_size = 20
    vf = 0.1
    args = (imsize, circle_size, vf)    
    run_tests = 6000
    tpc_results, vfs, vf_squares, dist_len = tpcs_radius(make_circles_tpc, run_tests, args=args)
    dist_len = np.array(dist_len)
    plt.plot(np.mean(tpc_results, axis=0), label='Mean TPC')

    vf_squared = np.mean(vf_squares)
    label_vf_squared = f'$E[\Phi^2]$ = {np.round(vf_squared, 4)}'
    plt.plot([vf_squared]*len(tpc_results[0]), label=label_vf_squared)
    # print(f'vf squared = {np.round(vf_squared, 7)}')

    true_vf_squared = np.mean(vfs)**2   
    label_true_vf_squared = f'$E[\Phi]^2$ = {np.round(true_vf_squared, 4)}'
    plt.plot([true_vf_squared]*len(tpc_results[0]), label=label_true_vf_squared)
    # print(f'true vf squared = {np.round(true_vf_squared, 7)}')
    
    plt.axvline(x=np.where(dist_len==circle_size*2)[0][0], color='black', linestyle='--', label='Circle Diameter')

    fill_1 = plt.fill_between(np.arange(len(tpc_results[0])), np.mean(tpc_results, axis=0),[vf_squared]*len(tpc_results[0]), alpha=0.2, 
                              where=np.mean(tpc_results, axis=0)>=vf_squared,label = f'Area A')
    fill_2 = plt.fill_between(np.arange(len(tpc_results[0])),[vf_squared]*len(tpc_results[0]), np.mean(tpc_results, axis=0), alpha=0.2, 
                              where=np.mean(tpc_results, axis=0)<=vf_squared,label = f'Area B')
    fill_3 = plt.fill_between(np.arange(len(tpc_results[0])), [vf_squared]*len(tpc_results[0]),[true_vf_squared]*len(tpc_results[0]), alpha=0.2, 
                              where=np.mean(tpc_results, axis=0)>=vf_squared,label = f'Area C')

    plt.text(2.5, 0.02, 'A', fontsize=12)
    plt.text(2.5, 0.01046, 'C', fontsize=12)
    plt.text(40, 0.01046, 'B', fontsize=12)
    plt.text(10, 0.079, '$\Phi$ calculates phase fraction.', fontsize=12)
    plt.text(10, 0.068, 'The variance of $\Phi$ is:', fontsize=12)
    plt.text(10, 0.058, '$Var[\Phi]=E[\Phi^2]-E[\Phi]^2$', fontsize=12)
    plt.text(10, 0.049, '$Var[\Phi]=Area(C)+Area(B)$', fontsize=12)
    plt.text(10, 0.041, '$Area(C)=(40/160)\cdot Area(B)$', fontsize=12)
    plt.text(10, 0.0335, 'So,', fontsize=12)
    plt.text(10, 0.0285, '$Var[\Phi]=(200/160)\cdot Area(B)$', fontsize=12)
    plt.text(10, 0.024, '$Area(A)$ is computed by our model.', fontsize=12)
    plt.text(10, 0.0205, '$Area(A)=Area(B)$', fontsize=12)
    plt.text(10, 0.0175, 'Which result in:', fontsize=12)
    plt.text(10, 0.0150, '$Var[\Phi]=(200/160)\cdot Area(A)$', fontsize=12)

    plt.title(f'PF = {vf}, Circle Diameter = {circle_size*2}, Image Size = {imsize}')
    
    plt.ylim(true_vf_squared-0.001, vf+0.01)
   
    plt.xticks(np.arange(0,len(tpc_results[0]),5), dist_len[::5])
    plt.xlabel('TPC distance')
    plt.ylabel('TPC (log scale)')
    plt.yscale('log')
    plt.yticks([0.01, 0.1], [0.01, 0.1])
    plt.legend()
    plt.savefig(f'tpc_results/circles_tpc_test.png')

    

    
