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
    end_dist = int(np.max(dist_arr))  # TODO this needs to be changed to the maximum distanse from the center 200*sqrt(2)
    sum_circle = np.sum(vec_arr[dist_arr<=end_dist])
    n_bins = 81
    jump = sum_circle/n_bins
    dist_indices = [0]
    tpc_res = [vf]  # tpc_res is the tpc result of the tpc by growing radiuses
    tpc_vec = vec_arr*tpc
    for i in range(0, end_dist+1, 1):
    # for i in range(len(dist_indices)-1):
        # dist_bool = (dist_arr>dist_indices[i]) & (dist_arr<=dist_indices[i+1])
        dist_bool = (dist_arr>=dist_indices[-1]) & (dist_arr<i) 
        if np.sum(vec_arr[dist_bool]) > jump or i == end_dist:
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
        tpc_fig.plot(mean_tpc_results, label='mean tpc')
        real_vf_squared = np.mean(vfs)**2
        # print(f'real vf squared = {np.round(real_vf_squared, 6)}')
        # print(f'vf squared = {np.round(np.mean(vf_squares), 6)}')
        tpc_fig.plot([real_vf_squared]*len(mean_tpc_results), label='real vf squared')
        tpc_fig.plot([np.mean(vf_squares)]*len(mean_tpc_results), label='vf squared')
        tpc_fig.xlabel('Growing Radius')
        tpc_fig.ylabel('TPC')
        tpc_fig.legend()
        tpc_fig.savefig(f'tpc_results/{n}_tpc.png')
        tpc_fig.close()
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
    tpc = util.tpc_radial(img, threed=dims == 3, periodic=False)
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
    dist_arr_reshape = dist_arr_T.reshape(np.prod(dist_arr_T.shape[:2]), dist_arr_T.shape[-1])
    distances = cdist(dist_arr_reshape, circle_centers.T)
    if distances.size == 0:
        return img
    min_distances = np.min(distances, axis=1).reshape(img.shape)
    img[min_distances <= circle_radius] = 1
    return img

if __name__ == '__main__':
    # tpc_check()
    vfs =[]
    imsize = 100
    circle_size = 20
    vf = 0.3
    args = (imsize, circle_size, vf)    
    run_tests = 20000

    fig, axs = plt.subplot_mosaic([['circle1', 'circle2', 'TPC figure', 'TPC figure'],
                                    ['circle3', 'circle4', 'TPC figure', 'TPC figure']])
    fig.set_size_inches(16,16*7/16)
    # axs['circle1'].set_aspect('equal')
    # tpc_fig.set_aspect('equal')

    tpc_fig = axs['TPC figure']
    plt.figtext(0.31, 0.9, f'4 random {imsize}$^2$ images with $E[\Phi]$ = {vf} and circle diameter = {circle_size*2}', ha='center', va='center', fontsize=12)

    plt.suptitle(f'Visual presentation of the proof of Theorem 2 in the simple case of random circles.')
    for i in range(1,5):
        circle_im = make_circles_2D(imsize, circle_size, vf)
        axs[f'circle{i}'].imshow(circle_im, cmap='gray')
        circle_pf = np.round(circle_im.mean(),3)
        axs[f'circle{i}'].set_ylabel(f'$\Phi(\omega_{i})={circle_pf}$')
        axs[f'circle{i}'].set_xlabel(f'$\omega_{i}$')
        axs[f'circle{i}'].set_xticks([])
        axs[f'circle{i}'].set_yticks([])
    
    tpc_results, vfs, vf_squares, dist_len = tpcs_radius(make_circles_tpc, run_tests, args=args)
    dist_len = np.array(dist_len)
    mean_tpc = np.mean(tpc_results, axis=0)
    tpc_fig.plot(mean_tpc, label='Mean TPC')
    len_tpc = len(tpc_results[0])   
    vf_squared = np.mean(vf_squares)
    label_vf_squared = f'$E[\Phi^2]$ = {np.round(vf_squared, 4)}'
    tpc_fig.plot([vf_squared]*len_tpc, label=label_vf_squared)
    # print(f'vf squared = {np.round(vf_squared, 7)}')

    true_vf_squared = np.mean(vfs)**2   
    label_true_vf_squared = f'$E[\Phi]^2$ = {np.round(true_vf_squared, 4)}'
    tpc_fig.plot([true_vf_squared]*len_tpc, label=label_true_vf_squared)
    # print(f'true vf squared = {np.round(true_vf_squared, 7)}')
    
    tpc_fig.axvline(x=np.where(dist_len==circle_size*2)[0][0], color='black', linestyle='--', label='Circle Diameter')

    len_tpc = len(tpc_results[0])
    fill_1 = tpc_fig.fill_between(np.arange(len_tpc), mean_tpc,[vf_squared]*len_tpc, alpha=0.2, 
                              where=mean_tpc>=vf_squared,label = f'Area $A_1$')
    fill_2 = tpc_fig.fill_between(np.arange(len_tpc), [vf_squared]*len_tpc,mean_tpc, alpha=0.2, 
                              where=np.logical_and(dist_len<=circle_size*2, np.array(mean_tpc<vf_squared)),label = f'Area $A_2$')
    fill_3 = tpc_fig.fill_between(np.arange(len_tpc),[vf_squared]*len_tpc, mean_tpc, alpha=0.2, 
                              where=dist_len>=circle_size*2,label = f'Area B')
    text_jump = 0.017
    tpc_fig.text(3, vf/2, '$A_1$', fontsize=12)
    tpc_fig.text(16.8, (vf_squared+true_vf_squared)/2, '$A_2$', fontsize=12)
    tpc_fig.text(40, (vf_squared+true_vf_squared)/2 - 0.0005, 'B', fontsize=12)
    # tpc_fig.text(10, 0.079, '$\Phi$ calculates phase fraction.', fontsize=12)
    # tpc_fig.text(22, 0.068, 'The variance of $\Phi$ is:', fontsize=12)
    tpc_fig.text(22, vf_squared+text_jump*7, r'How the model predicts the variance of $\Phi$:', fontsize=12)
    tpc_fig.text(22, vf_squared+text_jump*6, '$Var[\Phi]=E[\Phi^2]-E[\Phi]^2$', fontsize=12)
    tpc_fig.text(22, vf_squared+text_jump*5, r"$Var[\Phi]=\frac{1}{C_{40}}\cdot B$ (Normalization of B's width)", fontsize=12)
    tpc_fig.text(22, vf_squared+text_jump*4, r'$\sum_r{(E[T_r]-E[\Phi^2])}=0$, So', fontsize=12)
    tpc_fig.text(22, vf_squared+text_jump*3, r'$A_1-A_2=B$', fontsize=12)
    tpc_fig.text(22, vf_squared+text_jump*2, 'Which results in:', fontsize=12)
    tpc_fig.text(22, vf_squared+text_jump*1, r'$Var[\Phi]=\frac{1}{C_{40}}\cdot (A_1-A_2)=E[\Psi]$', fontsize=12)

    tpc_fig.set_title(f'Mean TPC of {run_tests} of these random {imsize}$^2$ circle images')
    
    tpc_fig.set_ylim(true_vf_squared-0.005, vf+0.01)
    dist_ticks = list(dist_len[:-5:5]) + [dist_len[-1]]
    x_ticks_labels = list(np.arange(0, len_tpc-5, 5)) + [len_tpc-1]
    tpc_fig.set_xticks(x_ticks_labels, dist_ticks)
    tpc_fig.set_xlabel('TPC distance r')
    tpc_fig.set_ylabel(f'Mean TPC $E[T_r]$', labelpad=-20)
    # tpc_fig.set_yscale('log')
    tpc_fig.set_yticks([vf**2, np.round(vf_squared, 4), vf], [vf**2, np.round(vf_squared, 4), vf])
    tpc_fig.legend()
    plt.savefig(f'tpc_results/circles_tpc_visual_proof.pdf', format='pdf')

    

    
