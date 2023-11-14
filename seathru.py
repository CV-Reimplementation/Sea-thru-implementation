import collections
import sys
import argparse
import numpy as np
import sklearn as sk
import scipy as sp
import scipy.optimize
import scipy.stats
import math
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from skimage import exposure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma
from skimage.morphology import closing, opening, erosion, dilation, disk, diamond, square
from tqdm import tqdm
import os


matplotlib.use('TkAgg')

'''
Finds points for which to estimate backscatter
by partitioning the image into different depth
ranges and taking the darkest RGB triplets 
from that set as estimations of the backscatter
'''
def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=2)
    points_r = []
    points_g = []
    points_b = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
        points_r.extend([(z, p[0]) for n, p, z in points])
        points_g.extend([(z, p[1]) for n, p, z in points])
        points_b.extend([(z, p[2]) for n, p, z in points])
    return np.array(points_r), np.array(points_g), np.array(points_b)

'''
Estimates coefficients for the backscatter curve
based on the backscatter point values and their depths
'''
def find_backscatter_values(B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    z_max, z_min = np.max(depths), np.min(depths)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
        val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
        return val
    def loss(B_inf, beta_B, J_prime, beta_D_prime):
        val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
        return val
    bounds_lower = [0,0,0,0]
    bounds_upper = [1,5,1,5]
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=estimate,
                xdata=B_depths,
                ydata=B_vals,
                p0=np.random.random(4) * bounds_upper,
                bounds=(bounds_lower, bounds_upper),
            )
            l = loss(*optp)
            if l < best_loss:
                best_loss = l
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)
    if best_loss > max_mean_loss:
        print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(B_depths, B_vals)
        BD = (slope * depths) + intercept
        return BD, np.array([slope, intercept])
    return estimate(depths, *coefs), coefs

'''
Estimate illumination map from local color space averaging
'''
def estimate_illumination(img, B, neighborhood_map, num_neighborhoods, p=0.5, f=2.0, max_iters=100, tol=1E-5):
    D = img - B
    avg_cs = np.zeros_like(img)
    avg_cs_prime = np.copy(avg_cs)
    sizes = np.zeros(num_neighborhoods)
    locs_list = [None] * num_neighborhoods
    for label in range(1, num_neighborhoods + 1):
        locs_list[label - 1] = np.where(neighborhood_map == label)
        sizes[label - 1] = np.size(locs_list[label - 1][0])
    for _ in range(max_iters):
        for label in range(1, num_neighborhoods + 1):
            locs = locs_list[label - 1]
            size = sizes[label - 1] - 1
            avg_cs_prime[locs] = (1 / size) * (np.sum(avg_cs[locs]) - avg_cs[locs])
        new_avg_cs = (D * p) + (avg_cs_prime * (1 - p))
        if(np.max(np.abs(avg_cs - new_avg_cs)) < tol):
            break
        avg_cs = new_avg_cs
    return f * denoise_bilateral(np.maximum(0, avg_cs))

'''
Estimate values for beta_D
'''
def estimate_wideband_attentuation(depths, illum, radius = 6, max_val = 10.0):
    eps = 1E-8
    BD = np.minimum(max_val, -np.log(illum + eps) / (np.maximum(0, depths) + eps))
    mask = np.where(np.logical_and(depths > eps, illum > eps), 1, 0)
    refined_attenuations = denoise_bilateral(closing(np.maximum(0, BD * mask), disk(radius)))
    return refined_attenuations, []

'''
Calculate the values of beta_D for an image from the depths, illuminations, and constants
'''
def calculate_beta_D(depths, a, b, c, d):
    if a is None:
        return depths
    return (a * np.exp(b * depths)) + (c * np.exp(d * depths))


def filter_data(X, Y, radius_fraction=0.01):
    idxs = np.argsort(X)
    X_s = X[idxs]
    Y_s = Y[idxs]
    x_max, x_min = np.max(X), np.min(X)
    radius = (radius_fraction * (x_max - x_min))
    ds = np.cumsum(X_s - np.roll(X_s, (1,)))
    dX = [X_s[0]]
    dY = [Y_s[0]]
    tempX = []
    tempY = []
    pos = 0
    for i in range(1, ds.shape[0]):
        if ds[i] - ds[pos] >= radius:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
            idxs = np.argsort(tempY)
            med_idx = len(idxs) // 2
            dX.append(tempX[med_idx])
            dY.append(tempY[med_idx])
            pos = i
        else:
            tempX.append(X_s[i])
            tempY.append(Y_s[i])
    return np.array(dX), np.array(dY)

'''
Estimate coefficients for the 2-term exponential
describing the wideband attenuation
'''
def refine_wideband_attentuation(depths, illum, estimation, restarts=10, min_depth_fraction = 0.1, max_mean_loss_fraction=np.inf, l=1.0, radius_fraction=0.01):
    eps = 1E-8
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_fraction * (z_max - z_min))
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    optp = None
    best_loss = np.inf
    locs = np.where(np.logical_and(illum > 0, np.logical_and(depths > min_depth, estimation > eps)))
    def calculate_reconstructed_depths(depths, illum, a, b, c, d):
        eps = 1E-5
        res = -np.log(illum + eps) / (calculate_beta_D(depths, a, b, c, d) + eps)
        return res
    def loss(a, b, c, d):
        return np.mean(np.abs(depths[locs] - calculate_reconstructed_depths(depths[locs], illum[locs], a, b, c, d)))
    dX, dY = filter_data(depths[locs], estimation[locs], radius_fraction)
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=calculate_beta_D,
                xdata=dX,
                ydata=dY,
                p0=np.abs(np.random.random(4)) * np.array([1., -1., 1., -1.]),
                bounds=([0, -100, 0, -100], [100, 0, 100, 0]))
            L = loss(*optp)
            if L < best_loss:
                best_loss = L
                coefs = optp
        except RuntimeError as re:
            continue

    if best_loss > max_mean_loss:
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(depths[locs], estimation[locs])
        BD = (slope * depths + intercept)
        return l * BD, np.array([slope, intercept])
    if coefs is None:
        BD = l * depths
    else:
        BD = l * calculate_beta_D(depths, *coefs)
    return BD, coefs

'''
Reconstruct the scene and globally white balance
based the Gray World Hypothesis
'''
def recover_image(img, depths, B, beta_D, nmap):
    res = (img - B) * np.exp(beta_D * np.expand_dims(depths, axis=2))
    res = np.maximum(0.0, np.minimum(1.0, res))
    res[nmap == 0] = 0
    res = scale(wbalance_no_red_10p(res))
    res[nmap == 0] = img[nmap == 0]
    return res


'''
Reconstruct the scene and globally white balance
'''
def recover_image_S4(img, B, illum, nmap):
    eps = 1E-8
    res = (img - B) / (illum + eps)
    res = np.maximum(0.0, np.minimum(1.0, res))
    res[nmap == 0] = img[nmap == 0]
    return scale(wbalance_no_red_gw(res))


'''
Constructs a neighborhood map from depths and 
epsilon
'''
def construct_neighborhood_map(depths, epsilon=0.05):
    eps = (np.max(depths) - np.min(depths)) * epsilon
    nmap = np.zeros_like(depths).astype(np.int32)
    n_neighborhoods = 1
    while np.any(nmap == 0):
        locs_x, locs_y = np.where(nmap == 0)
        start_index = np.random.randint(0, len(locs_x))
        start_x, start_y = locs_x[start_index], locs_y[start_index]
        q = collections.deque()
        q.append((start_x, start_y))
        while not len(q) == 0:
            x, y = q.pop()
            if np.abs(depths[x, y] - depths[start_x, start_y]) <= eps:
                nmap[x, y] = n_neighborhoods
                if 0 <= x < depths.shape[0] - 1:
                    x2, y2 = x + 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= x < depths.shape[0]:
                    x2, y2 = x - 1, y
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 0 <= y < depths.shape[1] - 1:
                    x2, y2 = x, y + 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
                if 1 <= y < depths.shape[1]:
                    x2, y2 = x, y - 1
                    if nmap[x2, y2] == 0:
                        q.append((x2, y2))
        n_neighborhoods += 1
    zeros_size_arr = sorted(zip(*np.unique(nmap[depths == 0], return_counts=True)), key=lambda x: x[1], reverse=True)
    if len(zeros_size_arr) > 0:
        nmap[nmap == zeros_size_arr[0][0]] = 0 #reset largest background to 0
    return nmap, n_neighborhoods - 1

'''
Finds the closest nonzero label to a location
'''
def find_closest_label(nmap, start_x, start_y):
    mask = np.zeros_like(nmap).astype(np.bool_)
    q = collections.deque()
    q.append((start_x, start_y))
    while not len(q) == 0:
        x, y = q.pop()
        if 0 <= x < nmap.shape[0] and 0 <= y < nmap.shape[1]:
            if nmap[x, y] != 0:
                return nmap[x, y]
            mask[x, y] = True
            if 0 <= x < nmap.shape[0] - 1:
                x2, y2 = x + 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= x < nmap.shape[0]:
                x2, y2 = x - 1, y
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 0 <= y < nmap.shape[1] - 1:
                x2, y2 = x, y + 1
                if not mask[x2, y2]:
                    q.append((x2, y2))
            if 1 <= y < nmap.shape[1]:
                x2, y2 = x, y - 1
                if not mask[x2, y2]:
                    q.append((x2, y2))


'''
Refines the neighborhood map to remove artifacts
'''
def refine_neighborhood_map(nmap, min_size = 10, radius = 3):
    refined_nmap = np.zeros_like(nmap)
    vals, counts = np.unique(nmap, return_counts=True)
    neighborhood_sizes = sorted(zip(vals, counts), key=lambda x: x[1], reverse=True)
    num_labels = 1
    for label, size in neighborhood_sizes:
        if size >= min_size and label != 0:
            refined_nmap[nmap == label] = num_labels
            num_labels += 1
    for label, size in neighborhood_sizes:
        if size < min_size and label != 0:
            for x, y in zip(*np.where(nmap == label)):
                refined_nmap[x, y] = find_closest_label(refined_nmap, x, y)
    refined_nmap = closing(refined_nmap, square(radius))
    return refined_nmap, num_labels - 1


def load_image_and_depth_map(img_fname, depths_fname):
    depths = Image.open(depths_fname)
    img = Image.open(img_fname)
    return np.float32(img) / 255.0, np.array(depths)

'''
White balance with 'grey world' hypothesis
'''
def wbalance_gw(img):
    dr = 1.0 / np.mean(img[:, :, 0])
    dg = 1.0 / np.mean(img[:, :, 1])
    db = 1.0 / np.mean(img[:, :, 2])
    dsum = dr + dg + db
    dr = dr / dsum * 3.
    dg = dg / dsum * 3.
    db = db / dsum * 3.

    img[:, :, 0] *= dr
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img


'''
White balance based on top 10% average values of each channel
'''
def wbalance_10p(img):
    dr = 1.0 / np.mean(np.sort(img[:, :, 0], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dsum = dr + dg + db
    dr = dr / dsum * 3.
    dg = dg / dsum * 3.
    db = db / dsum * 3.

    img[:, :, 0] *= dr
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

'''
White balance based on top 10% average values of blue and green channel
'''
def wbalance_no_red_10p(img):
    dg = 1.0 / np.mean(np.sort(img[:, :, 1], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    db = 1.0 / np.mean(np.sort(img[:, :, 2], axis=None)[int(round(-1 * np.size(img[:, :, 0]) * 0.1)):])
    dsum = dg + db
    dg = dg / dsum * 2.
    db = db / dsum * 2.
    img[:, :, 0] *= (db + dg) / 2
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

'''
White balance with 'grey world' hypothesis
'''
def wbalance_no_red_gw(img):
    dg = 1.0 / np.mean(img[:, :, 1])
    db = 1.0 / np.mean(img[:, :, 2])
    dsum = dg + db
    dg = dg / dsum * 2.
    db = db / dsum * 2.

    img[:, :, 0] *= (db + dg) / 2
    img[:, :, 1] *= dg
    img[:, :, 2] *= db
    return img

def scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def run_pipeline(img, depths, min_depth, p, f, l, spread_data_fraction):

    ptsR, ptsG, ptsB = find_backscatter_estimation_points(img, depths, fraction=0.01, min_depth_percent=min_depth)

    Br, coefsR = find_backscatter_values(ptsR, depths, restarts=25)
    Bg, coefsG = find_backscatter_values(ptsG, depths, restarts=25)
    Bb, coefsB = find_backscatter_values(ptsB, depths, restarts=25)

    nmap, _ = construct_neighborhood_map(depths, 0.1)

    nmap, n = refine_neighborhood_map(nmap, 50)

    illR = estimate_illumination(img[:, :, 0], Br, nmap, n, p=p, max_iters=100, tol=1E-5, f=f)
    illG = estimate_illumination(img[:, :, 1], Bg, nmap, n, p=p, max_iters=100, tol=1E-5, f=f)
    illB = estimate_illumination(img[:, :, 2], Bb, nmap, n, p=p, max_iters=100, tol=1E-5, f=f)
    ill = np.stack([illR, illG, illB], axis=2)

    beta_D_r, _ = estimate_wideband_attentuation(depths, illR)
    refined_beta_D_r, coefsR = refine_wideband_attentuation(depths, illR, beta_D_r, radius_fraction=spread_data_fraction, l=l)
    beta_D_g, _ = estimate_wideband_attentuation(depths, illG)
    refined_beta_D_g, coefsG = refine_wideband_attentuation(depths, illG, beta_D_g, radius_fraction=spread_data_fraction, l=l)
    beta_D_b, _ = estimate_wideband_attentuation(depths, illB)
    refined_beta_D_b, coefsB = refine_wideband_attentuation(depths, illB, beta_D_b, radius_fraction=spread_data_fraction, l=l)

    B = np.stack([Br, Bg, Bb], axis=2)
    beta_D = np.stack([refined_beta_D_r, refined_beta_D_g, refined_beta_D_b], axis=2)
    recovered = recover_image(img, depths, B, beta_D, nmap)

    return recovered

def preprocess_for_monodepth(img_fname, output_fname):
    img = Image.open(img_fname)
    img_adapteq = exposure.equalize_adapthist(np.array(img), clip_limit=0.03)
    Image.fromarray((np.round(img_adapteq * 255.0)).astype(np.uint8)).save(output_fname)

def preprocess_sfm_depth_map(depths, min_depth, max_depth):
    z_min = np.min(depths) + (min_depth * (np.max(depths) - np.min(depths)))
    z_max = np.min(depths) + (max_depth * (np.max(depths) - np.min(depths)))
    if max_depth != 0:
        depths[depths == 0] = z_max
    depths[depths < z_min] = 0
    return depths

def preprocess_monodepth_depth_map(depths, additive_depth, multiply_depth):
    depths = ((depths - np.min(depths)) / (
                np.max(depths) - np.min(depths))).astype(np.float32)
    depths = (multiply_depth * (1.0 - depths)) + additive_depth
    return depths

if __name__ == '__main__':

    monodepth_add_depth = 2.0 # Additive value for monodepth map
    monodepth_multiply_depth = 2.0 # Multiplicative value for monodepth map
    min_depth = 0.1 # Minimum depth value to use in estimations (range 0-1)
    max_depth = 1.0 # Replacement depth percentile value for invalid depths (range 0-1)
    f = 2.0 # f value (controls brightness)
    l = 0.5 # l value (controls balance of attenuation constants)
    p = 0.01 # p value (controls locality of illuminant map)
    spread_data_fraction = 0.01 # Require data to be this fraction of depth range away from each other in attenuation estimations

    preprocess_for_monodepth = False # Preprocess for monodepth depth maps
    monodepth = True # Preprocess for monodepth
    equalize_image = True # Histogram equalization for final output

    img_folder = 'input'
    dep_folder = 'depth'
    
    result = 'result'
    os.makedirs(result, exist_ok=True)


    images = os.listdir(img_folder)

    for idx in tqdm(images):
        rgb = os.path.join(img_folder, idx)
        dep = os.path.join(dep_folder, idx)

        if preprocess_for_monodepth:
            # generate depth map
            preprocess_for_monodepth(rgb, os.path.join(result, idx))

        img, depths = load_image_and_depth_map(rgb, dep)

        if monodepth:
            depths = preprocess_monodepth_depth_map(depths, monodepth_add_depth, monodepth_multiply_depth)
        else:
            depths = preprocess_sfm_depth_map(depths, min_depth, max_depth)
        
        recovered = run_pipeline(img, depths, min_depth, p, f, l, spread_data_fraction)

        if equalize_image:
            recovered = exposure.equalize_adapthist(np.array(recovered), clip_limit=0.03)
            sigma_est = estimate_sigma(recovered, channel_axis=2, average_sigmas=True)
            recovered = denoise_tv_chambolle(recovered, sigma_est, channel_axis=2)
        
        plt.imsave(os.path.join(result, idx), np.clip(recovered, a_min=0, a_max=1))

