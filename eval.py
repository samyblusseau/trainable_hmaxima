import sys
sys.path.append('../')
import numpy as np
import pdb
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
from skimage import io
from skimage.morphology import label
import csv

parser = argparse.ArgumentParser()
parser.add_argument("--res_dir", default="", type=str, help="where results are strored")
parser.add_argument("--gt_count_dir", default="", type=str, help="where the true numbers of cells are.")
parser.add_argument("--gt_h_path", default="", type=str, help="where the true h values are.")
parser.add_argument("--gt_rec_dir", default="", type=str, help="where the true reconstructions are.")
parser.add_argument("--count_only", type=int, help="True/False ; Counting loss only or joint loss")
args = parser.parse_args()

count_only= bool(args.count_only)

if __name__ == "__main__":
    
    df = pd.read_csv(os.path.join(args.res_dir,'results.csv'))
    with open(args.gt_h_path, 'r') as f:
        true_h_dict = json.load(f)
    im_ids = []
    true_counts = []
    estim_counts = []
    true_h_vals = []
    estim_h_vals = []
    if not count_only:
        rec_errors = []
    for k in range(len(df['image_id'])):
        im_id = df['image_id'][k]
        im_ids.append(im_id)
        imLab = io.imread(os.path.join(args.gt_count_dir, im_id+'.png'))
        imLab[imLab>0]=255
        cc, true_count = label(imLab, return_num = True, connectivity = 2)
        estim_count= df['estim_count'][k]
        true_h= true_h_dict[im_id]
        estim_h= df['estim_h'][k]
        true_counts.append(true_count)
        estim_counts.append(estim_count)
        true_h_vals.append(true_h)
        estim_h_vals.append(estim_h)
        if not count_only:
            true_rec = np.load(os.path.join(args.gt_rec_dir, im_id+'.npy'))
            estim_rec = np.load(os.path.join(args.res_dir, im_id+'_reconstruction.npy'))
            mse = np.mean((true_rec-estim_rec)**2)
            rec_errors.append(mse)
        
    true_h_vals = np.array(true_h_vals).astype(np.float32)
    estim_h_vals = np.array(estim_h_vals).astype(np.float32)
    true_counts = np.array(true_counts, dtype=int)
    estim_counts = np.array(estim_counts, dtype=int)
    if count_only:
        d = {'image_id':im_ids, 'true_count':true_counts, 'estim_count':estim_counts, 'true_h':true_h_vals, 'etim_h': estim_h_vals}
    else:
        rec_errors = np.array(rec_errors)
        d = {'image_id':im_ids, 'true_count':true_counts, 'estim_count':estim_counts, 'true_h':true_h_vals, 'etim_h': estim_h_vals, 'mse_rec': rec_errors}

    df_eval = pd.DataFrame(data=d)

    signed_error = true_counts-estim_counts
    abs_error = np.abs(signed_error)
    relative_abs_error = abs_error[true_counts>0]/true_counts[true_counts>0]
    average_relative_error = np.mean(relative_abs_error)
    total_relative_error = abs_error.sum()/np.sum(true_counts)
    percentage_error = signed_error[true_counts>0]/true_counts[true_counts>0]
    mean_percentage_error=np.mean(percentage_error)
    mean_abs_error = np.mean(abs_error)
    if not count_only:
        mean_mse = np.mean(rec_errors)

    df_eval.to_csv(os.path.join(args.res_dir,'eval.csv'))
    with open(os.path.join(args.res_dir,'summary_eval.csv'), 'w', newline='') as csvfile:
        csvw = csv.writer(csvfile, delimiter='\t')
        csvw.writerow(['Average relative error', average_relative_error])
        csvw.writerow(['Total relative error', total_relative_error])
        csvw.writerow(['Mean percentage error', mean_percentage_error])
        csvw.writerow(['Mean absolute error', mean_abs_error])
        if not count_only:
            csvw.writerow(['Mean reconstruction error', mean_mse])


    hmin=np.minimum(np.min(true_h_vals), np.min(estim_h_vals))-5
    hmin= np.maximum(hmin, 0)
    hmax=np.maximum(np.max(true_h_vals), np.max(estim_h_vals))+5
    hrange=np.arange(int(hmin), int(hmax))
    plt.plot(true_h_vals, estim_h_vals, 'o')
    plt.plot(hrange, hrange, color='red')
    plt.xlabel('True optimal $h$')
    plt.ylabel('Estimated optimal $h$')
    plt.xlim(hmin, hmax)
    plt.ylim(hmin, hmax)
    plt.axis('square')
    plt.savefig(os.path.join(args.res_dir,'plot_h.pdf'))

    nmin=np.minimum(np.min(true_counts), np.min(estim_counts))-5
    nmin=np.maximum(nmin, 0)
    nmax=np.maximum(np.max(true_counts), np.max(estim_counts))+5
    nrange=np.arange(int(nmin), int(nmax))
    plt.figure()
    plt.plot(true_counts, estim_counts, 'o')
    plt.plot(nrange, nrange, color='red')
    plt.xlabel('True count')
    plt.ylabel('Estimated count')
    plt.xlim(nmin, nmax)
    plt.ylim(nmin, nmax)
    plt.axis('square')
    plt.savefig(os.path.join(args.res_dir,'plot_count.pdf'))
    plt.close()
