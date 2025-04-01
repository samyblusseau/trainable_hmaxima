import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
from models import *
import pdb
import argparse
print(tf.__version__)
print('It should be >= 2.0.0.')
import os
from skimage.morphology import dilation, square, opening, closing, label
from skimage.measure import regionprops
import pandas as pd
from skimage import io
import time

##########

# Geodesic dilation
def geoDil(imMark, imMask, se):
    imDil = dilation(imMark, se)
    imRes = np.minimum(imDil, imMask)
    return imRes

# Geodesic reconstruction (iterated geodesic dilations until idempotence)
def geoRec(imMark, imMask, se):
    conv = ((imMask-imMark).max()==0)
    imMk = imMark
    while not conv:
        imAux = geoDil(imMk, imMask, se)
        conv = ((imAux-imMk).max()==0)
        imMk = imAux
    return imMk

# h-reconstruction : geoRec(im-h, im)
def hRec(im, h, se):
    im_h = im.astype('float')-h
    imRec = geoRec(im_h, im, se)
    return imRec

# The h-maxima are the regional maxima of the h-reconstruction
def hMax(im, h, se):
    imRec = hRec(im, h, se)
    imRec2 = hRec(imRec, 1e-5, se)
    idxs = np.where(imRec2 < imRec)
    imMax = np.zeros(imRec2.shape)
    npix_max = len(idxs[0]) 
    npix_tot = im.shape[0]*im.shape[1]
    if npix_max < npix_tot:
        imMax[idxs] = 1.0
    return imRec, imMax


def exactCount(im, h):
    imRec, imMax = hMax(im,h,square(3))
    cc, ncells = label(imMax, return_num = True, connectivity = 2)
    return ncells


##########


PATCH_SIZE = 1024
REDUCE_RATIO = 4
RESOLUTION = int(PATCH_SIZE/REDUCE_RATIO)
input_shape = [RESOLUTION,RESOLUTION,1]


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="Where the input npy files (preprocessed images) are")
parser.add_argument("--input_list", type=str, default=None, help="List of images to apply the model to")
parser.add_argument("--count_only", type=int, help="True/False ; Counting loss only or joint loss")
parser.add_argument("--exp_name", default="debug", type=str, help="experiment name")
parser.add_argument("--model_weights_path", type=str, help="Where the model weights are stored")
parser.add_argument("--res_dir", default="", type=str, help="Where to store the results")
args = parser.parse_args()

count_only= bool(args.count_only)

if __name__ == "__main__":

    superModel = H_maxima_model(input_shape = input_shape, count_only=count_only)
    fullModel, partialModel = superModel.get_simple_model()
    fullModel.load_weights(os.path.join(args.model_weights_path,'best_model_{}.h5'.format(args.exp_name)))
    partialWeights=fullModel.get_layer('h_extrema_denoising_block2').get_weights()
    partialModel.get_layer('h_extrema_denoising_block2').set_weights(partialWeights)
    fullModel.summary()

    im_ids=[]
    h_estim=[]
    ncells_estim=[]
    if args.input_list == None:
        Lraw = os.listdir(args.input_dir)
        L=[]
        for k in range(len(Lraw)):
            imname=Lraw[k]
            tab = imname.split('_after')
            if len(tab) > 1 and not tab[0] in L:
                L.append(tab[0])
    else:
        L=np.load(args.input_list)

    for k in range(len(L)):
        imname = L[k]
        if args.input_list == None:
            im_id = imname
        else:
            im_id = imname.split('.')[0]
        input_im=np.load(args.input_dir+im_id+'_after_opening_closing.npy')
        if count_only:
            n_out = fullModel.predict(input_im)[0][0]
        else:
            output=fullModel.predict(input_im)
            n_out = output[0][0,:,:,0][0,0]
            im_out = output[1][0,:,:,0]
        h_out = partialModel.predict(input_im)[0][0]
        im_ids.append(im_id)
        h_estim.append(h_out)
        ncells_estim.append(n_out)
        imRec, imMax = hMax(input_im[0,:,:,0], h_out, square(3))
        cc, ncells = label(imMax, return_num = True, connectivity = 2) # ncells should be = to n_out
        props = regionprops(cc)
        imDetec = np.zeros(imMax.shape)
        for k in range(len(props)):
            xc = int(props[k].centroid[0])
            yc = int(props[k].centroid[1])
            imDetec[xc, yc] = 255
        imDetec = dilation(imDetec, square(3))
        io.imsave(os.path.join(args.res_dir,im_id+'_detections.png'), imDetec)
        if not count_only:
            io.imsave(os.path.join(args.res_dir,im_id+'_reconstruction.png'), im_out)
            np.save(os.path.join(args.res_dir,im_id+'_reconstruction.npy'), im_out)

    d = {'image_id':im_ids, 'estim_h':h_estim, 'estim_count':ncells_estim}
    df = pd.DataFrame(data=d)
    df.to_csv(os.path.join(args.res_dir,'results.csv'))
