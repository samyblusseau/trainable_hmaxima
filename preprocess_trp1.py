import sys
sys.path.append('../')
import morpholayers.layers as ml
import tensorflow as tf
import numpy as np
import pdb
import argparse
import os
from skimage.morphology import square, dilation, label
from skimage.measure import regionprops
from skimage.transform import resize
from skimage import io
import time
import json

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

parser = argparse.ArgumentParser()
parser.add_argument("--set_name", default="set1", type=str, help="Set of images to preprocess.")
args = parser.parse_args()


original_data_dir='./trp1/database_melanocytes_trp1/'
set_name=args.set_name
images_dir=os.path.join(original_data_dir, set_name, 'images')
labels_dir=os.path.join(original_data_dir, set_name, 'labels')

input_np_dir=os.path.join('trp1','input_np', set_name, 'preprocessed')
resized_gray_np_dir=os.path.join('trp1','input_np', set_name, 'resized_gray')
best_h_dir=os.path.join('trp1','best_h')
best_im_dir=os.path.join('trp1','output_np', set_name)
if not os.path.exists(input_np_dir):
    os.makedirs(input_np_dir)
if not os.path.exists(resized_gray_np_dir):
    os.makedirs(resized_gray_np_dir)
if not os.path.exists(best_h_dir):
    os.makedirs(best_h_dir)
if not os.path.exists(best_im_dir):
    os.makedirs(best_im_dir)

sz = 3
b = square(sz) # structuring element
b = tf.expand_dims(tf.convert_to_tensor(b, dtype=tf.float32), axis=-1)

best_h_dict = dict()
L = os.listdir(images_dir)
for fname in L:
    im_id = fname.split('.')[0]
    imLabels = io.imread(os.path.join(labels_dir,im_id+'.png'))
    ntruth = int(imLabels.sum()/255)

    imColLarge = io.imread(os.path.join(images_dir, im_id+'.png'))
    H,W = imColLarge.shape[0], imColLarge.shape[1]
    imCol = resize(imColLarge,(H//4, W//4))
    im = (255*imCol[:,:,1]).astype('uint8')
    np.save(os.path.join(resized_gray_np_dir, im_id+'.npy'), im)
    
    tmp = np.expand_dims(np.array(im, dtype=np.float32),axis=(0,-1))
    x = ml.opening2d(tmp, b, strides = (1,1), padding = "same")
    imAF = ml.closing2d(x, b, strides = (1,1), padding = "same")
    np.save(os.path.join(input_np_dir, im_id+'_after_opening_closing.npy'), imAF)

    ##########
    # What follows is not necessary for the counting-loss-only experiments
    #########
    tstart = time.time()
    imAF = imAF.numpy()[0,:,:,0]
    max_h=imAF.max()-imAF.min()
    hstar=0
    besterr=500
    h=0
    delta_h=1.
    imRec = imAF
    imRecNew, imMax = hMax(imRec, 0, square(3))
    cc, ncells = label(imMax, return_num = True, connectivity = 2)
    improved=False
    reached=False
    while h <= max_h-delta_h and besterr > 0 and not reached:
        imRecNew, imMax = hMax(imRec, delta_h, square(3))
        cc, ncells = label(imMax, return_num = True, connectivity = 2)
        h=h+delta_h
        curr_err = np.abs(ntruth-ncells)
        if curr_err < besterr:
            besterr = curr_err
            hstar = h
            best_imRec = imRecNew
            improved_once=True
        else:
            if improved_once and curr_err > besterr:
                reached=True
        imRec=imRecNew

    print(im_id, ' best h: ', hstar, ' n_hmax: ', ncells, ' n_cells: ', ntruth)
    best_h_dict[im_id] = hstar
    np.save(os.path.join(best_im_dir, im_id+'.npy'), best_imRec)
    tend = time.time()
    telapsed=tend-tstart
    print('Elapsed time:', telapsed)


jsObj = json.dumps(best_h_dict)
fileObject = open(os.path.join(best_h_dir, 'best_h_opening_closing_{}.json'.format(set_name)), 'w')
fileObject.write(jsObj)  
fileObject.close()
