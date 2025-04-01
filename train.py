import sys
sys.path.append('../')
import numpy as np
import tensorflow as tf
from models import *
from data import *

import pdb
import argparse
print(tf.__version__)
print('It should be >= 2.0.0.')

import time
import os


PATCH_SIZE = 1024
REDUCE_RATIO = 4
RESOLUTION = int(PATCH_SIZE/REDUCE_RATIO)
input_shape = [RESOLUTION,RESOLUTION,1]

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_DIR", type=str, help="Path to original dataset (not filtered")
parser.add_argument("--train_set", default="set1", type=str, help="Name of training set")
parser.add_argument("--input_dir", type=str, help="Where are the input npy files (preprocessed images)")
parser.add_argument("--count_only", type=int, help="True/False ; Counting loss only or joint loss")
parser.add_argument("--alpha", type=float, help="Weight for reconstruction loss")
parser.add_argument("--beta", type=float, help="Weight for counting loss")
parser.add_argument("--model_weights_path", type=str, help="Where to store weights")
parser.add_argument("--exp_name", default="debug", type=str, help="Experiment name")
parser.add_argument("--exp_dir", default="debug", type=str, help="Experiment directory")
parser.add_argument("--Explicit_backpropagation_mode", default="Minus_one", choices=["minus_one", "Puissance_N"] ,type=str, help="Minus one or Puissance N")
parser.add_argument("--N", default=50, type=int, help="Value of N if Puissance N chosen")
parser.add_argument("--batch_size", default=15, type=int)
parser.add_argument("--n_epochs", default=3, type=int)
parser.add_argument("--best_rec_dir", default=None, type=str, help="Where are stored the h-recontstructions for optimal h")
args = parser.parse_args()

input_npy_dir = args.input_dir
BATCH_SIZE=args.batch_size
bp_mode = args.Explicit_backpropagation_mode
weights_path = args.model_weights_path
count_only= bool(args.count_only)

if __name__ == "__main__":
    
    DATA_DIR = args.DATA_DIR
    exp_name = args.exp_name
    exp_dir = args.exp_dir
    N = args.N
    alpha=args.alpha
    beta=args.beta
    sum_ab = alpha+beta
    alpha= alpha/sum_ab
    beta= beta/sum_ab

    random_seed = 4
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    x_train_dir = os.path.join(DATA_DIR, args.train_set, 'images')
    y_train_dir = os.path.join(DATA_DIR, args.train_set, 'labels')
    x_valid_dir = x_train_dir
    y_valid_dir = y_train_dir

                    
    # Let's look at data we have
    dataset = Dataset(x_train_dir,
                      y_train_dir,
                      input_npy_dir,
                      count_only = count_only)

    
    ids_images_train = os.listdir(x_train_dir)
    TRAIN_VAL_SPLIT = 0.8
    len_train = len(ids_images_train)
    idx_val = int(TRAIN_VAL_SPLIT*len_train)
    BATCH_SIZE = np.minimum(BATCH_SIZE, idx_val)
    
    
    np.random.shuffle(ids_images_train)
    ids_images_val = ids_images_train[idx_val:]
    ids_images_train = ids_images_train[:idx_val]
    np.save(os.path.join(args.exp_dir, 'train_ids.npy'), ids_images_train)
    np.save(os.path.join(args.exp_dir, 'val_ids.npy'), ids_images_val)
    print("train len:{}; val len:{}".format(len(ids_images_train), len(ids_images_val)))
    
    # Dataset for train images
    train_dataset = Dataset(
        x_train_dir, 
        y_train_dir,
        input_npy_dir,
        count_only = count_only,
        best_rec_dir = args.best_rec_dir,
        ids = ids_images_train,
    )
    # Dataset for validation images
    
    valid_dataset = Dataset(
        x_valid_dir, 
        y_valid_dir,
        input_npy_dir,
        count_only = count_only,
        best_rec_dir = args.best_rec_dir,
        ids = ids_images_val,
    )

    train_dataloader = Dataloader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, train = True)
    valid_dataloader = Dataloader(valid_dataset, batch_size=1, shuffle=False, train = False)

    # %%

    EPOCHS = args.n_epochs
    LR = .001

    superModel = H_maxima_model(input_shape = input_shape, back_prop_mode = bp_mode, N=N, count_only=count_only)
    fullModel = superModel.nn
    partialModel = superModel.nn_h
    #fullModel.summary()

    #Callback definition
    CBs = [
        tf.keras.callbacks.ModelCheckpoint(weights_path + '/best_model_{}.h5'.format(exp_name),
                                           monitor='val_loss',
                                           verbose=1 ,
                                           save_weights_only=True,
                                           save_best_only=True,
                                           mode='min', period=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.5,
                                             patience=200,
                                             min_lr=0.0001),
        tf.keras.callbacks.CSVLogger('./{}/losses.csv'.format(exp_dir)),
        # tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format(exp_name),
        #                                histogram_freq=0,
        #                                batch_size=BATCH_SIZE,
        #                                write_graph=True, write_grads=False, write_images=True,
        #                                embeddings_freq=0,
        #                                embeddings_layer_names=None,
        #                                embeddings_metadata=None, embeddings_data=None,
        #                                update_freq='epoch'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                         patience=300, verbose=1, mode='auto',
                                        baseline=None, restore_best_weights=False)
    ]
    

    if count_only == True:
        loss = 'mae'
        loss_weights = None
    else:
        loss = {'CountOutput': 'mae', 'RecOutput': 'mse'}
        loss_weights = [beta, alpha]
        
    fullModel.compile(loss=loss, loss_weights = loss_weights,
                      optimizer=tf.keras.optimizers.RMSprop(lr=LR,rho=0.9))
    fullModel.summary()

    start_time=time.time()
    fullModel.fit(train_dataloader,
                  epochs=EPOCHS,
                  verbose=1,
                  validation_data=valid_dataloader,
                  callbacks=CBs
    )
    end_time=time.time()
    print('Elapsed time: ', end_time-start_time)
