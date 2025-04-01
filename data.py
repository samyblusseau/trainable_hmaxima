import numpy as np
import os
from skimage import io
from skimage.transform import resize
from skimage.measure import block_reduce
from skimage.morphology import label
import keras
import pdb

# classes for data loading and preprocessing
class Dataset:
    """
    """
    def __init__(
            self, 
            images_dir, 
            masks_dir,
            npy_dir,
            count_only = True,
            best_rec_dir = None,# should not be None if count_only==False
            ids = None,
            classes=None, 
            augmentation=None, 
            preprocessing=None,
        
    ):  
        if ids:
            self.ids = ids
        else:
            self.ids = os.listdir(images_dir)

        self.count_only = count_only
        self.npy_dir = npy_dir
        
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        self.iminput_cache_dict = {}
        self.ntruth_cache_dict = {}
        if not count_only:
            self.best_rec_dir = best_rec_dir
            self.rec_cache_dict = {}


    
    def __getitem__(self, i):

        # read data
        image_name = self.images_fps[i].split("/")[-1].split(".")[0]

        #  Loading preprocessed input image
        if not image_name in self.iminput_cache_dict.keys():
            input_np_path = self.npy_dir
            image = np.load(input_np_path + "/" + image_name + "_after_opening_closing.npy")[0,:,:,0]
            self.iminput_cache_dict[image_name] = image
        else:
            image = self.iminput_cache_dict[image_name]

        #     Loading label image to compute true count
        if not image_name in self.ntruth_cache_dict.keys():
            mask = io.imread(self.masks_fps[i])
            mask[mask>0]=255
            cc, ntruth = label(mask, return_num = True, connectivity = 2)
            self.ntruth_cache_dict[image_name] = ntruth
        else:
            ntruth = self.ntruth_cache_dict[image_name]

        item_dict=dict()
        item_dict['input_image']=image
        item_dict['true_count']=ntruth

        if not self.count_only:
            if not image_name in self.rec_cache_dict.keys():
                npy_path_to_load = self.best_rec_dir + "/" + image_name + ".npy"
                rec_ = np.load(npy_path_to_load)
                self.rec_cache_dict[image_name] = rec_
            else:
                rec_ = self.rec_cache_dict[image_name]
            item_dict['best_rec']=rec_

        return item_dict

        
    def __len__(self):
        return len(self.ids)



class Dataloader(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, train=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))
        self.train = train
        self.count_only = dataset.count_only

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            item_j = self.dataset[j]
            image=item_j['input_image']
            ntruth=item_j['true_count']
            if not self.count_only:
                rec_=item_j['best_rec']
                data.append(list((image, ntruth, rec_)))
            else:
                data.append(list((image, ntruth)))
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        if self.train:
            
            if int(np.random.randint(2, size=1) % 2): # enlever int() et size=1
                batch[0] = np.flip(batch[0], 2)
                if not self.count_only:
                    batch[-1] = np.flip(batch[-1], 2)
                
                
            if int(np.random.randint(2, size=1) % 2):                
                batch[0] = np.flip(batch[0], 1)
                if not self.count_only:
                    batch[-1] = np.flip(batch[-1], 1)
                
            if int(np.random.randint(2, size=1) % 2):
                rotate_num =  np.random.randint(1,4)
                batch[0] = np.rot90(batch[0], rotate_num, (1,2))
                if not self.count_only:
                    batch[-1] = np.rot90(batch[-1], rotate_num, (1,2))
        else:
            pass


        if self.count_only:
            #batch = batch[0], batch[1]
            batch_out=batch[0], batch[1]
        else:
            batch[-1] = np.expand_dims(batch[-1],axis=-1)
            batch[1] = np.expand_dims(np.expand_dims(batch[1],axis=-1),axis=-1)
            batch[1] = np.expand_dims(batch[1],axis=-1)
            #batch_out = batch[0], [batch[1], batch[2]]
            labels = {'CountOutput': batch[1], 'RecOutput': batch[2]}
            batch_out = batch[0], labels
            
        return batch_out
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
