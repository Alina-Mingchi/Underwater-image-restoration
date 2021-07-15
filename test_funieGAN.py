"""
    
Code based on Jahid
Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)

"""

import cv2
import os
import time
import ntpath
import numpy as np
from keras.models import model_from_json
# local libs
from utils.data_utils import getPaths, read_and_resize, preprocess, deprocess
from utils.data_utils import get_local_test_data

# testing data
data_dir = "data/test/random/"

test_paths = getPaths(data_dir)
print ("{0} test images are loaded".format(len(test_paths)))

# create dir for validation data
samples_dir = "data/output/"
if not os.path.exists(samples_dir): os.makedirs(samples_dir)

# %%
# test funie-gan
checkpoint_dir  = 'saved_models/gen_p/'
model_name_by_epoch = "model_15320_" 

# check existence of model
model_h5 = checkpoint_dir + model_name_by_epoch + ".h5"  
model_json = checkpoint_dir + model_name_by_epoch + ".json"
assert (os.path.exists(model_h5) and os.path.exists(model_json))

# load model
with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
funie_gan_generator = model_from_json(loaded_model_json)
funie_gan_generator.load_weights(model_h5)
print("Loaded data and model")

# testing loop
times = []; s = time.time()
for img_path in test_paths:
    # prepare data
    img_name = ntpath.basename(img_path).split('.')[0]
    im = read_and_resize(img_path, (256, 256))
    im = preprocess(im)
    im = np.expand_dims(im, axis=0)
    # im is of shape [1,256,256,3]
    
    # generate enhanced image
    s = time.time()
    gen = funie_gan_generator.predict(im)
    gen = deprocess(gen) # Rescale to 0-1
    tot = time.time()-s
    times.append(tot)
    
    # save sample images
    cv2.imwrite(samples_dir+img_name+'_real.png', im[0])
    cv2.imwrite(samples_dir+img_name+'_gen.png', gen[0])



num_test = len(test_paths)
if (num_test==0):
    print ("Found no images for test")
else:
    print ("Total images: {0}".format(num_test))
    # accumulate frame processing times (without bootstrap)
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:]) 
    print ("Time taken: {0} sec at {1} fps".format(Ttime, 1./Mtime))
    print("Saved generated images in in {0}".format(samples_dir))

