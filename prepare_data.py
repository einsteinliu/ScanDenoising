import cv2
import os
import numpy as np

train_image_folder = "../train/"
clean_images_folder = "../train_cleaned/"

def prepare_data_set(folder_dirty,folder_clean, size = 28):
    image_files = os.listdir(folder_dirty)
    dirty_images = []
    clean_images = []
    for i in range(0,len(image_files)):
        curr_image = cv2.imread(folder_dirty + image_files[i])
        curr_clean = cv2.imread(folder_clean + image_files[i])
        curr_shape = curr_image.shape
        num_w = curr_shape[0]//size
        num_h = curr_shape[1]//size
        for x in range(num_w):
            for y in range(num_h):
                dirty_images.append(curr_image[x*size:x*size+28,y*size:y*size+28,0])
                clean_images.append(curr_clean[x*size:x*size+28,y*size:y*size+28,0])
    return {"dirty":dirty_images,"clean":clean_images}

def reconstruct_image(w,h,clean_patches,dirty_patches):
    size = clean_patches[0].shape[0]
    reconstructed_image = np.zeros((w*size,h*size))
    reconstructed_image_dirty = np.zeros((w*size,h*size))
    for x in range(w):
        for y in range(h):
            reconstructed_image[x*size:x*size+28,y*size:y*size+28] = clean_patches[y+x*h][:,:,0]
            reconstructed_image_dirty[x*size:x*size+28,y*size:y*size+28] = dirty_patches[y+x*h]
    return 1.0-np.multiply(1.0-reconstructed_image,1.0-reconstructed_image_dirty),reconstructed_image_dirty

def prepare_test(input_image,size = 28):
    patches = []
    whole_image = cv2.imread(input_image)
    curr_shape = whole_image.shape
    num_w = curr_shape[0]//size
    num_h = curr_shape[1]//size
    for x in range(num_w):
        for y in range(num_h):
            patches.append(whole_image[x*size:x*size+28,y*size:y*size+28,0])
    patches = np.float32(np.asarray(patches))/255.0
    return {"w":num_w,"h":num_h,"patches":patches}