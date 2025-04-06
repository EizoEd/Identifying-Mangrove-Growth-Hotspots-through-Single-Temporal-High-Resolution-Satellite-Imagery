import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import geometry_mask
import geopandas as gpd
import os
import numpy as np
from scipy.ndimage import uniform_filter
from time import time
from skimage.transform import resize
import xgboost as xgb
from scipy.ndimage import gaussian_filter
from math import sqrt
import pickle
import gc
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import json
import joblib

# Modify the model to get output from the last convolutional layer
# We'll return an IntermediateLayerGetter containing feature maps
from scipy.interpolate import interp1d
from joblib import dump

# Generate random indices and split
def clip_by_percentage(img, percentage=97, save_rate=0.1):
    # Get the percentile of growth rate at 'percentage', retain 'save_rate' of values below the percentile,
    # and keep all values above the percentile
    quantile = np.percentile(img, percentage)
    print(f"{percentage}% percentile is: {quantile}")
    # Indices greater than the percentile
    greater_indices = np.where(img > quantile)[0]

    # Indices of data less than or equal to the percentile
    less_equal_indices = np.where(img <= quantile)[0]
    
    # Randomly retain data less than or equal to the percentile
    random_selection_mask = np.random.rand(len(less_equal_indices)) < save_rate
    randomly_kept_indices = less_equal_indices[random_selection_mask]
    # Merge indices
    final_indices = np.concatenate((greater_indices, randomly_kept_indices))
    print(len(greater_indices), len(randomly_kept_indices))
    return final_indices

def compute_distance_transform(shapefile_path, tif_path, step_size):
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)

    # Read TIF file
    with rasterio.open(tif_path) as src:
        image = src.read(1)  # Read the first band of the image
        transform = src.transform
        width, height = src.width, src.height

    # Create a mask with the same size as the image
    shapes = gdf.geometry.values
    mask = geometry_mask(shapes, transform=transform, invert=True, out_shape=(height, width))

    # Convert boolean mask to uint8 type
    mask = mask.astype(np.uint8) * 255

    # Extract mask boundaries
    edges = cv2.Canny(mask, 100, 200)
    
    # Compute distance transform
    dist_transform = cv2.distanceTransform(1 - (edges // 255), cv2.DIST_L2, 5)

    # Sample values by step size and flatten
    sampled_distances = dist_transform[::step_size, ::step_size]
    # Add a new dimension
    sampled_distances_with_dim = np.expand_dims(sampled_distances, axis=-1)
    
    output_tif_path = tif_path.replace('.tif','_distance.tif')
    # Save as TIF file
    # with rasterio.open(
    #     output_tif_path,
    #     'w',
    #     driver='GTiff',
    #     height=sampled_distances_with_dim.shape[0],
    #     width=sampled_distances_with_dim.shape[1],
    #     count=1,
    #     dtype=sampled_distances_with_dim.dtype,
    #     crs=src.crs,
    #     transform=from_origin(transform[2], transform[5], step_size, step_size)
    # ) as dst:
    #     dst.write(sampled_distances_with_dim[:, :, 0], 1)
    return sampled_distances_with_dim


def expand_and_overlap(image):
    rows, cols = image.shape
    # Create new image array
    new_rows = rows * 2 + 1
    new_cols = cols * 2 + 1
    new_image = np.zeros((new_rows, new_cols), dtype=float)
    
    # Count matrix to record how many times each pixel is accumulated
    count_matrix = np.zeros_like(new_image)
    
    # Expand pixel values to new image
    for i in range(rows):
        for j in range(cols):
            new_i = i * 2
            new_j = j * 2
            # Assign original pixel value to 3x3 block
            new_image[new_i:new_i+3, new_j:new_j+3] += image[i, j]
            count_matrix[new_i:new_i+3, new_j:new_j+3] += 1
    
    # Process overlapping areas by calculating mean
    new_image /= count_matrix
    
    return new_image

seed_num = 200
tree_depth = 7
np.random.seed(seed_num)


# File path list
files = [
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402_ndvi_90m_feature.tif',
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402_ndwi_90m_feature.tif',
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402.tif',
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402_GLCM25_band1.tif',
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402_GLCM25_band2.tif',
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402_GLCM25_band3.tif',
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402_GLCM25_band4.tif'
]
output_filename = [
    f'ZhanjiangGF1B_PMS_20240402/ZhanjiangGF1B_PMS_20240402_ZhanjiangModel_CatBoost_predict.tif'
]


shapefile_paths = [
    'ZhanjiangMangroveDistributionData/mangroveage.shp'
]
# List to store each image data
images = []
predict_imgs = []
trans = [] 
origion_shape = []
concatenated_shape = []
out_meta = []
filter_size = 45
stride = 15
shapefile_idx = 0
sigma = 1

for idx, file in enumerate(files):
    #print(file)
    with rasterio.open(file) as src:
        # Read image data and add to list
        img = src.read()
        img = np.moveaxis(img, 0, -1) # Move band dimension to last axis
        
        if idx%7 <= 1:
            images.append(img)
            continue
        else:
            # Create new array to store filtered results
            filtered_shape = [img.shape[0]//stride + 1, img.shape[1]//stride + 1]
            if img.shape[0]%stride == 0:filtered_shape[0] -= 1
            if img.shape[1]%stride == 0:filtered_shape[1] -= 1
            img_filtered = np.zeros((filtered_shape[0], filtered_shape[1], img.shape[2]), dtype=img.dtype)
            for i in range(img.shape[-1]):
                temp = uniform_filter(img[:, :, i], size=filter_size, mode='reflect')[::stride, ::stride]
                img_filtered[:, :, i] = temp
                
            images.append(img_filtered)
            
    flag = idx + 1
    if flag % 7 == 0:
        # Add pixel-to-boundary distance to the final data
        distance = compute_distance_transform(shapefile_paths[shapefile_idx], file, stride)
        invalid_idx = np.array(np.where(distance >500))
        images.append(distance)
        shapefile_idx += 1
        concatenated_image = np.concatenate(images, axis=-1) # [width, height, N]
        width, height, n_bands = img.shape
        n_bands = concatenated_image.shape[2]
        origion_shape.append([width,height])
        concatenated_shape.append([concatenated_image.shape[0], concatenated_image.shape[1]])
        images = []
        trans.append(gaussian_filter(concatenated_image, sigma=(sigma, sigma, 0)))
        out_meta.append([src.meta.copy(), src.transform])
        
for idx, imgs in enumerate(trans):
    # First split then normalize
    min_vals = imgs.min(axis=0)[:,64:]
    max_vals = imgs.max(axis=0)[:,64:]
    
    # Perform min-max normalization band by band
    imgs[:,:,64:] = (imgs[:,:,64:] - min_vals) / (max_vals - min_vals+0.0001)
    imgs = imgs.reshape(-1, n_bands)
    trans[idx] = imgs
    
transformed_img = np.concatenate(trans, axis=0)
del concatenated_image, images, img, min_vals, max_vals
gc.collect()


# Load corresponding model
model = joblib.load('zhanjiang_zhanjiang_CatBoost_depth10_lr0.1_100.joblib')

for idx, imgs in enumerate(trans):
    predictions = model.predict(imgs)
    p_width, p_height = origion_shape[idx]
    predictions = predictions.reshape(concatenated_shape[idx][0], concatenated_shape[idx][1])
    # Set obviously wrong areas (too far from boundary) to 0 as they can't have mangrove growth
    #predictions[invalid_idx] = 0
    output_prediction = expand_and_overlap(predictions)
    width, height = output_prediction.shape
    
    # Here 90m is -1, 150m is -2. Because after reshaping at 150m, the image becomes original dimensions *3+2
    scale_factor_x = p_width/(width-1)
    scale_factor_y = p_height/(height-1)
    original_meta, original_transform = out_meta[idx]
    new_transform = Affine(original_transform.a*scale_factor_x, original_transform.b, original_transform.c,
                                   original_transform.d, original_transform.e*scale_factor_y, original_transform.f)
    original_meta.update({
        'count':1,
        'transform':new_transform,
        'width': height,
        'height': width
    })
    print("Completed prediction for entire image, now starting data cleanup and saving!")

    with rasterio.open(output_filename[idx], 'w', **original_meta) as dest:
        dest.write(output_prediction, 1)
        print(f'Regression prediction results have been written to {output_filename[idx]}')

del model
gc.collect()