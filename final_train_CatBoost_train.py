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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

# Generate random indices and split
# To reduce model fitting bias from experimental data, we filter the data to minimize impact of low-growth areas
def clip_by_percentage(img, percentage=97, save_rate=0.1):
    # Get the percentile value of growth rate
    # Keep save_rate of values below percentile and all values above
    quantile = np.percentile(img, percentage)
    print(f"{percentage}th percentile value: {quantile}")
    # Indices greater than percentile
    greater_indices = np.where(img > quantile)[0]

    # Indices less than or equal to percentile
    less_equal_indices = np.where(img <= quantile)[0]
    
    # Randomly retain data below percentile
    random_selection_mask = np.random.rand(len(less_equal_indices)) < save_rate
    randomly_kept_indices = less_equal_indices[random_selection_mask]
    # Combine indices
    final_indices = np.concatenate((greater_indices, randomly_kept_indices))
    print(f"Count above percentile: {len(greater_indices)}, Count below kept: {len(randomly_kept_indices)}")
    return final_indices

def compute_distance_transform(shapefile_path, tif_path, step_size):
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)

    # Read TIF file
    with rasterio.open(tif_path) as src:
        image = src.read(1)  # Read first band
        transform = src.transform
        width, height = src.width, src.height

    # Create mask matching image dimensions
    shapes = gdf.geometry.values
    mask = geometry_mask(shapes, transform=transform, invert=True, out_shape=(height, width))

    # Convert boolean mask to uint8
    mask = mask.astype(np.uint8) * 255

    # Extract mask boundaries
    edges = cv2.Canny(mask, 100, 200)
    
    # Compute distance transform
    dist_transform = cv2.distanceTransform(1 - (edges // 255), cv2.DIST_L2, 5)

    # Sample values by step size
    sampled_distances = dist_transform[::step_size, ::step_size]
    # Apply Gaussian transformation (mean=0, std=15)
    sigma = 15
    gaussian_transform = np.exp(- (sampled_distances ** 2) / (2 * sigma ** 2))

    # Add new dimension
    gaussian_transform_with_dim = np.expand_dims(gaussian_transform, axis=-1)

    return gaussian_transform_with_dim

# Restore model predictions since model uses 90m window but mangrove product resolution is 30m
def expand_and_overlap(image):
    rows, cols = image.shape
    # Create new image array
    new_rows = rows * 2 + 1
    new_cols = cols * 2 + 1
    new_image = np.zeros((new_rows, new_cols), dtype=float)
    
    # Count matrix for pixel accumulation
    count_matrix = np.zeros_like(new_image)
    
    # Expand pixel values
    for i in range(rows):
        for j in range(cols):
            new_i = i * 2
            new_j = j * 2
            # Assign original pixel to 3x3 block
            new_image[new_i:new_i+3, new_j:new_j+3] += image[i, j]
            count_matrix[new_i:new_i+3, new_j:new_j+3] += 1
    
    # Process overlaps by averaging
    new_image /= count_matrix
    
    return new_image

seed_num = 200
tree_depth = 7
np.random.seed(seed_num)

# File path list (keeping original Chinese paths)
files = [
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104_ndvi_90m_feature.tif',
    # ... (other file paths remain same)
]

# Label files
labels_filename = [
    f'深圳湾GF1_PMS2_20161104/2016_2019深圳湾红树林生长率_90m_30%重叠_45_回归.tif',
    # ... (other label paths remain same)
]

shapefile_paths = [
    '深圳湾红树林分布数据/2016mangroveage.shp',
    # ... (other shapefile paths remain same)
]

# Lists for storing image data
images = []
predict_imgs = []
trans = [] 
labels_filelist = []
origion_shape = []
concatenated_shape = []
out_meta = []
filter_size = 45
stride = 15
shapefile_idx = 0
sigma = 1

# Read images and features, perform feature integration
for idx, file in enumerate(files):
    with rasterio.open(file) as src:
        # Read image data
        img = src.read()
        img = np.moveaxis(img, 0, -1) # Move bands to last dimension
        width, height, n_bands = img.shape
        
        if idx%7 <= 1:
            images.append(img)
            continue
        else:
            # Create array for filtered results
            filtered_shape = [img.shape[0]//stride + 1, img.shape[1]//stride + 1]
            if img.shape[0]%stride == 0: filtered_shape[0] -= 1
            if img.shape[1]%stride == 0: filtered_shape[1] -= 1
            img_filtered = np.zeros((filtered_shape[0], filtered_shape[1], img.shape[2]), dtype=img.dtype)
            for i in range(img.shape[-1]):
                temp = uniform_filter(img[:, :, i], size=filter_size, mode='reflect')[::stride, ::stride]
                img_filtered[:, :, i] = temp
                
            images.append(img_filtered)
            
    if (idx + 1) % 7 == 0:
        # Add pixel-to-boundary distance
        distance = compute_distance_transform(shapefile_paths[shapefile_idx], file, stride)
        invalid_idx = np.array(np.where(distance > 500))
        images.append(distance)
        shapefile_idx += 1
        concatenated_image = np.concatenate(images, axis=-1)
        n_bands = concatenated_image.shape[2]
        origion_shape.append([width, height])
        concatenated_shape.append([concatenated_image.shape[0], concatenated_image.shape[1]])
        images = []
        trans.append(gaussian_filter(concatenated_image, sigma=(sigma, sigma, 0)))
        out_meta.append([src.meta.copy(), src.transform])
        
# Normalize after tiling
for idx, imgs in enumerate(trans):
    # First 64 features from TIFs, exclude histogram and distance features
    min_vals = imgs.min(axis=0)[:,64:]
    max_vals = imgs.max(axis=0)[:,64:]
    
    # Min-max normalization per band
    imgs[:,:,64:] = (imgs[:,:,64:] - min_vals) / (max_vals - min_vals + 0.0001)
    imgs = imgs.reshape(-1, n_bands)
    trans[idx] = imgs

# Read label data
for idx, file in enumerate(labels_filename):
    with rasterio.open(file) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1)
    labels_filelist.append(img)

# Feature construction complete, begin model training
labels_img = np.concatenate(labels_filelist, axis=0)
transformed_img = np.concatenate(trans, axis=0)
del concatenated_image, images, img, min_vals, max_vals
gc.collect()

filter_index = clip_by_percentage(labels_img)
x = transformed_img[filter_index, :]
y = labels_img[filter_index, :]

X_train, X_test, y_train, y_test = train_test_split(
    x,    # Features
    y,    # Labels
    test_size=0.4,    # Test set ratio
)

# Initialize CatBoostRegressor
model = CatBoostRegressor(
    iterations=100,       # Number of trees
    learning_rate=0.1,    # Learning rate
    depth=10,            # Tree depth
    verbose=False        # No training output
)

print("Starting model training...")

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model test set performance:")
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R2: ", r2)

# 1. Create DataFrame
df = pd.DataFrame({'y_test': y_test, 'y_predict': y_pred})

# 2. Save as CSV
df.to_csv('CatBoost_prediction_result.csv', index=False, encoding='utf-8-sig')

# Save model
joblib.dump(model, "shenzhenwan_shenzhenwan_CatBoost_depth10_lr0.1_100.pkl")

del model
gc.collect()