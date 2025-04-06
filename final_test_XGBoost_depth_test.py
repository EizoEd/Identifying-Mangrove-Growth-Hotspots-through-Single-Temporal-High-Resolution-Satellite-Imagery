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

# Generate random indices and split
# To reduce the impact of data bias from experimental data on model fitting, we need to filter the overall data,
# minimizing the influence of areas with lower growth rates on model fitting
def clip_by_percentage(img, percentage=97, save_rate = 0.1):
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
    # Apply Gaussian transformation to sampled distances (mean=0, std=15)
    sigma = 15
    # gaussian_transform = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(- (sampled_distances ** 2) / (2 * sigma ** 2)
    gaussian_transform = np.exp(- (sampled_distances ** 2) / (2 * sigma ** 2))

    # 9. Add a new dimension
    gaussian_transform_with_dim = np.expand_dims(gaussian_transform, axis=-1)

    return gaussian_transform_with_dim
    
    # Save as TIF file
    # output_tif_path = tif_path.replace('.tif','_distance.tif')
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
    # return sampled_distances_with_dim


# Used to restore model prediction results. Since the model will calculate MGHI using a 90m window,
# but the actual mangrove product resolution is 30m, we need to restore the model prediction results
def expand_and_overlap(image):
    rows, cols  = image.shape
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
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104_ndvi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104_ndwi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104.tif',
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104_GLCM25_band1.tif',
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104_GLCM25_band2.tif',
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104_GLCM25_band3.tif',
    f'深圳湾GF1_PMS2_20161104/深圳湾GF1_PMS2_20161104_GLCM25_band4.tif',
    
    f'深圳湾GF1_PMS1_20170121/深圳湾GF1_PMS1_20170121_ndvi_90m_feature.tif',
    f'深圳湾GF1_PMS1_20170121/深圳湾GF1_PMS1_20170121_ndwi_90m_feature.tif',
    f'深圳湾GF1_PMS1_20170121/深圳湾GF1_PMS1_20170121.tif',
    f'深圳湾GF1_PMS1_20170121/深圳湾GF1_PMS1_20170121_GLCM25_band1.tif',
    f'深圳湾GF1_PMS1_20170121/深圳湾GF1_PMS1_20170121_GLCM25_band2.tif',
    f'深圳湾GF1_PMS1_20170121/深圳湾GF1_PMS1_20170121_GLCM25_band3.tif',
    f'深圳湾GF1_PMS1_20170121/深圳湾GF1_PMS1_20170121_GLCM25_band4.tif',
    
    
    f'深圳湾GF1_PMS1_20171211/深圳湾GF1_PMS1_20171211_ndvi_90m_feature.tif',
    f'深圳湾GF1_PMS1_20171211/深圳湾GF1_PMS1_20171211_ndwi_90m_feature.tif',
    f'深圳湾GF1_PMS1_20171211/深圳湾GF1_PMS1_20171211.tif',
    f'深圳湾GF1_PMS1_20171211/深圳湾GF1_PMS1_20171211_GLCM25_band1.tif',
    f'深圳湾GF1_PMS1_20171211/深圳湾GF1_PMS1_20171211_GLCM25_band2.tif',
    f'深圳湾GF1_PMS1_20171211/深圳湾GF1_PMS1_20171211_GLCM25_band3.tif',
    f'深圳湾GF1_PMS1_20171211/深圳湾GF1_PMS1_20171211_GLCM25_band4.tif',
    
    f'深圳湾GF1_PMS2_20170215/深圳湾GF1_PMS2_20170215_ndvi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20170215/深圳湾GF1_PMS2_20170215_ndwi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20170215/深圳湾GF1_PMS2_20170215.tif',
    f'深圳湾GF1_PMS2_20170215/深圳湾GF1_PMS2_20170215_GLCM25_band1.tif',
    f'深圳湾GF1_PMS2_20170215/深圳湾GF1_PMS2_20170215_GLCM25_band2.tif',
    f'深圳湾GF1_PMS2_20170215/深圳湾GF1_PMS2_20170215_GLCM25_band3.tif',
    f'深圳湾GF1_PMS2_20170215/深圳湾GF1_PMS2_20170215_GLCM25_band4.tif',
    
    f'深圳湾GF1_PMS2_20191002/深圳湾GF1_PMS2_20191002_ndvi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20191002/深圳湾GF1_PMS2_20191002_ndwi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20191002/深圳湾GF1_PMS2_20191002.tif',
    f'深圳湾GF1_PMS2_20191002/深圳湾GF1_PMS2_20191002_GLCM25_band1.tif',
    f'深圳湾GF1_PMS2_20191002/深圳湾GF1_PMS2_20191002_GLCM25_band2.tif',
    f'深圳湾GF1_PMS2_20191002/深圳湾GF1_PMS2_20191002_GLCM25_band3.tif',
    f'深圳湾GF1_PMS2_20191002/深圳湾GF1_PMS2_20191002_GLCM25_band4.tif',
    
    f'深圳湾GF1_PMS2_20200416/深圳湾GF1_PMS2_20200416_ndvi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20200416/深圳湾GF1_PMS2_20200416_ndwi_90m_feature.tif',
    f'深圳湾GF1_PMS2_20200416/深圳湾GF1_PMS2_20200416.tif',
    f'深圳湾GF1_PMS2_20200416/深圳湾GF1_PMS2_20200416_GLCM25_band1.tif',
    f'深圳湾GF1_PMS2_20200416/深圳湾GF1_PMS2_20200416_GLCM25_band2.tif',
    f'深圳湾GF1_PMS2_20200416/深圳湾GF1_PMS2_20200416_GLCM25_band3.tif',
    f'深圳湾GF1_PMS2_20200416/深圳湾GF1_PMS2_20200416_GLCM25_band4.tif',
    
]
# Label list
labels_filename = [
    f'深圳湾GF1_PMS2_20161104/2016_2019深圳湾红树林生长率_90m_30%重叠_45_回归.tif',
    f'深圳湾GF1_PMS1_20170121/2017_2020深圳湾红树林生长率_90m_30%重叠_45_回归.tif',
    f'深圳湾GF1_PMS1_20171211/2017_2020深圳湾红树林生长率_90m_30%重叠_45_回归.tif',
    f'深圳湾GF1_PMS2_20170215/2017_2020深圳湾红树林生长率_90m_30%重叠_45_回归.tif',
    f'深圳湾GF1_PMS2_20191002/2019_2022深圳湾红树林生长率_90m_30%重叠_45_回归.tif',
    f'深圳湾GF1_PMS2_20200416/2020_2023深圳湾红树林生长率_90m_30%重叠_45_回归.tif',
    
    
]

shapefile_paths = [

    '深圳湾红树林分布数据/2016mangroveage.shp',
    '深圳湾红树林分布数据/2017mangroveage.shp',
    '深圳湾红树林分布数据/2017mangroveage.shp',
    '深圳湾红树林分布数据/2017mangroveage.shp',
    '深圳湾红树林分布数据/2019mangroveage.shp',
    '深圳湾红树林分布数据/2020mangroveage.shp',
]
# List to store each image data
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

# Read images and features, perform feature integration processing
for idx, file in enumerate(files):
    with rasterio.open(file) as src:
        # Read image data and add to list
        img = src.read()
        img = np.moveaxis(img, 0, -1) # Move band dimension to last axis
        width, height, n_bands = img.shape
        
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
        n_bands = concatenated_image.shape[2]
        origion_shape.append([width,height])
        concatenated_shape.append([concatenated_image.shape[0], concatenated_image.shape[1]])
        images = []
        trans.append(gaussian_filter(concatenated_image, sigma=(sigma, sigma, 0)))
        out_meta.append([src.meta.copy(), src.transform])
        
for idx, imgs in enumerate(trans):
    # First split then normalize. The first 64 dimensions of features include all features read from tif files,
    # but during normalization we don't consider the histogram and distance features that come later
    min_vals = imgs.min(axis=0)[:,64:]
    max_vals = imgs.max(axis=0)[:,64:]
    
    # Perform min-max normalization band by band
    imgs[:,:,64:] = (imgs[:,:,64:] - min_vals) / (max_vals - min_vals+0.0001)
    imgs = imgs.reshape(-1, n_bands)
    trans[idx] = imgs


# Read label data
for idx, file in enumerate(labels_filename):
    with rasterio.open(file) as src:
        img = src.read()
        img = np.moveaxis(img, 0, -1) # Move band dimension to last axis
    labels_filelist.append(img)


labels_img      = np.concatenate(labels_filelist, axis=0)
transformed_img = np.concatenate(trans, axis=0)
del concatenated_image, images, img, min_vals, max_vals
gc.collect()


# Above completes the feature construction, now start model training
filter_index = clip_by_percentage(labels_img)

x = transformed_img[filter_index, :]
y = labels_img[filter_index, :]

X_train, X_test, y_train, y_test = train_test_split(
    x,    # Features
    y,                # Labels
    test_size=0.4,    # Test set proportion
)
print("Start training the model...")
for i in range(1, 65):
    model = xgb.XGBRegressor(
        n_estimators=100,   # Number of trees
        max_depth=i,        # Maximum tree depth
        learning_rate=0.1,  # Learning rate
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Performance of model with depth {i} on test set:")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("R2: ", r2)
    print("---------------------------------")

# Save the model
# joblib.dump(model, "shenzhenwan_shenzhenwan_XGBoost_depth10_lr0.1_100.pkl")

del model
gc.collect()

