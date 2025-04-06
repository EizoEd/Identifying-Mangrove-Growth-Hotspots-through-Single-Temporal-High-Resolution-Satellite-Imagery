import rasterio
import numpy as np
from skimage import exposure

def linear_percentile_stretch(arr, lower_percentile=1, upper_percentile=99, levels=32):
    # Calculate lower and upper percentile bounds
    lower_bound = np.percentile(arr, lower_percentile)
    upper_bound = np.percentile(arr, upper_percentile)
    
    # Clip array to remove out-of-bound values
    clipped_arr = np.clip(arr, lower_bound, upper_bound)
    
    # Scale linearly to range [0, levels-1]
    scaled_arr = (clipped_arr - lower_bound) / (upper_bound - lower_bound) * (levels - 1)
    
    # Round and convert to integer
    final_arr = np.round(scaled_arr).astype(int)
    
    return final_arr

def compute_histograms(image, window_size, stride):
    """ Compute histogram feature map """
    rows, cols = image.shape
    out_rows = rows // stride + 1
    out_cols = cols // stride + 1
    if rows % stride == 0:
        out_rows -= 1
    if cols % stride == 0:
        out_cols -= 1
    feature_map = np.zeros((out_rows, out_cols, 32), dtype=np.float32)
    
    for i in range(out_rows):
        for j in range(out_cols):
            row_start = i * stride
            col_start = j * stride
            window = image[row_start:row_start+window_size, col_start:col_start+window_size]
            if window.size == 0 or np.all(window == 0):
                normalized_histogram = np.zeros(32)  # Set histogram to all 0s for invalid windows
            else:
                histogram, _ = np.histogram(window, bins=32, range=(0, 31))
                total_pixels = window.size  # Calculate total number of pixels in the window
                normalized_histogram = histogram / total_pixels  # Normalize histogram
            feature_map[i, j, :] = normalized_histogram
    
    return feature_map

def process_image(input_ndvi_path, input_ndwi_path, output_ndvi_path, output_ndwi_path, window_size, stride):
    """ Read from tif files, calculate indices, generate and save feature maps """
    with rasterio.open(input_ndvi_path) as src:
        ndvi = src.read(1)
    with rasterio.open(input_ndwi_path) as src:
        ndwi = src.read(1)
        # Stretch to 32 grayscale levels and compute feature maps
        ndvi = linear_percentile_stretch(ndvi)
        print('ndvi')
        ndwi = linear_percentile_stretch(ndwi)
        print('ndwi')
        ndvi_feature_map = compute_histograms(ndvi, window_size, stride)
        ndwi_feature_map = compute_histograms(ndwi, window_size, stride)
        
        # Output metadata setup
        out_meta = src.meta.copy()
        out_meta.update({
            'dtype': 'float32',
            'count': 32,
            'height': ndvi_feature_map.shape[0],
            'width': ndvi_feature_map.shape[1]
        })
        
    # Save NDVI feature map
    with rasterio.open(output_ndvi_path, 'w', **out_meta) as dst:
        for k in range(32):
            dst.write(ndvi_feature_map[:, :, k], k+1)

    # Save NDWI feature map
    with rasterio.open(output_ndwi_path, 'w', **out_meta) as dst:
        for k in range(32):
            dst.write(ndwi_feature_map[:, :, k], k+1)

# Configuration parameters
files = [
    "湛江GF1B_PMS_20240402",
]
for file_path in files:
    filename = file_path.split(r'/')[-1]
    input_ndvi_path = f'{file_path}/{filename}_ndvi.tif'
    input_ndwi_path = f'{file_path}/{filename}_ndwi.tif'
    output_ndvi_feature_map = f'{file_path}/{filename}_ndvi_90m_feature.tif'
    output_ndwi_feature_map = f'{file_path}/{filename}_ndwi_90m_feature.tif'
    window_size = 45  # Window size
    stride = 15       # Stride

    # Process image
    process_image(input_ndvi_path, input_ndwi_path, output_ndvi_feature_map, output_ndwi_feature_map, window_size, stride)
