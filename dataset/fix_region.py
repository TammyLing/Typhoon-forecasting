import os
import h5py
import numpy as np
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def process_hdf5_files(input_directory, output_directory, metadata_directory):
    """
    Process the HDF5 files in the input directory and save the processed files in the output directory.
    The processed files are cropped and zoomed to a fixed size and location.

    Args:
    - input_directory: The directory containing the input HDF5 files
    - output_directory: The directory to save the processed HDF5 files
    - metadata_directory: The directory containing the metadata CSV files

    Returns:
    - None
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # The size after cropping and zooming
    target_size = 128
    
    # Fixed latitude and longitude range for the target image (Shanghai)
    # target_lon_start = 116.4737
    # target_lon_end = 126.4737
    # target_lat_start = 26.2304
    # target_lat_end = 36.2304

    # Fixed latitude and longitude range for the target image (Taiwan)
    target_lon_start = 116.0794
    target_lon_end = 126.0794
    target_lat_start = 18.9037
    target_lat_end = 28.9037

    # Compute the pixel resolutions for the target region
    target_lon_per_pixel = (target_lon_end - target_lon_start) / target_size
    target_lat_per_pixel = (target_lat_end - target_lat_start) / target_size

    for typhoon_number in sorted(os.listdir(input_directory)):
        typhoon_input_dir = os.path.join(input_directory, typhoon_number)
        typhoon_output_dir = os.path.join(output_directory, typhoon_number)
        metadata_file_path = os.path.join(metadata_directory, typhoon_number + '.csv')

        if not os.path.exists(typhoon_output_dir):
            os.makedirs(typhoon_output_dir)

        if typhoon_number == '.DS_Store':
            continue
        
        # Load the metadata
        metadata = np.genfromtxt(metadata_file_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        latitudes = metadata['lat']
        longitudes = metadata['lng']
        files = metadata['file_1']

        # Process each HDF5 file
        for file_name in sorted(os.listdir(typhoon_input_dir)):
            if file_name.endswith('.h5'):
                input_file_path = os.path.join(typhoon_input_dir, file_name)
                output_file_path = os.path.join(typhoon_output_dir, file_name)

                try:
                    index = np.where(files == file_name)[0][0]
                except IndexError:
                    print(f"File {file_name} not found in metadata")
                    continue

                with h5py.File(input_file_path, 'r') as f:
                    data = f['Infrared'][:]
                    lat = latitudes[index]
                    lon = longitudes[index]

                    # Compute the pixel resolutions for the original data
                    km_per_pixel = 2500 / 128
                    lat_per_pixel = km_per_pixel / 111
                    lon_per_pixel = km_per_pixel / (111 * np.cos(np.radians(lat)))

                    center_lon = lon
                    center_lat = lat

                    # Compute the start and end coordinates of the original data
                    lon_start = center_lon - (data.shape[1] // 2) * lon_per_pixel
                    lon_end = center_lon + (data.shape[1] // 2) * lon_per_pixel
                    lat_start = center_lat - (data.shape[0] // 2) * lat_per_pixel
                    lat_end = center_lat + (data.shape[0] // 2) * lat_per_pixel

                    # Compute the coordinates of the target data in the original data
                    x_min = int((target_lon_start - lon_start) / lon_per_pixel)
                    x_max = int((target_lon_end - lon_start) / lon_per_pixel)
                    y_min = int((target_lat_start - lat_start) / lat_per_pixel)
                    y_max = int((target_lat_end - lat_start) / lat_per_pixel)

                    x_min = max(0, x_min)
                    x_max = min(data.shape[1], x_max)
                    y_min = max(0, y_min)
                    y_max = min(data.shape[0], y_max)

                    if x_min < x_max and y_min < y_max:
                        cropped_data = data[y_min:y_max, x_min:x_max]  # crop the data
                    else:
                        cropped_data = np.full((1, 1), np.nan, dtype=data.dtype)  # fill with NaN
                    
                    print(cropped_data.shape)

                    # Compute the start and end coordinates of the cropped data
                    cropped_lon_start = lon_start + x_min * lon_per_pixel
                    cropped_lon_end = lon_start + x_max * lon_per_pixel
                    cropped_lat_start = lat_start + y_min * lat_per_pixel
                    cropped_lat_end = lat_start + y_max * lat_per_pixel

                    # Calculate zoom factors based on degree size instead of km
                    zoom_factor_x = target_size / cropped_data.shape[1]
                    zoom_factor_y = target_size / cropped_data.shape[0]

                    zoomed_data = zoom(cropped_data, (zoom_factor_y, zoom_factor_x))

                    # Create a new array with the target size and fill it with NaN
                    data_fixed = np.full((target_size, target_size), np.nan, dtype=data.dtype)

                    # Compute the coordinates of the zoomed data in the target array
                    target_x_min = int((cropped_lon_start - target_lon_start) / target_lon_per_pixel)
                    target_y_min = int((cropped_lat_start - target_lat_start) / target_lat_per_pixel)

                    target_x_max = target_x_min + zoomed_data.shape[1]
                    target_y_max = target_y_min + zoomed_data.shape[0]

                    # Clip the coordinates to the target array
                    target_x_min = max(0, target_x_min)
                    target_x_max = min(target_size, target_x_max)
                    target_y_min = max(0, target_y_min)
                    target_y_max = min(target_size, target_y_max)

                    # Compute the overlap between the target array and the zoomed data
                    overlap_x_min = max(0, -target_x_min)
                    overlap_x_max = max(0, target_x_max - target_x_min)
                    overlap_y_min = max(0, -target_y_min)
                    overlap_y_max = max(0, target_y_max - target_y_min)

                    # Ensure the overlap does not exceed the zoomed_data shape
                    overlap_x_max = min(overlap_x_max, zoomed_data.shape[1])
                    overlap_y_max = min(overlap_y_max, zoomed_data.shape[0])

                    # Fill the target array with zoomed data
                    data_fixed[target_y_min:target_y_max, target_x_min:target_x_max] = zoomed_data[
                        overlap_y_min:overlap_y_max, overlap_x_min:overlap_x_max
                    ]

                    with h5py.File(output_file_path, 'w') as f_fixed:
                        f_fixed.create_dataset('data', data=data_fixed)

def visualize_comparison(original_directory, fixed_directory, metadata_directory):
    km_per_pixel = 2500 / 128
    # Fixed latitude and longitude range for the target image (Shanghai)
    # target_lat = 31.2304
    # target_lon = 121.4737

    # Fixed latitude and longitude range for the target image (Taiwan)
    target_lat = 23.9037
    target_lon = 121.0794

    # Create a figure for each pair of original and fixed data
    for typhoon_number in sorted(os.listdir(original_directory)):
        if typhoon_number == '.DS_Store':
            continue

        original_typhoon_dir = os.path.join(original_directory, typhoon_number)
        fixed_typhoon_dir = os.path.join(fixed_directory, typhoon_number)
        metadata_file_path = os.path.join(metadata_directory, typhoon_number + '.csv')

        metadata = np.genfromtxt(metadata_file_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        latitudes = metadata['lat']
        longitudes = metadata['lng']
        files = metadata['file_1']

        for file_name in sorted(os.listdir(original_typhoon_dir)):
            if file_name.endswith('.h5'):
                original_file_path = os.path.join(original_typhoon_dir, file_name)
                fixed_file_path = os.path.join(fixed_typhoon_dir, file_name)

                try:
                    index = np.where(files == file_name)[0][0]
                except IndexError:
                    print(f"File {file_name} not found in metadata")
                    continue
                
                # Load the original and fixed data
                with h5py.File(original_file_path, 'r') as f_original, h5py.File(fixed_file_path, 'r') as f_fixed:
                    original_data = f_original['Infrared'][:]
                    fixed_data = f_fixed['data'][:]

                    center_lat = latitudes[index]
                    center_lon = longitudes[index]
                    lon_per_pixel = km_per_pixel / (111 * np.cos(np.radians(center_lat)))
                    lat_per_pixel = km_per_pixel / 111

                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                    # plot the original data
                    axes[0].imshow(np.flipud(original_data), extent=[center_lon - original_data.shape[1] * lon_per_pixel / 2,
                                                          center_lon + original_data.shape[1] * lon_per_pixel / 2,
                                                          center_lat - original_data.shape[0] * lat_per_pixel / 2,
                                                          center_lat + original_data.shape[0] * lat_per_pixel / 2])
                    axes[0].set_title(f'Original: {file_name}, Lat: {latitudes[index]}, Lon: {longitudes[index]}')
                    axes[0].set_xlabel('Longitude')
                    axes[0].set_ylabel('Latitude')

                    # add red point to mark the center of target region
                    axes[0].plot(target_lon, target_lat, 'ro') 

                    # plot the fixed data
                    cmap = cm.jet
                    cmap.set_bad(color='white')
                    # axes[1].imshow(np.flipud(fixed_data), extent=[116.4737, 126.4737, 26.2304, 36.2304],
                    #                vmin=original_data.min(), vmax=original_data.max(), cmap=cmap)
                    axes[1].imshow(np.flipud(fixed_data), extent=[116.0794, 126.0794, 18.9037, 28.9037],
                                   vmin=original_data.min(), vmax=original_data.max(), cmap=cmap)
                    axes[1].set_title(f'Fixed: {file_name}')
                    axes[1].set_xlabel('Longitude')
                    axes[1].set_ylabel('Latitude')
                    

                    # add red point to mark the center of target region
                    axes[1].plot(target_lon, target_lat, 'ro') 

                    plt.show()

if __name__ == '__main__':
    input_hdf5_directory = './dataset/pre_processed_dataset/Digital_typhoon/filtered_images_taiwan'
    output_hdf5_directory = './dataset/pre_processed_dataset/Digital_typhoon/fixed_images_taiwan'
    metadata_directory = './dataset/pre_processed_dataset/Digital_typhoon/filtered_metadata_taiwan'
    # process_hdf5_files(input_hdf5_directory, output_hdf5_directory, metadata_directory)
    visualize_comparison(input_hdf5_directory, output_hdf5_directory, metadata_directory)
