import os
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from pyphoon2.DigitalTyphoonDataset import DigitalTyphoonDataset

def select_digital_typhoon_data():
    path_image = '/Users/tammy/Documents/毕设/WP/image/'
    path_metadata = '/Users/tammy/Documents/毕设/WP/metadata/'
    path_metadata_json = '/Users/tammy/Documents/毕设/WP/metadata.json'

    chosen_typhoon_file = '/Users/tammy/Documents/毕设/Typhoon-forecasting/pyphoon2/Shanghai.txt'
    with open(chosen_typhoon_file, 'r') as file:
        chosen_typhoon_ids = file.read().splitlines()

    filtered_image_dir = '/Users/tammy/Documents/毕设/WP/filtered_images/'
    filtered_metadata_dir = '/Users/tammy/Documents/毕设/WP/filtered_metadata/'
    filtered_metadata_json = '/Users/tammy/Documents/毕设/WP/filtered_metadata.json'
    if not os.path.exists(filtered_image_dir):
        os.makedirs(filtered_image_dir)
    if not os.path.exists(filtered_metadata_dir):
        os.makedirs(filtered_metadata_dir)

    for typhoon_id in chosen_typhoon_ids:
        image_path = os.path.join(path_image, typhoon_id)
        metadata_path = os.path.join(path_metadata, typhoon_id + '.csv')
        if not os.path.exists(image_path) or not os.path.exists(metadata_path):
            print(f"Typhoon {typhoon_id} not found in the dataset")
            continue
        os.system(f'cp -r {image_path} {filtered_image_dir}')
        os.system(f'cp -r {metadata_path} {filtered_metadata_dir}')

    metadata = {}
    with open(path_metadata_json, 'r') as file:
        metadata = json.load(file)
    filtered_metadata = {}
    for typhoon_id in chosen_typhoon_ids:
        if typhoon_id in metadata:
            filtered_metadata[typhoon_id] = metadata[typhoon_id]
    with open(filtered_metadata_json, 'w') as file:
        json.dump(filtered_metadata, file)
        
    dataset_obj = DigitalTyphoonDataset(
        filtered_image_dir,
        filtered_metadata_dir,
        filtered_metadata_json,
        ('lat', 'lng'), 
        verbose=False
    )

    length = len(dataset_obj)
    print(length)
    print(dataset_obj.get_number_of_sequences())
    return dataset_obj

def process_hdf5_files(input_directory, output_directory, metadata_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # The size after cropping and zooming
    target_size = 512  
    target_center_lon = 121.4737
    target_center_lat = 31.2304
    target_km_per_pixel = 1250 / 512  

    # Compute the start and end coordinates of the target data  
    target_lon_start = target_center_lon - 625 / (111 * np.cos(np.radians(target_center_lat)))
    target_lon_end = target_center_lon + 625 / (111 * np.cos(np.radians(target_center_lat)))
    target_lat_start = target_center_lat - 625 / 111
    target_lat_end = target_center_lat + 625 / 111

    for typhoon_number in sorted(os.listdir(input_directory)):
        typhoon_input_dir = os.path.join(input_directory, typhoon_number)
        typhoon_output_dir = os.path.join(output_directory, typhoon_number)
        metadata_file_path = os.path.join(metadata_directory, typhoon_number + '.csv')

        if not os.path.exists(typhoon_output_dir):
            os.makedirs(typhoon_output_dir)

        if typhoon_number == '.DS_Store':
            continue

        metadata = np.genfromtxt(metadata_file_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
        latitudes = metadata['lat']
        longitudes = metadata['lng']
        files = metadata['file_1']

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

                    km_per_pixel = 2500 / 512
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
                        cropped_data = np.full((1, 1), 255, dtype=data.dtype)  # fill with white

                    # Compute the start and end coordinates of the cropped data
                    cropped_lon_start = lon_start + x_min * lon_per_pixel
                    cropped_lon_end = lon_start + x_max * lon_per_pixel
                    cropped_lat_start = lat_start + y_min * lat_per_pixel
                    cropped_lat_end = lat_start + y_max * lat_per_pixel

                    zoom_factor_x = km_per_pixel / target_km_per_pixel
                    zoom_factor_y = km_per_pixel / target_km_per_pixel

                    zoomed_data = zoom(cropped_data, (zoom_factor_y, zoom_factor_x))

                    # Create a new array with the target size and fill it with white
                    data_fixed = np.full((target_size, target_size), 255, dtype=data.dtype)

                    # Compute the coordinates of the zoomed data in the target array
                    target_x_min = int((cropped_lon_start - target_lon_start) / target_km_per_pixel)
                    target_y_min = int((cropped_lat_start - target_lat_start) / target_km_per_pixel)

                    target_x_max = target_x_min + zoomed_data.shape[1]
                    target_y_max = target_y_min + zoomed_data.shape[0]

                    # Clip the coordinates to the target array
                    target_x_min = max(0, target_x_min)
                    target_x_max = min(target_size, target_x_max)
                    target_y_min = max(0, target_y_min)
                    target_y_max = min(target_size, target_y_max)

                    # Compute the overlap between the target array and the zoomed data
                    overlap_x_min = max(0, -target_x_min)
                    overlap_x_max = overlap_x_min + (target_x_max - target_x_min)
                    overlap_y_min = max(0, -target_y_min)
                    overlap_y_max = overlap_y_min + (target_y_max - target_y_min)

                    data_fixed[overlap_y_min:overlap_y_max, overlap_x_min:overlap_x_max] = zoomed_data[
                        :overlap_y_max-overlap_y_min, :overlap_x_max-overlap_x_min
                    ]

                    with h5py.File(output_file_path, 'w') as f_fixed:
                        f_fixed.create_dataset('data', data=data_fixed)


def visualize_comparison(original_directory, fixed_directory, metadata_directory):
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

                with h5py.File(original_file_path, 'r') as f_original, h5py.File(fixed_file_path, 'r') as f_fixed:
                    original_data = f_original['Infrared'][:]
                    fixed_data = f_fixed['data'][:]


                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    axes[0].imshow(original_data)
                    axes[0].set_title(f'Original: {file_name}, Lat: {latitudes[index]}, Lon: {longitudes[index]}')
                    axes[1].imshow(fixed_data, vmin=original_data.min(), vmax=original_data.max())
                    axes[1].set_title(f'Fixed: {file_name}')
                    plt.show()

if __name__ == '__main__':
    dataset_obj = select_digital_typhoon_data()
    input_hdf5_directory = '/Users/tammy/Documents/毕设/WP/filtered_images'
    output_hdf5_directory = '/Users/tammy/Documents/毕设/WP/fixed_filtered_images'
    metadata_directory = '/Users/tammy/Documents/毕设/WP/filtered_metadata'
    process_hdf5_files(input_hdf5_directory, output_hdf5_directory, metadata_directory)
    visualize_comparison(input_hdf5_directory, output_hdf5_directory, metadata_directory)
