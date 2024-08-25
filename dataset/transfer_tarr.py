import os
import re
import datetime
import h5py
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import shutil

def parse_typhoon_filenames(base_dir):
    """
    Parse the filenames of Digital Typhoon dataset to extract the time information.
    
    Args:
    - base_dir (str): The base directory of the Digital Typhoon dataset.
    
    Returns:
    - typhoon_times (dict): A dictionary containing the time information for each typhoon.
    """
    typhoon_times = {}
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            subdir_times = []
            for filename in os.listdir(subdir_path):
                # match the filename pattern to extract the time information
                match = re.match(r'(\d{10})-\d{6}-.+\.h5', filename)
                if match:
                    time_str = match.group(1)
                    time = datetime.datetime.strptime(time_str, "%Y%m%d%H")
                    subdir_times.append((time, filename))
            subdir_times.sort(key=lambda x: x[0])  # sort by time
            typhoon_times[subdir] = subdir_times
            if not subdir_times:
                print(f"No valid times for {subdir}")
    return typhoon_times

def read_typhoon_images(typhoon_file):
    """
    Read the image data from a Digital Typhoon dataset file.
    
    Args:
    - typhoon_file (str): The file path of the Digital Typhoon dataset.
    
    Returns:
    - image_data (np.ndarray): The image data read from the file.
    """
    with h5py.File(typhoon_file, 'r') as f:
        first_key = list(f.keys())[0]
        image_data = f[first_key][:]
    return image_data

def print_nc_time(era5_file):
    """
    Print the time dimension in an ERA5 NetCDF file.
    
    Args:
    - era5_file (str): The file path of the ERA5 NetCDF file.
    
    Returns:
    - None
    """
    if not os.path.isfile(era5_file):
        print(f"File {era5_file} does not exist.")
        return

    try:
        era5_ds = xr.open_dataset(era5_file)
        print("Time in ERA5 dataset:")
        print(era5_ds['time'].values)
    except Exception as e:
        print(f"An error occurred: {e}")

def open_and_select_era5_data(era5_file, time_coords):
    """
    Open an ERA5 NetCDF file and select data for specific time coordinates.
    
    Args:
    - era5_file (str): The file path of the ERA5 NetCDF file.
    - time_coords (list): A list of time coordinates to select data for.
    
    Returns:
    - selected_data (list): A list of selected data arrays for each time coordinate.
    """
    print(f"Opening and selecting data from {era5_file}")
    if not os.path.isfile(era5_file):
        print(f"File {era5_file} does not exist.")
        return []

    try:
        # Open the ERA5 dataset
        era5_ds = xr.open_dataset(era5_file)
        era5_time_values = np.array(era5_ds.time.values, dtype='datetime64[ns]')
        selected_data = []

        variables_to_select = ['u10', 'v10', 'sp', 't2m']

        for time in time_coords:
            time = np.datetime64(time, 'ns')
            if any(np.isclose((era5_time_values - time).astype('timedelta64[h]').astype(int), 0, atol=1)):
                selected_vars = era5_ds.sel(time=time, method='nearest')[variables_to_select]
                selected_data.append(selected_vars)
            else:
                print(f"No matching data for time {time}")
                print(f"Closest available times: {era5_time_values[np.argsort(np.abs(era5_time_values - time))[:5]]}")

        return selected_data
    except Exception as e:
        print(f"An error occurred while opening and selecting data: {e}")
        return []

def align_and_save_datasets(era5_dir, typhoon_dir, output_dir, typhoon):
    """
    Align the ERA5 and Digital Typhoon datasets for a specific typhoon and save the aligned data to a Zarr file.
    
    Args:
    - era5_dir (str): The directory containing the ERA5 NetCDF files.
    - typhoon_dir (str): The directory containing the Digital Typhoon dataset files.
    - output_dir (str): The directory to save the aligned data to.
    - typhoon (str): The name of the typhoon to process.
    
    Returns:
    - success (bool): True if the data was successfully aligned and saved, False otherwise.
    """
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # create typhoon directory if it does not exist
    if not os.path.exists(os.path.join(output_dir, typhoon)):
        os.mkdir(os.path.join(output_dir, typhoon))
    
    typhoon_times = parse_typhoon_filenames(typhoon_dir)

    times = typhoon_times.get(typhoon, [])
    if not times:
        print(f"No times found for typhoon {typhoon}")
        return

    images = []
    time_coords = []
    valid_times = []

    # Read image data for each time point
    for time, typhoon_file in times:
        typhoon_filepath = os.path.join(typhoon_dir, typhoon, typhoon_file)
        typhoon_img = read_typhoon_images(typhoon_filepath)
        
        # Check for NaN values in image data
        if np.isnan(typhoon_img).any():
            print(f"Skipping file {typhoon_file} due to NaN values in image data")
            continue

        images.append(typhoon_img)
        time_coords.append(np.datetime64(time, 'ns'))
        valid_times.append(time)

    if not images:
        print(f"No valid images found for typhoon {typhoon}")
        return

    images = np.array(images)

    era5_file = os.path.join(era5_dir, f'ERA5_FIXED_REGION_{typhoon}.nc')
    print(f"Using ERA5 file: {era5_file}")

    # Open and select ERA5 data for the same time points
    aligned_era5_data = open_and_select_era5_data(era5_file, time_coords)

    if not aligned_era5_data:
        print(f"No ERA5 data found for typhoon {typhoon} at times {time_coords}")
        return

    # Check for NaN values in ERA5 data for each time point
    valid_indices = []
    for i, era5_data in enumerate(aligned_era5_data):
        skip = False
        for var in era5_data.data_vars:
            if era5_data[var].isnull().any():
                print(f"Skipping time {time_coords[i]} due to NaN values in ERA5 data {var}")
                skip = True
                break
        if not skip:
            valid_indices.append(i)

    if not valid_indices:
        print(f"No valid ERA5 data found for typhoon {typhoon}")
        return

    # Filter out invalid data
    images = images[valid_indices]
    time_coords = [time_coords[i] for i in valid_indices]
    aligned_era5_data = [aligned_era5_data[i] for i in valid_indices]

    # Combine the ERA5 and image data
    combined_era5_ds = xr.concat(aligned_era5_data, dim='time').drop_duplicates(dim='time')

    image_da = xr.DataArray(
        images, 
        coords={'time': np.array(time_coords, dtype='datetime64[ns]'), 'y': np.arange(images.shape[1]), 'x': np.arange(images.shape[2])}, 
        dims=['time', 'y', 'x'], 
        name='image_data'
    )

    # calculate the magnitude of the wind
    u = combined_era5_ds['u10'].values
    v = combined_era5_ds['v10'].values
    tarr = np.sqrt(u**2 + v**2)

    magnitude_da = xr.DataArray(
        tarr,
        coords={'time': combined_era5_ds['time'].values, 'latitude': combined_era5_ds['latitude'].values, 'longitude': combined_era5_ds['longitude'].values},
        dims=['time', 'latitude', 'longitude'],
        name='magnitude'
    )

    combined_ds = xr.merge([combined_era5_ds, image_da, magnitude_da])

    typhoon_folder = os.path.basename(os.path.dirname(typhoon_filepath))
    output_path = os.path.join(output_dir, typhoon_folder, f"{typhoon}.zarr")

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))

    combined_ds.to_zarr(output_path, consolidated=True)
    print(f"Saved aligned data to {output_path}")
    return True


def list_zarr_variables(zarr_path):
    """
    List the variables in a Zarr dataset.
    
    Args:
    - zarr_path (str): The file path of the Zarr dataset.
    
    Returns:
    - ds (xr.Dataset): The Zarr dataset
    """
    try:
        ds = xr.open_zarr(zarr_path)
        print("Variables in the dataset:")
        for var in ds.variables:
            print(var)
        return ds
    except Exception as e:
        print(f"An error occurred while listing Zarr variables: {e}")
        return None

def compare_era5_and_typhoon_images(zarr_ds, typhoon):
    """
    Compare and visualize the ERA5 and Digital Typhoon images for a specific typhoon.
    
    Args:
    - zarr_ds (xr.Dataset): The Zarr dataset containing the aligned data.
    - typhoon (str): The name of the typhoon to process.
    
    Returns:
    - None
    """
    try:
        era5_var_names = ['u10', 'v10', 'sp', 't2m', 'magnitude']
        n_vars = len(era5_var_names)

        # Visualize the data for each time step
        for i in range(len(zarr_ds.time)):
            fig, axes = plt.subplots(2, 3, figsize=(24, 12)) 
            time = zarr_ds.time.isel(time=i).values

            lon, lat = np.meshgrid(zarr_ds.longitude, zarr_ds.latitude)
            u = zarr_ds['u10'].isel(time=i).values
            v = zarr_ds['v10'].isel(time=i).values

            # Skip every other point for wind data
            skip = (slice(None, None, 2), slice(None, None, 2))

            for j, var_name in enumerate(era5_var_names):
                ax = axes[j//3, j%3]
                if var_name in zarr_ds:
                    era5_data = zarr_ds[var_name].isel(time=i).values
                    contour = ax.contourf(lon, lat, era5_data, cmap='viridis')
                    if var_name in ['u10', 'v10']:  # add arrows for wind data  
                        ax.quiver(lon[skip], lat[skip], u[skip], v[skip], color='white', scale=50, width=0.003)
                    ax.set_title(f"{var_name} at {np.datetime_as_string(time, unit='s')}")
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    fig.colorbar(contour, ax=ax)
                else:
                    ax.set_title(f"{var_name} not found")
                    ax.axis('off')

            ax_typhoon = axes[1, 2]
            typhoon_img = zarr_ds['image_data'].isel(time=i).values
            # lon_start, lon_end = 116.4737, 126.4737
            # lat_start, lat_end = 26.2304, 36.2304

            lon_start, lon_end = 116.0794, 126.0794
            lat_start, lat_end = 18.9037, 28.9037
    
            extent = [lon_start, lon_end, lat_start, lat_end]
            typhoon_plot = ax_typhoon.imshow(np.flipud(typhoon_img), extent=extent, cmap='viridis', aspect='auto')
            ax_typhoon.set_title('Typhoon Image')
            ax_typhoon.set_xlabel('Longitude')
            ax_typhoon.set_ylabel('Latitude')
            fig.colorbar(typhoon_plot, ax=ax_typhoon) 

            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"An error occurred while comparing and visualizing images: {e}")


def visualize_original_data(typhoon_file, era5_file):
    """
    Visualize the original data from the Digital Typhoon and ERA5 datasets.
    
    Args:
    - typhoon_file (str): The file path of the Digital Typhoon dataset.
    - era5_file (str): The file path of the ERA5 NetCDF file.
    
    Returns:
    - None
    """

    try:
        typhoon_img = read_typhoon_images(typhoon_file)
        era5_ds = xr.open_dataset(era5_file)

        # Get the time of the first data point in the ERA5 dataset
        era5_time = era5_ds.time.isel(time=0).values

        # Visualize the data
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].set_title(f'Data from {typhoon_file}')
        im = axes[0].imshow(typhoon_img, cmap='jet', origin='lower')
        plt.colorbar(im, ax=axes[0])

        axes[1].set_title(f'Data from NetCDF file: Temperature at {era5_time}')
        cs = axes[1].contourf(era5_ds.longitude, era5_ds.latitude, era5_ds['t2m'].isel(time=0).values, cmap='jet')
        plt.colorbar(cs, ax=axes[1])


        axes[0].set_aspect('equal', adjustable='box')
        axes[1].set_aspect('equal', adjustable='box')

        plt.show()
    except Exception as e:
        print(f"An error occurred while visualizing original data: {e}")

if __name__ == "__main__":
    typhoon_dir = './dataset/pre_processed_dataset/Digital_typhoon/fixed_images_taiwan_64' 
    era5_dir = './dataset/ERA5_single_level_taiwan'  
    output_dir = './dataset/pre_processed_dataset/ERA5_without_nan_taiwan_64' 

    typhoon_times = parse_typhoon_filenames(typhoon_dir)
    for typhoon in typhoon_times.keys():
        if typhoon != '.DS_Store':
            print(f"Processing typhoon: {typhoon}")

            typhoon_file = os.path.join(typhoon_dir, typhoon, typhoon_times[typhoon][0][1])
            # visualize_original_data(typhoon_file, os.path.join(era5_dir, f'ERA5_FIXED_REGION_{typhoon}.nc'))

            if align_and_save_datasets(era5_dir, typhoon_dir, output_dir, typhoon):
                zarr_path = os.path.join(output_dir, typhoon, f"{typhoon}.zarr")
                print(f"Loading aligned dataset from: {zarr_path}")
                zarr_ds = list_zarr_variables(zarr_path)

            zarr_ds = xr.open_zarr(os.path.join(output_dir, typhoon, f"{typhoon}.zarr"))

            compare_era5_and_typhoon_images(zarr_ds, typhoon)
