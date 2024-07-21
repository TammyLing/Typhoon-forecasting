import os
import json

def select_digital_typhoon_data():
    '''
    Selects the typhoons that are in the Shanghai region from the Digital Typhoon dataset
    
    Args:   
    - None

    Returns:
    - None
    '''
    # get the base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir_wp = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path_image = os.path.join(base_dir_wp, 'WP', 'image')
    path_metadata = os.path.join(base_dir_wp, 'WP', 'metadata')
    path_metadata_json = os.path.join(base_dir_wp, 'WP', 'metadata.json')

    # get the typhoon ids that are in the Taiwan region
    chosen_typhoon_file = os.path.join(base_dir, 'Taiwan.txt')
    with open(chosen_typhoon_file, 'r') as file:
        chosen_typhoon_ids = file.read().splitlines()

    # create the filtered dataset path
    filtered_image_dir = os.path.join(base_dir, 'pre_processed_dataset', 'Digital_typhoon', 'filtered_images_Taiwan')
    filtered_metadata_dir = os.path.join(base_dir, 'pre_processed_dataset', 'Digital_typhoon', 'filtered_metadata_Taiwan')
    filtered_metadata_json = os.path.join(base_dir, 'pre_processed_dataset', 'Digital_typhoon', 'filtered_metadata_Taiwan.json')

    if not os.path.exists(filtered_image_dir):
        os.makedirs(filtered_image_dir)
    if not os.path.exists(filtered_metadata_dir):
        os.makedirs(filtered_metadata_dir)

    # copy the images and metadata of the chosen typhoons to the filtered dataset
    for typhoon_id in chosen_typhoon_ids:
        image_path = os.path.join(path_image, typhoon_id)
        metadata_path = os.path.join(path_metadata, typhoon_id + '.csv')
        if not os.path.exists(image_path) or not os.path.exists(metadata_path):
            print(f"Typhoon {typhoon_id} not found in the dataset")
            continue
        os.system(f'cp -r {image_path} {filtered_image_dir}')
        os.system(f'cp -r {metadata_path} {filtered_metadata_dir}')

    # filter the metadata json file
    metadata = {}
    with open(path_metadata_json, 'r') as file:
        metadata = json.load(file)
    filtered_metadata = {}
    for typhoon_id in chosen_typhoon_ids:
        if typhoon_id in metadata:
            filtered_metadata[typhoon_id] = metadata[typhoon_id]
    with open(filtered_metadata_json, 'w') as file:
        json.dump(filtered_metadata, file)

if __name__ == '__main__':
    select_digital_typhoon_data()
