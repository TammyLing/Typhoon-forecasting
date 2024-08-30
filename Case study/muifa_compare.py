import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import xarray as xr
import random
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from PIL import Image
import sys
from ema_pytorch import EMA
from torchvision import transforms as T, utils
from CDDPM.models.model import Palette
from CDDPM.models.network import Network
import argparse
from scipy.linalg import sqrtm
from torchvision import models
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from CDDPM.core.logger import VisualWriter, InfoLogger
import CDDPM.core.praser as Praser
import CDDPM.core.util as Util
from CDDPM.models import create_model, define_network, define_loss, define_metric

class TyphoonDataset(Dataset):
    '''
    Define a custom dataset class for the typhoon dataset
    '''
    def __init__(self, dataset, times, transform=None, augment=False, seed=42):
        super().__init__()
        self.dataset = dataset
        self.times = times 

        self.image_data = self.dataset['image_data'].values
        self.u10 = self.dataset['u10'].values
        self.v10 = self.dataset['v10'].values
        self.sp = self.dataset['sp'].values
        self.t2m = self.dataset['t2m'].values

        self.transform = transform
        self.augment = augment
        self.seed = seed

        # Resize the image data to 64x64
        self.image_data_resized = np.array([np.array(Image.fromarray(img).resize((64, 64), Image.BICUBIC)) for img in self.image_data])
        self.u10_resized = np.array([np.array(Image.fromarray(u).resize((64, 64), Image.BICUBIC)) for u in self.u10])
        self.v10_resized = np.array([np.array(Image.fromarray(v).resize((64, 64), Image.BICUBIC)) for v in self.v10])
        self.sp_resized = np.array([np.array(Image.fromarray(sp).resize((64, 64), Image.BICUBIC)) for sp in self.sp])
        self.t2m_resized = np.array([np.array(Image.fromarray(t).resize((64, 64), Image.BICUBIC)) for t in self.t2m])

        self.sp_min = np.min(self.sp_resized)
        self.sp_max = np.max(self.sp_resized)
        self.t2m_min = np.min(self.t2m_resized)
        self.t2m_max = np.max(self.t2m_resized)

        self.image_min = np.min(self.image_data_resized)
        self.image_max = np.max(self.image_data_resized)
        self.u10_min = np.min(self.u10_resized)
        self.u10_max = np.max(self.u10_resized)
        self.v10_min = np.min(self.v10_resized)
        self.v10_max = np.max(self.v10_resized)

    def __len__(self):
        return self.image_data.shape[0]

    def __getitem__(self, index):
        img = self.image_data_resized[index]
        # normarlize the image data
        img_normalized = (img - self.image_min) / (self.image_max - self.image_min)
        img_normalized = torch.tensor(img_normalized, dtype=torch.float32).unsqueeze(0)

        img_64 = self.image_data_resized[index]
        img_64 = (img_64 - self.image_min) / (self.image_max - self.image_min)
        img_64 = torch.tensor(img_64, dtype=torch.float32).unsqueeze(0)

        u10 = self.u10_resized[index]
        u10 = (u10 - self.u10_min) / (self.u10_max - self.u10_min)
        u10 = torch.tensor(u10, dtype=torch.float32).unsqueeze(0)

        v10 = self.v10_resized[index]
        v10 = (v10 - self.v10_min) / (self.v10_max - self.v10_min)
        v10 = torch.tensor(v10, dtype=torch.float32).unsqueeze(0)

        sp = self.sp_resized[index]
        sp = (sp - self.sp_min) / (self.sp_max - self.sp_min)
        sp = torch.tensor(sp, dtype=torch.float32).unsqueeze(0)

        t2m = self.t2m_resized[index]
        t2m = (t2m - self.t2m_min) / (self.t2m_max - self.t2m_min)
        t2m = torch.tensor(t2m, dtype=torch.float32).unsqueeze(0)

        img_64 = torch.cat((img_64, img_64, img_64, img_64), dim=0)

        return img_normalized, img_64, u10, v10, sp, t2m

def load_csv_data(csv_path):
    '''
    Load the CSV data containing the typhoon information from IBTrACS   
    '''
    df = pd.read_csv(csv_path)
    times = pd.to_datetime(df['ISO_TIME'])
    latitudes = df['LAT'].values
    longitudes = df['LON'].values
    pressures_dict = {
        "TOKYO_PRES": df['TOKYO_PRES'].values,
        "CMA_PRES": df['CMA_PRES'].values,
        "KMA_PRES": df['KMA_PRES'].values,
        "REUNION_PRES": df['REUNION_PRES'].values,
        "BOM_PRES": df['BOM_PRES'].values
    }
    return times, latitudes, longitudes, pressures_dict

def load_all_zarr_files(directory):
    '''
    Load all the zarr files in the specified directory
    '''
    datasets = []
    time_stamps = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.endswith(".zarr"):
                zarr_path = os.path.join(root, dir)
                print(f"Found zarr directory: {zarr_path}")
                sys.stdout.flush()
                ds = xr.open_zarr(zarr_path)
                datasets.append(ds)
                time_stamps.extend(ds['time'].values)
    if not datasets:
        raise ValueError("No zarr files found in the specified directory。")
    combined_dataset = xr.concat(datasets, dim="time")
    return combined_dataset, time_stamps

def set_seed(seed):
    '''
    Set the seed for reproducibility
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_con_diffusion_model(model, model_path, device):
    '''
    Load the diffusion model from the specified path
    '''
    state_dict = torch.load(model_path, map_location=device)
    model.netG.load_state_dict(state_dict, strict=False)
    model.netG.to(device)
    model.netG.set_new_noise_schedule(device=device, phase='test')
    model.netG.num_timesteps = model.netG.num_timesteps if hasattr(model.netG, 'num_timesteps') else 1000
    if 'ema' in state_dict:
        ema_model = EMA(model, beta=0.995)
        ema_model.load_state_dict(state_dict['ema'])
        ema_model.to(device)
        model.ema = ema_model
    return model

def evaluate_model(test_loader, model, device, num_timesteps=1000):
    '''
    Evaluate the model on the test set
    '''
    all_predictions = []
    for batch_idx, data in enumerate(test_loader):
        img, img_64, u10, v10, sp, t2m = data
        inputs = img_64.to(device)

        model_class_name = model.__class__.__name__

        if model_class_name == 'GaussianDiffusion':
            model.ema.ema_model.eval()
            with torch.inference_mode():
                t = torch.randint(0, num_timesteps, (inputs.size(0),), device=device).long()
                noise = torch.randn_like(inputs)
                noisy_input = model.q_sample(inputs, t, noise)
                pred_img = noisy_input.clone()
                for time_step in reversed(range(model.num_timesteps)):
                    pred_img, _ = model.p_sample(pred_img, time_step)
                outputs = pred_img
        elif model_class_name == 'Palette':
            netG = model.netG.module if hasattr(model.netG, 'module') else model.netG
            netG.eval()
            with torch.inference_mode():
                cond_image = inputs
                outputs, _ = netG.restoration(cond_image, sample_num=8)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)

        batch_times = test_loader.dataset.dataset['time'][batch_idx * test_loader.batch_size: (batch_idx + 1) * test_loader.batch_size]  
        for i, batch_time in enumerate(batch_times):
            single_time_predictions = {
                'time': batch_time, 
                'predictions': [outputs[i].cpu()], 
                'model': model_class_name
            }
            all_predictions.append(single_time_predictions)

    return all_predictions

def denormalize(tensor, min_val, max_val):
    '''
    Denormalize the tensor
    '''
    return tensor * (max_val - min_val) + min_val

def find_nearest_grid_point(lat, lon, lat_start, lat_end, lon_start, lon_end, lat_resolution, lon_resolution):
    '''
    Find the nearest grid point to the given latitude and longitude
    '''
    lat_idx = int((lat - lat_start) / (lat_end - lat_start) * (lat_resolution - 1))
    lon_idx = int((lon - lon_start) / (lon_end - lon_start) * (lon_resolution - 1))
    
    return lat_idx, lon_idx

def compare_predictions_with_csv(all_predictions, csv_times, csv_lats, csv_lons, csv_pressures_dict, model_names, denorm_dataset):
    '''
    Compare the model predictions with the CSV data
    '''
    lat_start, lat_end = 18.9037, 28.9037
    lon_start, lon_end = 116.0794, 126.0794
    lat_resolution, lon_resolution = 64, 64

    csv_time_dict = {
        pd.to_datetime(csv_time).floor('H'): (lat, lon, pressures)
        for csv_time, lat, lon, pressures in zip(csv_times, csv_lats, csv_lons, zip(*csv_pressures_dict.values()))
        if not pd.isnull(lat) and not pd.isnull(lon)  
    }

    compare_results = {model_name: {key: [] for key in csv_pressures_dict.keys()} for model_name in model_names}

    for model_name in model_names:
        model_predictions = all_predictions.get(model_name, [])
        print(f"Processing model: {model_name}, Found {len(model_predictions)} predictions")

        for prediction in model_predictions:
            time_array = prediction['time'].values
            time = pd.to_datetime(time_array).floor('H')

            if time in csv_time_dict:
                lat, lon, actual_pressures = csv_time_dict[time]

                if (lat_start <= lat <= lat_end) and (lon_start <= lon <= lon_end):
                    try:
                        u10_normalized = prediction['predictions'][0][0].numpy()
                        v10_normalized = prediction['predictions'][0][1].numpy()
                        sp_normalized = prediction['predictions'][0][2].numpy()

                        u10 = denormalize(u10_normalized, denorm_dataset.u10_min, denorm_dataset.u10_max)
                        v10 = denormalize(v10_normalized, denorm_dataset.v10_min, denorm_dataset.v10_max)
                        sp = denormalize(sp_normalized, denorm_dataset.sp_min, denorm_dataset.sp_max)

                        lat_idx, lon_idx = find_nearest_grid_point(lat, lon, lat_start, lat_end, lon_start, lon_end, lat_resolution, lon_resolution)

                        central_pressure = sp[lat_idx, lon_idx]

                        for pressure_source, actual_pressure in zip(csv_pressures_dict.keys(), actual_pressures):
                            if not pd.isnull(actual_pressure):
                                compare_results[model_name][pressure_source].append((time, central_pressure, actual_pressure))

                    except IndexError as e:
                        print(f"IndexError: {e}")
                        print(f"Problematic prediction data: {prediction['predictions']}")
                    except Exception as e:
                        print(f"An error occurred: {e}")

    for model_name in model_names:
        for pressure_source in csv_pressures_dict.keys():
            filtered_results = [(time, pred_pressure, actual_pressure) for time, pred_pressure, actual_pressure in compare_results[model_name][pressure_source] if isinstance(pred_pressure, (int, float)) and isinstance(actual_pressure, (int, float))]
            pred_vs_actual = np.array(filtered_results, dtype=object) 

            if len(pred_vs_actual) > 0:
                avg_rmse_pressure = np.sqrt(mean_squared_error(pred_vs_actual[:, 2].astype(float), pred_vs_actual[:, 1].astype(float)))
                avg_mae_pressure = mean_absolute_error(pred_vs_actual[:, 2].astype(float), pred_vs_actual[:, 1].astype(float))
                print(f"{model_name} - {pressure_source} - Average Pressure RMSE: {avg_rmse_pressure}, Average Pressure MAE: {avg_mae_pressure}")

    return compare_results

def main():
    set_seed(42)

    base_directory = '/vol/bitbucket/zl1823/Typhoon-forecasting/Case_study/muifa'
    denorm_directory = '/vol/bitbucket/zl1823/Typhoon-forecasting/dataset/pre_processed_dataset/ERA5_without_nan_taiwan'

    combined_dataset, _ = load_all_zarr_files(base_directory)
    denorm_dataset, _ = load_all_zarr_files(denorm_directory)

    csv_path = '/vol/bitbucket/zl1823/Typhoon-forecasting/Case_study/MUIFA.csv'
    csv_times, csv_lats, csv_lons, csv_pressures_dict = load_csv_data(csv_path)

    # Create the dataset
    dataset = TyphoonDataset(combined_dataset, csv_times)
    denorm_dataset = TyphoonDataset(denorm_dataset, csv_times)

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader([], batch_size=32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/vol/bitbucket/zl1823/Typhoon-forecasting/CDDPM/config/typhoon.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    args = parser.parse_args()
    opt = Praser.parse(args)

    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    trained_models = {
        # 'CNN': ConditionCNN(),
        # 'SENet': SENet(),
        # 'DDPM': GaussianDiffusion(Unet(dim=64, channels=4), image_size=(64, 64)),
        'CDDPM': create_model(
            opt = opt,
            networks = networks,
            phase_loader = train_loader,
            test_loader = test_loader,
            losses = losses,
            metrics = metrics,
            logger = phase_logger,
            writer = phase_writer
        )
    }

    model_files = {
        # 'CNN': '/vol/bitbucket/zl1823/Typhoon-forecasting/pyphoon2/cnn_normal_model.pth',
        # 'SENet': '/vol/bitbucket/zl1823/Typhoon-forecasting/pyphoon2/cnn_se_normal_model.pth',
        # 'DDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/pyphoon2/results/model-30.pt',
        'CDDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/CDDPM/experiments/train_typhoon_forecasting_240803_211802/checkpoint/3350_Network.pth',
    }

    # trained_models['CNN'] = load_cnn_model(trained_models['CNN'], model_files['CNN'], device)
    # trained_models['SENet'] = load_cnn_model(trained_models['SENet'], model_files['SENet'], device)
    # trained_models['DDPM'] = load_diffusion_model(trained_models['DDPM'], model_files['DDPM'], device)
    trained_models['CDDPM'] = load_con_diffusion_model(trained_models['CDDPM'], model_files['CDDPM'], device)

    evaluation_results = {}

    for model_name, model in trained_models.items():
        predictions = evaluate_model(test_loader, model, device, num_timesteps=1000)
        evaluation_results[model_name] = predictions

    compare_results = compare_predictions_with_csv(
        evaluation_results,
        csv_times,
        csv_lats,
        csv_lons,
        csv_pressures_dict,
        list(trained_models.keys()),
        denorm_dataset
    )

    results_folder = "comparison_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for model_name, pressure_source in compare_results.items():
        for source, results in pressure_source.items():
            output_file = os.path.join(results_folder, f"{model_name}_{source}_comparison_results.csv")
            df = pd.DataFrame(results, columns=["Time", "Predicted Pressure", "Actual Pressure"]) 
            df.to_csv(output_file, index=False)
            print(f"Saved comparison results for {model_name} - {source} to {output_file}")

if __name__ == '__main__':
    main()
