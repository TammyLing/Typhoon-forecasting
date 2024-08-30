import torch
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
import os
import xarray as xr
import random
import lpips
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.stats import entropy
from PIL import Image
import sys
import matplotlib.pyplot as plt
import logging
from ema_pytorch import EMA
from torchvision import transforms as T, utils
from CDDPM.models.model import Palette
from CDDPM.models.network import Network
import argparse
from scipy.linalg import sqrtm
from torchvision import models
from pixelmatch.contrib.PIL import pixelmatch
import shutil
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 导入模型
from CNN.cnn import CNN
from SENet.senet import SENet
from DDPM.diffusion import GaussianDiffusion, Unet

from CDDPM.core.logger import VisualWriter, InfoLogger
import CDDPM.core.praser as Praser
import CDDPM.core.util as Util
from CDDPM.data import define_dataloader
from CDDPM.models import create_model, define_network, define_loss, define_metric
from pytorch_fid import fid_score

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def random_flip_and_rotate(tensor, seed):
    '''
    Randomly flip and rotate the tensor. 
    '''
    random.seed(seed)
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[2])
    if random.random() > 0.5:
        tensor = torch.flip(tensor, dims=[1])
    rotations = random.choice([0, 1, 2, 3])
    tensor = torch.rot90(tensor, k=rotations, dims=[1, 2])
    return tensor

def add_noise(tensors, noise_level=0.01):
    '''
    Add Gaussian noise to the tensors.
    '''
    noisy_tensors = [tensor + noise_level * torch.randn_like(tensor) for tensor in tensors]
    return noisy_tensors

def apply_smoothing(tensors, kernel_size=3):
    '''
    Apply Gaussian smoothing to the tensors.
    '''
    smoothed_tensors = [T.GaussianBlur(kernel_size=kernel_size)(tensor) for tensor in tensors]
    return smoothed_tensors

def adjust_contrast(tensors, contrast_factor=0.1):
    '''
    Adjust the contrast of the tensors.
    '''
    contrast_tensors = []
    for tensor in tensors:
        channels = tensor.split(1, dim=0)
        adjusted_channels = [T.functional.adjust_contrast(ch, contrast_factor=random.uniform(1-contrast_factor, 1+contrast_factor)) for ch in channels]
        contrast_tensor = torch.cat(adjusted_channels, dim=0)
        contrast_tensors.append(contrast_tensor)
    return contrast_tensors

def conservative_augmentation(tensors, seed):
    '''
    Apply conservative augmentation to the tensors.
    '''
    random.seed(seed)
    torch.manual_seed(seed)
    
    if random.random() > 0.5:
        tensors = add_noise(tensors, noise_level=0.01)
    
    if random.random() > 0.5:
        tensors = apply_smoothing(tensors, kernel_size=3)
    
    if random.random() > 0.5:
        tensors = adjust_contrast(tensors, contrast_factor=0.1)
    
    return tensors

class TyphoonDataset(Dataset):
    '''
    Dataset class for the typhoon dataset.
    '''
    def __init__(self, dataset, transform=None, augment=False, seed=42):
        super().__init__()
        self.dataset = dataset

        self.image_data = self.dataset['image_data'].values
        self.u10 = self.dataset['u10'].values
        self.v10 = self.dataset['v10'].values
        self.sp = self.dataset['sp'].values
        self.t2m = self.dataset['t2m'].values

        self.transform = transform
        self.augment = augment
        self.seed = seed

        # Resize the images to 64x64
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

        # normalize the data
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

        original_u10 = self.u10[index].astype(np.float32)
        original_u10 = (original_u10 - self.u10_min) / (self.u10_max - self.u10_min)
        original_u10 = torch.tensor(original_u10, dtype=torch.float32).unsqueeze(0)

        original_v10 = self.v10[index].astype(np.float32)
        original_v10 = (original_v10 - self.v10_min) / (self.v10_max - self.v10_min)
        original_v10 = torch.tensor(original_v10, dtype=torch.float32).unsqueeze(0)

        original_sp = self.sp[index].astype(np.float32)
        original_sp = (original_sp - self.sp_min) / (self.sp_max - self.sp_min)
        original_sp = torch.tensor(original_sp, dtype=torch.float32).unsqueeze(0)

        original_t2m = self.t2m[index].astype(np.float32)
        original_t2m = (original_t2m - self.t2m_min) / (self.t2m_max - self.t2m_min)
        original_t2m = torch.tensor(original_t2m, dtype=torch.float32).unsqueeze(0)

        img_64 = torch.cat((img_64, img_64, img_64, img_64), dim=0)

        # data augmentation
        if self.augment:
            tensors = [img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m]
            tensors = conservative_augmentation(tensors, self.seed + index)
            img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = tensors

        return img_normalized, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m

def load_all_zarr_files(directory):
    '''
    Load all zarr files in the specified directory.
    
    Args:
    - directory: The directory containing the zarr files.
    
    Returns:
    - combined_dataset: The combined dataset containing all the zarr files.
    '''
    datasets = []
    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if dir.endswith(".zarr"):
                zarr_path = os.path.join(root, dir)
                print(f"Found zarr directory: {zarr_path}")
                sys.stdout.flush()
                ds = xr.open_zarr(zarr_path)
                datasets.append(ds)
    if not datasets:
        raise ValueError("No zarr files found in the specified directory。")
    combined_dataset = xr.concat(datasets, dim="time")
    return combined_dataset

def set_seed(seed):
    '''
    Set the seed for reproducibility.
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_cnn_model(model, model_path, device):
    '''
    Load the CNN model from the specified path.
    '''
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model

def load_diffusion_model(model, model_path, device):
    '''
    Load the diffusion model from the specified path.
    '''
    state_dict = torch.load(model_path, map_location=device)
    model_state_dict = state_dict['model']
    model.load_state_dict(model_state_dict, strict=False)
    model.to(device)
    if 'ema' in state_dict:
        print("Loading EMA...")
        ema_model = EMA(model, beta=0.995)
        ema_model.load_state_dict(state_dict['ema'])
        ema_model.to(device)
        model.ema = ema_model
    else:
        print("EMA not found in state_dict.")
    return model

def load_con_diffusion_model(model, model_path, device):
    '''
    Load the conditional diffusion model from the specified path.
    '''
    state_dict = torch.load(model_path, map_location=device)
    model.netG.load_state_dict(state_dict, strict=False)
    model.netG.to(device)
    model.netG.set_new_noise_schedule(device=device, phase='test')
    print(f"num_timesteps after setting noise schedule: {model.netG.num_timesteps}")
    model.netG.num_timesteps = model.netG.num_timesteps if hasattr(model.netG, 'num_timesteps') else 1000
    if 'ema' in state_dict:
        print("Loading EMA...")
        ema_model = EMA(model, beta=0.995)
        ema_model.load_state_dict(state_dict['ema'])
        ema_model.to(device)
        model.ema = ema_model
    else:
        print("EMA not found in state_dict.")
    return model

def save_images(images, folder, prefix):
    '''
    Save the images to the specified folder.
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, image in enumerate(images):
        utils.save_image(image, os.path.join(folder, f"{prefix}_{i}.png"))

def evaluate_model(test_loader, model, model_name, device, num_timesteps=1000):
    '''
    Evaluate the model on the test set.
    
    Args:
    - test_loader: The DataLoader for the test set.
    - model: The model to evaluate.
    - model_name: The name of the model.
    - device: The device to use.
    - num_timesteps: The number of timesteps for the diffusion model.
    '''
    lpips_loss = lpips.LPIPS(net='alex').to(device)

    # Initialize lists to store the evaluation metrics
    kl_div_list = {i: [] for i in range(4)}
    rmse_list = {i: [] for i in range(4)}
    mae_list = {i: [] for i in range(4)}
    psnr_list = {i: [] for i in range(4)}
    ssim_list = {i: [] for i in range(4)}
    fid_list = {i: [] for i in range(4)}
    lpips_list = {i: [] for i in range(4)}
    all_predictions = []
    diff_images = {i: [] for i in range(4)}

    sum_kl_div = {i: 0 for i in range(4)}
    sum_rmse = {i: 0 for i in range(4)}
    sum_mae = {i: 0 for i in range(4)}
    sum_psnr = {i: 0 for i in range(4)}
    sum_ssim = {i: 0 for i in range(4)}
    sum_lpips = {i: 0 for i in range(4)}
    sum_fid = {i: 0 for i in range(4)}
    count = 0

    all_fid_scores = {i: [] for i in range(4)}

    sum_mismatch = {i: 0 for i in range(4)}
    mismatch_list = {i: [] for i in range(4)}

    for batch_idx, data in enumerate(test_loader):
        img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
        targets = [u10, v10, sp, t2m]

        inputs = img_64.to(device)
        targets = [t.to(device) for t in targets]
        model_class_name = model.__class__.__name__

        # Generate predictions
        if model_class_name == 'GaussianDiffusion':
            model.ema.ema_model.eval()
            with torch.inference_mode():
                t = torch.randint(0, num_timesteps, (inputs.size(0),), device=device).long()
                noise = torch.randn_like(inputs)
                noisy_input = model.q_sample(inputs, t, noise)
                pred_img = noisy_input.clone()
                for time_step in reversed(range(model.num_timesteps)):
                    pred_img, _ = model.p_sample(pred_img, time_step)
                outputs = pred_img.split(1, dim=1)
        elif model_class_name == 'Palette':
            if hasattr(model.netG, 'module'):
                netG = model.netG.module
            else:
                netG = model.netG
            netG.eval()
            with torch.inference_mode():
                cond_image = inputs
                outputs, _ = netG.restoration(cond_image, sample_num=8)
                outputs = outputs.split(1, dim=1)
        else:
            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                outputs = outputs.split(1, dim=1)

        batch_size = inputs.size(0)
        batch_predictions = [[] for _ in range(batch_size)]
        
        for j in range(4):  
            real_images_folder = f"results/{model_name}/real_images/variable_{j}/batch_{batch_idx}"
            generated_images_folder = f"results/{model_name}/generated_images/variable_{j}/batch_{batch_idx}"
            
            if os.path.exists(real_images_folder):
                shutil.rmtree(real_images_folder)
            os.makedirs(real_images_folder)
            if os.path.exists(generated_images_folder):
                shutil.rmtree(generated_images_folder)
            os.makedirs(generated_images_folder)
            for i in range(batch_size):
                output_np = outputs[j][i].cpu().numpy().squeeze()
                target_np = targets[j][i].cpu().numpy().squeeze()

                # Save images
                save_images([outputs[j][i]], generated_images_folder, f"generated_{batch_idx}_{i}_{j}")
                save_images([targets[j][i]], real_images_folder, f"real_{batch_idx}_{i}_{j}")

                kl_div = entropy(target_np.flatten(), output_np.flatten())
                kl_div_list[j].append(kl_div)
                sum_kl_div[j] += kl_div

                rmse = np.sqrt(mean_squared_error(target_np.flatten(), output_np.flatten()))
                rmse_list[j].append(rmse)
                sum_rmse[j] += rmse

                mae = mean_absolute_error(target_np.flatten(), output_np.flatten())
                mae_list[j].append(mae)
                sum_mae[j] += mae

                psnr = peak_signal_noise_ratio(target_np, output_np, data_range=1.0)
                psnr_list[j].append(psnr)
                sum_psnr[j] += psnr

                ssim = structural_similarity(target_np, output_np, data_range=1.0)
                ssim_list[j].append(ssim)
                sum_ssim[j] += ssim

                target_tensor = torch.tensor(target_np).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
                output_tensor = torch.tensor(output_np).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)

                lpips_val = lpips_loss(target_tensor, output_tensor).mean().item()
                lpips_list[j].append(lpips_val)
                sum_lpips[j] += lpips_val

                batch_predictions[i].append(outputs[j][i].cpu())
                
                # Calculate pixel difference
                mismatch, diff_img = calculate_pixel_difference(target_np, output_np)
                diff_images[j].append(diff_img)

                mismatch_list[j].append(mismatch)
                sum_mismatch[j] += mismatch

            # Calculate FID
            fid_score_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder], 32, device, 2048)
            fid_list[j].extend([fid_score_value] * batch_size)
            all_fid_scores[j].append(fid_score_value)

        all_predictions.append(batch_predictions)

        count += batch_size

    avg_kl_div = {i: sum_kl_div[i] / count for i in range(4)}
    avg_rmse = {i: sum_rmse[i] / count for i in range(4)}
    avg_mae = {i: sum_mae[i] / count for i in range(4)}
    avg_psnr = {i: sum_psnr[i] / count for i in range(4)}
    avg_ssim = {i: sum_ssim[i] / count for i in range(4)}
    avg_lpips = {i: sum_lpips[i] / count for i in range(4)}
    avg_fid = {i: np.mean(all_fid_scores[i]) for i in range(4)}

    print("\nAverage values:")
    for i in range(4):
        print(f"Variable {i}: KL Div = {avg_kl_div[i]}, RMSE = {avg_rmse[i]}, MAE = {avg_mae[i]}, PSNR = {avg_psnr[i]}, SSIM = {avg_ssim[i]}, LPIPS = {avg_lpips[i]}, FID = {avg_fid[i]}")

    return kl_div_list, rmse_list, mae_list, psnr_list, ssim_list, fid_list, lpips_list, mismatch_list, all_predictions, diff_images

def denormalize(tensor, min_val, max_val):
    '''
    Denormalize the tensor.
    '''
    return tensor * (max_val - min_val) + min_val


def plot_results(input_image, targets, predictions, variable_names, model_names, index, results_folder):
    '''
    Plot the results for the input image, true values, and predictions.
    '''
    num_variables = len(variable_names)
    num_models = len(model_names)
    
    fig = plt.figure(figsize=(24, 16))
    gs = gridspec.GridSpec(nrows=num_variables, ncols=num_models + 3, figure=fig, width_ratios=[1, 1] + [1] * num_models + [0.05])

    lon_start, lon_end = 116.0794, 126.0794
    lat_start, lat_end = 18.9037, 28.9037
    lon, lat = np.meshgrid(np.linspace(lon_start, lon_end, 64), np.linspace(lat_start, lat_end, 64))

    for i, var_name in enumerate(variable_names):
        # Determine the vmin and vmax across all plots in the row
        vmin = min([targets[i].min()] + [predictions[model_name][i].min() for model_name in model_names])
        vmax = max([targets[i].max()] + [predictions[model_name][i].max() for model_name in model_names])

        # Plot input image
        ax = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax.imshow(input_image[i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='gray')
        ax.add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
        ax.coastlines()
        ax.set_title('Input (img_64)')

        # Plot true value
        ax = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
        im = ax.imshow(targets[i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
        ax.coastlines()
        ax.set_title(f'True {var_name}')

        # Plot predictions for each model
        for j, model_name in enumerate(model_names):
            ax = fig.add_subplot(gs[i, j + 2], projection=ccrs.PlateCarree())
            im = ax.imshow(predictions[model_name][i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='coolwarm', vmin=vmin, vmax=vmax)
            ax.add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
            ax.coastlines()
            ax.set_title(f'{model_name} Prediction')

        # Add a color bar 
        cbar_ax = fig.add_subplot(gs[i, -1])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical') 
        cbar.set_label('Value')

    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.01)
    
    plt.savefig(os.path.join(results_folder, f'results_{index}.png'))
    plt.close(fig)


def calculate_pixel_difference(img_a, img_b):
    '''
    Calculate the pixel difference between two images.
    '''
    img_a_pil = Image.fromarray((img_a * 255).astype(np.uint8))
    img_b_pil = Image.fromarray((img_b * 255).astype(np.uint8))
    img_diff = Image.new("RGBA", img_a_pil.size)
    mismatch = pixelmatch(img_a_pil, img_b_pil, img_diff, includeAA=True)
    return mismatch, img_diff

def plot_saved_images_and_differences(input_image, targets, differences, variable_names, model_names, index, results_folder):
    '''
    Plot the saved images and differences.
    '''
    num_variables = len(variable_names)
    num_models = len(model_names)
    
    fig, axs = plt.subplots(num_variables, num_models + 2, figsize=(18, 12), subplot_kw={'projection': ccrs.PlateCarree()})

    lon_start, lon_end = 116.0794, 126.0794
    lat_start, lat_end = 18.9037, 28.9037
    lon = np.linspace(lon_start, lon_end, 64)
    lat = np.linspace(lat_start, lat_end, 64)
    lon, lat = np.meshgrid(lon, lat)

    for i, var_name in enumerate(variable_names):
        axs[i, 0].imshow(input_image[i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='gray')
        axs[i, 0].add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
        axs[i, 0].coastlines()
        axs[i, 0].set_title(f'Input (img_64)')

        im = axs[i, 1].imshow(targets[i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='coolwarm')
        axs[i, 1].add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
        axs[i, 1].coastlines()
        axs[i, 1].set_title(f'True {var_name}')

        for j, model_name in enumerate(model_names):
            diff_img = differences[model_name][i]

            if isinstance(diff_img, Image.Image):
                diff_img = T.functional.to_tensor(diff_img)

            if diff_img.shape[0] == 4:
                diff_img = diff_img.permute(1, 2, 0)

            axs[i, j + 2].imshow(diff_img.cpu().numpy(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='coolwarm')
            axs[i, j + 2].add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
            axs[i, j + 2].coastlines()
            axs[i, j + 2].set_title(f'{model_name} Difference')

    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.01) 
    
    plt.savefig(os.path.join(results_folder, f'difference_{index}.png'))
    plt.close(fig)


def main():
    set_seed(42)

    results_folder = "test_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    print("Loading dataset...")
    sys.stdout.flush()
    base_directory = '/vol/bitbucket/zl1823/Typhoon-forecasting/dataset/pre_processed_dataset/ERA5_without_nan_taiwan'
    combined_dataset = load_all_zarr_files(base_directory)

    print("Creating datasets...")
    sys.stdout.flush()
    dataset = TyphoonDataset(combined_dataset)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    augmentation_ratio = 0.5
    augmented_size = int(train_size * augmentation_ratio)
    train_indices = train_dataset.indices
    train_combined_data = combined_dataset.isel(time=train_indices)
    augmented_dataset = TyphoonDataset(train_combined_data, augment=True)
    augmented_dataset = random_split(augmented_dataset, [augmented_size, len(augmented_dataset) - augmented_size], generator=torch.Generator().manual_seed(42))[0]

    train_dataset = ConcatDataset([train_dataset, augmented_dataset])

    print("Creating dataloaders...")
    sys.stdout.flush()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ''' parser configs '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='/vol/bitbucket/zl1823/Typhoon-forecasting/CDDPM/config/typhoon.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train','test'], help='Run train or test', default='train')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)

    args = parser.parse_args()
    opt = Praser.parse(args)

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]


    trained_models = {
        'CNN': CNN(),
        'SENet': SENet(),
        'DDPM': GaussianDiffusion(Unet(dim=64, channels=4), image_size=(64, 64)),
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
        'CNN': '/vol/bitbucket/zl1823/Typhoon-forecasting/CNN/cnn_model.pth',
        'SENet': '/vol/bitbucket/zl1823/Typhoon-forecasting/SENet/senet_normal_model.pth',
        'DDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/DDPM/model-30.pt',
        'CDDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/CDDPM/experiments/train_typhoon_forecasting_240803_211802/checkpoint/3350_Network.pth',
    }

    # Load models
    trained_models['CNN'] = load_cnn_model(trained_models['CNN'], model_files['CNN'], device)
    trained_models['SENet'] = load_cnn_model(trained_models['SENet'], model_files['SENet'], device)
    trained_models['DDPM'] = load_diffusion_model(trained_models['DDPM'], model_files['DDPM'], device)
    trained_models['CDDPM'] = load_con_diffusion_model(trained_models['CDDPM'], model_files['CDDPM'], device)

    evaluation_results = {}

    # Initialize the Inception model
    inception_model = models.inception_v3(pretrained=True)
    inception_model.fc = nn.Identity() 
    inception_model.eval()
    inception_model.to(device)

    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name} model...")
        sys.stdout.flush()
        kl_div_list, rmse_list, mae_list, psnr_list, ssim_list, lpips_list, fid_list, mismatch_list, predictions, diff_images = evaluate_model(test_loader, model, model_name, device, num_timesteps=1000)

        evaluation_results[model_name] = {
            'KL Div': kl_div_list,
            'RMSE': rmse_list,
            'MAE': mae_list,
            'PSNR': psnr_list,
            'SSIM': ssim_list,
            'FID': fid_list,
            'LPIPS': lpips_list,
            'mismatch': mismatch_list,
            'predictions': predictions,
            'diff_images': diff_images
        }

    # Visualization of model outputs
    print("Visualizing model outputs...")
    sys.stdout.flush()
    variable_names = ['u10', 'v10', 'sp', 't2m']
    model_names = list(trained_models.keys())

    input_images, target_images = [], []

    # Save results to csv
    avg_results = {model_name: {i: {} for i in range(4)} for model_name in evaluation_results.keys()}
    all_models_results = []

    for model_name, results in evaluation_results.items():
        total_kl_div = 0
        total_rmse = 0
        total_mae = 0
        total_psnr = 0
        total_ssim = 0
        total_fid = 0
        total_lpips = 0
        total_mismatch = 0
        
        for i in range(4):
            df = pd.DataFrame({
                'KL Div': results['KL Div'][i],
                'RMSE': results['RMSE'][i],
                'MAE': results['MAE'][i],
                'PSNR': results['PSNR'][i],
                'SSIM': results['SSIM'][i],
                'FID': results['FID'][i],
                'LPIPS': results['LPIPS'][i],
                'mismatch':results['mismatch'][i]
            })
            avg_results[model_name][i] = {
                'KL Div': df['KL Div'].mean(),
                'RMSE': df['RMSE'].mean(),
                'MAE': df['MAE'].mean(),
                'PSNR': df['PSNR'].mean(),
                'SSIM': df['SSIM'].mean(),
                'FID': df['FID'].mean(),
                'LPIPS': df['LPIPS'].mean(),
                'mismatch':df['mismatch'].mean()
            }
            
            # Accumulate total results
            total_kl_div += df['KL Div'].mean()
            total_rmse += df['RMSE'].mean()
            total_mae += df['MAE'].mean()
            total_psnr += df['PSNR'].mean()
            total_ssim += df['SSIM'].mean()
            total_fid += df['FID'].mean()
            total_lpips += df['LPIPS'].mean()
            total_mismatch += df['mismatch'].mean()

        # Add total sum results as an additional row to the average results
        avg_results[model_name]['Average'] = {
            'KL Div': total_kl_div/4,
            'RMSE': total_rmse/4,
            'MAE': total_mae/4,
            'PSNR': total_psnr/4,
            'SSIM': total_ssim/4,
            'FID': total_fid/4,
            'LPIPS': total_lpips/4,
            'mismatch': total_mismatch/4
        }

        # Convert the results to DataFrame for this model
        model_df = pd.DataFrame(avg_results[model_name]).T
        model_df = model_df.round(3)
        model_df['Model'] = model_name 
        all_models_results.append(model_df)

    # Combine all model results into a single DataFrame
    combined_df = pd.concat(all_models_results)

    # Save combined results to a single CSV file
    combined_df.to_csv('evaluation.csv')
    print(f'\nAll models evaluation results saved to evaluation.csv:')
    print(combined_df)
    sys.stdout.flush()


    for model_name, results in avg_results.items():
        avg_df = pd.DataFrame(results).T
        avg_df = avg_df.round(3)
        avg_df.to_csv(f'{model_name}_average_evaluation_results.csv')
        print(f'\n{model_name} average evaluation results:')
        print(avg_df)
        sys.stdout.flush()

    for index, data in enumerate(test_loader):
        img, img_64, u10, v10, sp, t2m, original_u10, original_v10, original_sp, original_t2m = data
        inputs = img_64.to(device)
        targets = [u10.to(device), v10.to(device), sp.to(device), t2m.to(device)]

        for i in range(inputs.size(0)):
            denorm_targets = [
                denormalize(u10[i], dataset.u10_min, dataset.u10_max),
                denormalize(v10[i], dataset.v10_min, dataset.v10_max),
                denormalize(sp[i], dataset.sp_min, dataset.sp_max),
                denormalize(t2m[i], dataset.t2m_min, dataset.t2m_max)
            ]     

            denorm_predictions = {
                model_name: [
                    denormalize(evaluation_results[model_name]['predictions'][index][i][j], dataset.u10_min if j == 0 else dataset.v10_min if j == 1 else dataset.sp_min if j == 2 else dataset.t2m_min, dataset.u10_max if j == 0 else dataset.v10_max if j == 1 else dataset.sp_max if j == 2 else dataset.t2m_max)
                    for j in range(4)
                ] for model_name in model_names
            }

            single_input = inputs[i]
            single_targets = [target for target in denorm_targets]

            single_predictions = {}
            for model_name in model_names:
                single_predictions[model_name] = denorm_predictions[model_name]
            
            single_difference = {
            model_name: [evaluation_results[model_name]['diff_images'][j][i] for j in range(4)]
            for model_name in model_names
        }

            plot_results(single_input.cpu(), single_targets, single_predictions, variable_names, model_names, f"{index}_{i}", results_folder)

            plot_saved_images_and_differences(single_input.cpu(), single_targets, single_difference, variable_names, model_names, f"{index}_{i}", results_folder)

    print("Finish visualization")
if __name__ == '__main__':
    main()
