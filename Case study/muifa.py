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

from CDDPM.core.logger import VisualWriter, InfoLogger
import CDDPM.core.praser as Praser
import CDDPM.core.util as Util
from CDDPM.data import define_dataloader
from CDDPM.models import create_model, define_network, define_loss, define_metric
from pytorch_fid import fid_score

# Initialize the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class TyphoonDataset(Dataset):
    '''
    Define a custom dataset class for the typhoon dataset
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

        # Resize all images to 64x64
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

        img_64 = torch.cat((img_64, img_64, img_64, img_64), dim=0)

        return img_normalized, img_64, u10, v10, sp, t2m

def load_all_zarr_files(directory):
    '''
    Load all zarr files in the specified directory and concatenate them into a single xarray dataset
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
    Load a diffusion model from a checkpoint
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
    Save a list of images to a folder with a specified prefix
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i, image in enumerate(images):
        utils.save_image(image, os.path.join(folder, f"{prefix}_{i}.png"))

def evaluate_model(test_loader, model, model_name, device, num_timesteps=1000, denorm_dataset=None):
    '''
    Evaluate a model on the test set
    '''
    lpips_loss = lpips.LPIPS(net='alex').to(device)

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
    count = 0

    all_fid_scores = {i: [] for i in range(4)}

    sum_mismatch = {i: 0 for i in range(4)}
    mismatch_list = {i: [] for i in range(4)}

    for batch_idx, data in enumerate(test_loader):
        img, img_64, u10, v10, sp, t2m = data
        targets = [u10, v10, sp, t2m]

        inputs = img_64.to(device)
        targets = [t.to(device) for t in targets]
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
        
        for j in range(4):  # Iterate over variables
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

                # save images
                save_images([outputs[j][i]], generated_images_folder, f"generated_{batch_idx}_{i}_{j}")
                save_images([targets[j][i]], real_images_folder, f"real_{batch_idx}_{i}_{j}")

                # denormalization
                if denorm_dataset is not None:
                    output_np = denormalize(output_np, denorm_dataset.u10_min if j == 0 else denorm_dataset.v10_min if j == 1 else denorm_dataset.sp_min if j == 2 else denorm_dataset.t2m_min, 
                                            denorm_dataset.u10_max if j == 0 else denorm_dataset.v10_max if j == 1 else denorm_dataset.sp_max if j == 2 else denorm_dataset.t2m_max)
                    target_np = denormalize(target_np, denorm_dataset.u10_min if j == 0 else denorm_dataset.v10_min if j == 1 else denorm_dataset.sp_min if j == 2 else denorm_dataset.t2m_min, 
                                            denorm_dataset.u10_max if j == 0 else denorm_dataset.v10_max if j == 1 else denorm_dataset.sp_max if j == 2 else denorm_dataset.t2m_max)

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
                
                mismatch, diff_img = calculate_pixel_difference(target_np, output_np)
                diff_images[j].append(diff_img)

                mismatch_list[j].append(mismatch)
                sum_mismatch[j] += mismatch

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
    avg_mismatch = {i: sum_mismatch[i] / count for i in range(4)}

    print("\nAverage values:")
    for i in range(4):
        print(f"Variable {i}: KL Div = {avg_kl_div[i]}, RMSE = {avg_rmse[i]}, MAE = {avg_mae[i]}, PSNR = {avg_psnr[i]}, SSIM = {avg_ssim[i]}, LPIPS = {avg_lpips[i]}, FID = {avg_fid[i]}")

    return kl_div_list, rmse_list, mae_list, psnr_list, ssim_list, fid_list, lpips_list, mismatch_list, all_predictions, diff_images

def denormalize(tensor, min_val, max_val):
    '''
    Denormalization
    '''
    return tensor * (max_val - min_val) + min_val

def calculate_pixel_difference(img_a, img_b):
    '''
    Calculate the pixel difference between two images
    '''
    img_a_pil = Image.fromarray((img_a * 255).astype(np.uint8))
    img_b_pil = Image.fromarray((img_b * 255).astype(np.uint8))
    img_diff = Image.new("RGBA", img_a_pil.size)
    mismatch = pixelmatch(img_a_pil, img_b_pil, img_diff, includeAA=True)
    return mismatch, img_diff

def plot_results(input_image, targets, predictions, variable_names, model_names, index, results_folder):
    '''
    Plot the input image, true values, and predictions for each variable
    '''
    num_variables = len(variable_names)
    num_models = len(model_names)
    
    fig = plt.figure(figsize=(9, 15)) 
    gs = gridspec.GridSpec(nrows=num_variables, ncols=num_models + 2, figure=fig, wspace=0.05, hspace=0)

    lon_start, lon_end = 116.0794, 126.0794
    lat_start, lat_end = 18.9037, 28.9037
    lon, lat = np.meshgrid(np.linspace(lon_start, lon_end, 64), np.linspace(lat_start, lat_end, 64))

    for i, var_name in enumerate(variable_names):
        # Plot input image
        ax = fig.add_subplot(gs[i, 0], projection=ccrs.PlateCarree())
        ax.imshow(input_image[i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='gray')
        ax.add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
        ax.coastlines()
        ax.set_title('Input (img_64)', fontsize=12)

        # Plot true value
        ax = fig.add_subplot(gs[i, 1], projection=ccrs.PlateCarree())
        im = ax.imshow(targets[i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='coolwarm')
        ax.add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
        ax.coastlines()
        ax.set_title(f'True {var_name}', fontsize=12)

        # Plot predictions for each model
        for j, model_name in enumerate(model_names):
            ax = fig.add_subplot(gs[i, j + 2], projection=ccrs.PlateCarree())
            im = ax.imshow(predictions[model_name][i].cpu().numpy().squeeze(), extent=[lon_start, lon_end, lat_start, lat_end], cmap='coolwarm')
            ax.add_feature(cfeature.LAND, zorder=100, facecolor="none", edgecolor='k', linewidth=0.5)
            ax.coastlines()
            ax.set_title(f'{model_name} Prediction', fontsize=12)
        
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.01)
    
    plt.savefig(os.path.join(results_folder, f'results_{index}.png'))
    plt.close(fig)

def create_animation_from_folder(image_folder, output_file):
    '''
    Create an animation from a folder of images
    '''
    images = []
    sorted_file_names = sorted(
        [f for f in os.listdir(image_folder) if f.endswith(".png")],
        key=lambda x: int(x.split('_')[-1].split('.')[0])
    )

    for file_name in sorted_file_names:
        file_path = os.path.join(image_folder, file_name)
        images.append(Image.open(file_path))

    images[0].save(output_file, save_all=True, append_images=images[1:], duration=300, loop=0)

def main():
    set_seed(42)

    results_folder = "muifa_results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    print("Loading dataset...")
    sys.stdout.flush()
    base_directory = '/vol/bitbucket/zl1823/Typhoon-forecasting/Case_study/muifa'
    denorm_directory = '/vol/bitbucket/zl1823/Typhoon-forecasting/dataset/pre_processed_dataset/ERA5_without_nan_taiwan'

    combined_dataset = load_all_zarr_files(base_directory)
    denorm_dataset = load_all_zarr_files(denorm_directory)

    print("Creating datasets...")
    sys.stdout.flush()

    dataset = TyphoonDataset(combined_dataset)
    denorm_dataset = TyphoonDataset(denorm_dataset)

    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    train_loader = DataLoader([], batch_size=32)

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
        # 'CNN': '/vol/bitbucket/zl1823/Typhoon-forecasting/CNN/cnn_model.pth',
        # 'SENet': '/vol/bitbucket/zl1823/Typhoon-forecastingSENet/senet_model.pth',
        # 'DDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/DDPM/model-30.pt',
        'CDDPM': '/vol/bitbucket/zl1823/Typhoon-forecasting/CDDPM/experiments/train_typhoon_forecasting_240803_211802/checkpoint/3350_Network.pth',
    }

    # Load models
    # trained_models['CNN'] = load_cnn_model(trained_models['CNN'], model_files['CNN'], device)
    # trained_models['SENet'] = load_cnn_model(trained_models['SENet'], model_files['SENet'], device)
    # trained_models['DDPM'] = load_diffusion_model(trained_models['DDPM'], model_files['DDPM'], device)
    trained_models['CDDPM'] = load_con_diffusion_model(trained_models['CDDPM'], model_files['CDDPM'], device)

    evaluation_results = {}

    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name} model...")
        sys.stdout.flush()
        kl_div_list, rmse_list, mae_list, psnr_list, ssim_list, lpips_list, fid_list, mismatch_list, predictions, diff_images = evaluate_model(test_loader, model, model_name, device, num_timesteps=1000, denorm_dataset=denorm_dataset)

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

    for index, data in enumerate(test_loader):
        img, img_64, u10, v10, sp, t2m = data
        inputs = img_64.to(device)

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
            
            plot_results(single_input.cpu(), single_targets, single_predictions, variable_names, model_names, f"{index}_{i}", results_folder)

    create_animation_from_folder(results_folder, os.path.join(results_folder, "combined_animation.gif"))

    print("Finish visualization")

if __name__ == '__main__':
    main()
