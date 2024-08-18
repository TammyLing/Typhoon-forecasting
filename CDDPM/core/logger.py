import os
from PIL import Image
import importlib
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import core.util as Util

class InfoLogger():
    """
    use logging to record log, only work on GPU 0 by judging global_rank
    """
    def __init__(self, opt):
        self.opt = opt
        self.rank = opt['global_rank']
        self.phase = opt['phase']

        self.setup_logger(None, opt['path']['experiments_root'], opt['phase'], level=logging.INFO, screen=False)
        self.logger = logging.getLogger(opt['phase'])
        self.infologger_ftns = {'info', 'warning', 'debug'}

    def __getattr__(self, name):
        if self.rank != 0: # info only print on GPU 0.
            def wrapper(info, *args, **kwargs):
                pass
            return wrapper
        if name in self.infologger_ftns:
            print_info = getattr(self.logger, name, None)
            def wrapper(info, *args, **kwargs):
                print_info(info, *args, **kwargs)
            return wrapper
    
    @staticmethod
    def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
        """ set up logger """
        l = logging.getLogger(logger_name)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        log_file = os.path.join(root, '{}.log'.format(phase))
        fh = logging.FileHandler(log_file, mode='a+')
        fh.setFormatter(formatter)
        l.setLevel(level)
        l.addHandler(fh)
        if screen:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            l.addHandler(sh)

class VisualWriter():
    """ 
    use tensorboard to record visuals, support 'add_scalar', 'add_scalars', 'add_image', 'add_images', etc. funtion.
    Also integrated with save results function.
    """
    def __init__(self, opt, logger):
        log_dir = opt['path']['tb_logger']
        self.result_dir = opt['path']['results']
        enabled = opt['train']['tensorboard']
        self.rank = opt['global_rank']

        self.writer = None
        self.selected_module = ""

        if enabled and self.rank==0:
            log_dir = str(log_dir)

            # Retrieve vizualization writer.
            succeeded = False
            for module in ["tensorboardX", "torch.utils.tensorboard"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module

            if not succeeded:
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.epoch = 0
        self.iter = 0
        self.phase = ''

        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.custom_ftns = {'close'}
        self.timer = datetime.now()

    def set_iter(self, epoch, iter, phase='train'):
        self.phase = phase
        self.epoch = epoch
        self.iter = iter

    def save_np_arrays(self, results):
        base_result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(base_result_path, exist_ok=True)
        epoch_path = os.path.join(base_result_path, str(self.epoch))
        os.makedirs(epoch_path, exist_ok=True)

        channels = ['u10', 'v10', 'sp', 't2m']
        for channel in channels:
            channel_path = os.path.join(epoch_path, channel)
            os.makedirs(channel_path, exist_ok=True)

        names = results['name']
        arrays = results['result']
        for name, array_dict in zip(names, arrays):
            gt_image = array_dict['gt']
            cond_image = array_dict['cond']
            process_image = array_dict['out_vis']
            out_image = array_dict['out']

            for i, channel in enumerate(channels):
                gt_channel = gt_image[i].squeeze()
                cond_channel = cond_image[i].squeeze()
                process_channel = process_image[i].squeeze()
                out_channel = out_image[i].squeeze()

                # cond_channel = cond_channel * (self.dataset.u10_max - self.dataset.u10_min) + self.dataset.u10_min

                # if channel == 'U10':
                #     out_channel = out_channel * (self.dataset.u10_max - self.dataset.u10_min) + self.dataset.u10_min
                #     gt_channel = gt_channel * (self.dataset.u10_max - self.dataset.u10_min) + self.dataset.u10_min
                # elif channel == 'V10':
                #     out_channel = out_channel * (self.dataset.v10_max - self.dataset.v10_min) + self.dataset.v10_min
                #     gt_channel = gt_channel * (self.dataset.v10_max - self.dataset.v10_min) + self.dataset.v10_min
                # elif channel == 'Pressure':
                #     out_channel = out_channel * (self.dataset.sp_max - self.dataset.sp_min) + self.dataset.sp_min
                #     gt_channel = gt_channel * (self.dataset.sp_max - self.dataset.sp_min) + self.dataset.sp_min
                # else:
                #     out_channel = out_channel * (self.dataset.t2m_max - self.dataset.t2m_min) + self.dataset.t2m_min
                #     gt_channel = gt_channel * (self.dataset.t2m_max - self.dataset.t2m_min) + self.dataset.t2m_min

                #fig, axs = plt.subplots(1, 3, figsize=(15, 10))  # 四个子图：GT, Cond, Process, Out

                lon_start, lon_end = 116.0794, 126.0794
                lat_start, lat_end = 18.9037, 28.9037
                lon = np.linspace(lon_start, lon_end, gt_channel.shape[1])
                lat = np.linspace(lat_start, lat_end, gt_channel.shape[0])
                lon, lat = np.meshgrid(lon, lat)


                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                ax = axs[0]
                input_plot = ax.imshow(cond_channel, extent=[lon_start, lon_end, lat_start, lat_end], cmap='jet', aspect='equal')
                ax.set_title('Input Image')
                fig.colorbar(input_plot, ax=ax)
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                        
                ax = axs[1]
                true_plot = ax.imshow(gt_channel, extent=[lon_start, lon_end, lat_start, lat_end], cmap='jet', aspect='equal')
                ax.set_title('True Value')
                fig.colorbar(true_plot, ax=ax)

                ax = axs[2]
                pred_plot = ax.imshow(out_channel, extent=[lon_start, lon_end, lat_start, lat_end], cmap='jet', aspect='equal')
                ax.set_title('Predicted Value')
                fig.colorbar(pred_plot, ax=ax)
                

                # im0 = axs[0, 0].imshow(gt_channel, extent=[116.0794, 126.0794, 18.9037, 28.9037], cmap='viridis', aspect='auto', origin='lower')
                # axs[0, 0].set_title('Ground Truth')
                # plt.colorbar(im0, ax=axs[0, 0])

                # im1 = axs[0, 1].imshow(cond_channel, extent=[116.0794, 126.0794, 18.9037, 28.9037], cmap='viridis', aspect='auto', origin='lower')
                # axs[0, 1].set_title('Conditional Image')
                # plt.colorbar(im1, ax=axs[0, 1])

                # im2 = axs[1, 0].imshow(process_channel, extent=[116.0794, 126.0794, 18.9037, 28.9037], cmap='viridis', aspect='auto', origin='lower')
                # axs[1, 0].set_title('Output_vis')
                # plt.colorbar(im2, ax=axs[1, 0])

                # im3 = axs[1, 1].imshow(out_channel, extent=[116.0794, 126.0794, 18.9037, 28.9037], cmap='viridis', aspect='auto', origin='lower')
                # axs[1, 1].set_title('Output')
                # plt.colorbar(im3, ax=axs[1, 1])

                fig_path = os.path.join(epoch_path, channel, f"{name}.png")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close(fig)


    def save_images(self, results):
        result_path = os.path.join(self.result_dir, self.phase)
        os.makedirs(result_path, exist_ok=True)
        result_path = os.path.join(result_path, str(self.epoch))
        os.makedirs(result_path, exist_ok=True)

        ''' get names and corresponding images from results[OrderedDict] '''
        try:
            names = results['name']
            outputs = Util.postprocess(results['result'])
            for i in range(len(names)): 
                Image.fromarray(outputs[i]).save(os.path.join(result_path, names[i]))
        except:
            raise NotImplementedError('You must specify the context of name and result in save_current_results functions of model.')
    
    def close(self):
        self.writer.close()
        print('Close the Tensorboard SummaryWriter.')

        
    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return a blank function handle that does nothing
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # add phase(train/valid) tag
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(self.phase, tag)
                    add_data(tag, data, self.iter, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr


class LogTracker:
    """
    record training numerical indicators.
    """
    def __init__(self, *keys, phase='train'):
        self.phase = phase
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return {'{}/{}'.format(self.phase, k):v for k, v in dict(self._data.average).items()}
