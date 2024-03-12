from datetime import datetime
import sys 

from model import *
from dataloader import data_generator_motion
from trainer import Trainer

sys.path.append("/home/ubuntu/projects/TFC-pretraining/code/")
from config_files.MotionSample_Configs import Config as Configs
from optimizers import get_optimizer

with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

SAMPLE_RUN = True
if SAMPLE_RUN: 
  SAMPLE_COUNT, SAMPLE_COUNT_VAL = -1, -1
  pretrain_dataset = 'MotionSample'
else: 
  SAMPLE_COUNT, SAMPLE_COUNT_VAL = 1000, 200
  pretrain_dataset = 'Motion'

training_mode = 'pre_train'
MODEL_DIR = "/home/ubuntu/projects/TFC-pretraining/output/model"
SAVE_MODEL_OR_CHECKPOINTS = 'checkpoints'
configs = Configs()

run_date = datetime.today().strftime('%Y-%m-%d')
run_name = f"model_{run_date}"
print(run_name)

save_model_dir = MODEL_DIR

# Load data 
if SAMPLE_RUN: 
  train_data_dir = '/home/ubuntu/projects/TFC-pretraining/data/train_data_sample.npy'
  train_motion_names_dir = '/home/ubuntu/projects/TFC-pretraining/data/train_motion_names_sample.parquet'
  train_fft_dir = '/home/ubuntu/projects/TFC-pretraining/data/train_fft_sample.npy'
  val_data_dir = '/home/ubuntu/projects/TFC-pretraining/data/val_data_sample.npy'
  val_motion_names_dir = '/home/ubuntu/projects/TFC-pretraining/data/val_motion_names_sample.parquet'
  val_fft_dir = '/home/ubuntu/projects/TFC-pretraining/data/val_fft_sample.npy'
else: 
  train_data_dir = '/dbfs/FileStore/fengdiguo/PDS184_driver_no_distraction/PDS0512_TFC/fft_features/a6415950/train_data.npy'
  train_motion_names_dir = '/dbfs/FileStore/fengdiguo/PDS184_driver_no_distraction/PDS0488_motion_periods/ts2vec_features/a6415950/train_motion_names.parquet'
  train_fft_dir = '/dbfs/FileStore/fengdiguo/PDS184_driver_no_distraction/PDS0512_TFC/fft_features/a6415950/train_fft.npy'
  val_data_dir = '/dbfs/FileStore/fengdiguo/PDS184_driver_no_distraction/PDS0512_TFC/fft_features/a6415950/val_data.npy'
  val_motion_names_dir = '/dbfs/FileStore/fengdiguo/PDS184_driver_no_distraction/PDS0488_motion_periods/ts2vec_features/a6415950/val_motion_names.parquet'
  val_fft_dir = '/dbfs/FileStore/fengdiguo/PDS184_driver_no_distraction/PDS0512_TFC/fft_features/a6415950/val_fft.npy'

taindata_path_set = {'train_data_dir': train_data_dir, 
                     'train_motion_names_dir': train_motion_names_dir, 
                     'train_fft_dir': train_fft_dir}
valdata_path_set = {'val_data_dir': val_data_dir, 
                    'val_motion_names_dir': val_motion_names_dir, 
                    'val_fft_dir': val_fft_dir}

train_loader, val_loader, train_motion_names, val_motion_names = data_generator_motion(taindata_path_set, valdata_path_set, configs, training_mode, SAMPLE_COUNT, SAMPLE_COUNT_VAL)

"""Load model"""
# model = TFC(configs).to(device)
# model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
model = gpt4ts(configs).to(device)
optim_class = get_optimizer(configs.optimizer)
model_optimizer = optim_class(model.parameters(), lr=configs.gpt_lr, weight_decay=0)


"""Model training"""
train_loss, val_loss = Trainer(model, model_optimizer, train_loader, val_loader, device, configs,training_mode, save_model_dir, save_model_or_checkpoints=SAVE_MODEL_OR_CHECKPOINTS)
