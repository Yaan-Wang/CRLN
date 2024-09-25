import numpy as np
import torch

from build_data import *
from utils.pyt_utils import load_model
from torch.utils.data import Dataset, DataLoader

from Model.Vnet import VNet,opmoudle
from validate import *
from sklearn.utils import resample
import json



def save_sample_names(sample_names_list, file_path):
    with open(file_path, 'w') as f:
        for name in sample_names_list:
            f.write(name + '\n')

def load_sample_names(file_path):
    with open(file_path, 'r') as f:
        return [line.strip() for line in f]


class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def custom_collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    names = [item[2] for item in batch]  #


    return inputs, labels, names


def data_loader_to_list(data_loader):
    data_list = []
    for batch in data_loader:
        x, y, name = batch
        data_list.append((x, y, name))
    return data_list


def list_to_dataloader(data_list, batch_size):
    dataset = CustomDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    return dataloader

def bootstrap_evaluation(model, op_module, test_loader, num_samples=100, load_path=None, metrics_save_file=None):
    metrics_list = []
    test_data_list = data_loader_to_list(test_loader)
    results_log = open(metrics_save_file, 'a')  # Open the log file for appending
    lines = []

    if load_path:
        sampled_data_list = []
        for i in range(num_samples):
            sampled_name_file = f"{load_path}_sample_{i}.txt"
            sampled_name_list = load_sample_names(sampled_name_file)
            # sampled_name_list = load_sample_names(load_path, i)
            resampled_data_list = []
            for name in sampled_name_list:
                for item in test_data_list:
                    if item[2][0] == name:
                        resampled_data_list.append(item)
                        break
            sampled_data_list.append(resampled_data_list)

            batch_size = 1
            sampled_test_loader = list_to_dataloader(resampled_data_list, batch_size)

            metrics_sample = []
            for x, y,names in sampled_test_loader:
                y = y[0].squeeze(0)  
                x=x[0]
                y = y.squeeze(0)
                y_tilde, y_hat = test_single_case(model, op_module, x,
                                                  stride_xy=16, stride_z=16,
                                                  patch_size=(96, 96, 96), num_classes=2)
            
                if np.sum(y_tilde) == 0:
                    single_metric = (0, 0, 0, 0)
                else:
                    single_metric = calculate_metric_percase(np.array(y_tilde), np.array(y[:].squeeze(0)))

                metrics_sample.append(single_metric)

            metrics_array = np.asarray(metrics_sample)
            metric_mean = np.nanmean(metrics_array, axis=0)
            metrics_list.append(metric_mean)

            lines.append(
                'Sample {}: Dice: {:.6f}%  Jaccard: {:.6f}%  HD95: {:.6f}%  ASD: {:.6f}%'.format(
                    i, metric_mean[0], metric_mean[1], metric_mean[2], metric_mean[3]
                )
            )
            print( 'Sample {}: Dice: {:.6f}%  Jaccard: {:.6f}%  HD95: {:.6f}%  ASD: {:.6f}%'.format(
                    i, metric_mean[0], metric_mean[1], metric_mean[2], metric_mean[3]))
    else:
        for i in tqdm(range(num_samples), desc='Bootstrapping Iterations'):
            resampled_data_list = resample(test_data_list, replace=True)
            sampled_name_list = [item[2][0] for item in resampled_data_list]
            batch_size = 1
            sampled_test_loader = list_to_dataloader(resampled_data_list, batch_size)

            metrics_sample = []
            for x, y,names in sampled_test_loader:
                y = y[0].squeeze(0)  
                x=x[0]
                y_tilde, y_hat = test_single_case(model, op_module, x,
                                                  stride_xy=16, stride_z=16,
                                                  patch_size=(96, 96, 96), num_classes=2)

                if np.sum(y_tilde) == 0:
                    single_metric = (0, 0, 0, 0)
                else:
                    single_metric = calculate_metric_percase(np.array(y_tilde), np.array(y[:].squeeze(0)))

                metrics_sample.append(single_metric)

            metrics_array = np.asarray(metrics_sample)
            metric_mean = np.nanmean(metrics_array, axis=0)
            metrics_list.append(metric_mean)

            lines.append(
                'Sample {}: Dice: {:.6f}%  Jaccard: {:.6f}%  HD95: {:.6f}%  ASD: {:.6f}%'.format(
                    i, metric_mean[0], metric_mean[1], metric_mean[2], metric_mean[3]
                )
            )

    metrics_array = np.asarray(metrics_list)
    mean_metrics = np.mean(metrics_array, axis=0)
    std_metrics = np.std(metrics_array, axis=0)

    # Save all metric_means to a .txt file
    results_log.write("\n".join(lines))  # Write all lines to the file
    results_log.write('\n')
    results_log.close()

    return mean_metrics, std_metrics


# Load configuration from JSON file
parser = argparse.ArgumentParser(description='Semi-supervised Segmentation with Perfect Labels')
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

# Load the configuration file
with open(args.config, 'r') as f:
    config = json.load(f)

##
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed(config["seed"])
os.environ['PYTHONHASHSEED'] = str(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
torch.use_deterministic_algorithms(True)

data_loader = BuildDataLoader(config["dataset"], config["num_labels"], config["data_path"])
train_l_loader, train_u_loader, val_loader, test_loader = data_loader.build(supervised=False)

device = torch.device("cuda:{:d}".format(config["gpu"]) if torch.cuda.is_available() else "cpu")

model = VNet(n_channels=1, n_classes=2, mun_pro=config["mun_pro"],
             normalization='batchnorm', has_dropout=True).to(device)
op_module = opmoudle().to(device)

model_path = os.path.join(config["model_path"],'model_last.pth')
op_model_path= os.path.join(config["model_path"],'model_op_last.pth')

model = load_model(model, model_path)
op_module =load_model(op_module, op_model_path)
model.eval()
op_module.eval()

# Perform bootstrapping
mean_metrics, std_metrics = bootstrap_evaluation(
    model, op_module, test_loader, num_samples=100, load_path=config["load_path"],metrics_save_file=config["record_dir"])

# Log results
print(f'Bootstrapping Average metrics: {mean_metrics} ± {std_metrics}')
with open(config["record_dir"], 'a') as results_log:
    results_log.write("mean bootstrapping:\n")
    results_log.write(f'Bootstrapping dice: {mean_metrics[0]}% ± {std_metrics[0]}%  '
                        f'jaccard: {mean_metrics[1]}% ± {std_metrics[1]}%  '
                        f'hd95: {mean_metrics[2]}% ± {std_metrics[2]}%  '
                        f'asd: {mean_metrics[3]}% ± {std_metrics[3]}%\n')

