import torch
import torchvision.models as models
import torch.optim as optim

from build_data import *
from module_list import *
from utilsm.loss import *
from validate import *
import torchvision
import wandb
import copy
from Model.Unet import UNet,opmoudle
import json
from utils.pyt_utils import load_model

wandb.init(project="Teacher-student-bra", entity="wyy-team",name="my-bra-multi-fu-50-multi4-0.1-val")

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

data_loader = BuildDataLoader(config["dataset"], config["num_labels"], config["data_path"])
train_l_loader, train_u_loader, val_loader, test_loader = data_loader.build(supervised=False)

device = torch.device("cuda:{:d}".format(config["gpu"]) if torch.cuda.is_available() else "cpu")


model= UNet(in_channels=1, is_batchnorm=True, n_classes=2,mun_pro=config["mun_pro"]).to(device)
op_module=opmoudle().to(device)

total_epoch = 200
optimizer = optim.SGD(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, total_epoch, power=0.9)

optimizer_op = optim.SGD(op_module.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], momentum=0.9, nesterov=True)
scheduler_op = PolyLR(optimizer_op, total_epoch, power=0.9)

ema = EMA(model, 0.99)  # Mean teacher model


train_epoch = len(train_l_loader)
test_epoch = len(test_loader)
val_epoch = len(val_loader)
avg_cost = np.zeros((total_epoch, 10))
avg_acc_reco=np.zeros((total_epoch, 3))
iteration = 0
##train
current_index = -1
save_best = os.path.join(config["snapshot_path"] + '/model', 'model_best.pth')
save_best_op = os.path.join(config["snapshot_path"] + '/model', 'model_op_best.pth')
save_last = os.path.join(config["snapshot_path"] + '/model', 'model_last.pth')
save_last_op = os.path.join(config["snapshot_path"] + '/model', 'model_op_last.pth')

best_dice = 0
results_log = open(config["record_dir"], 'a')
lines = []
for index in range(current_index + 1, total_epoch):
    cost = np.zeros(4)
    acc_record = np.zeros(3)
    train_l_dataset = iter(train_l_loader)
    train_u_dataset = iter(train_u_loader)
    model.train()
    op_module.train()
    ema.model.train()

    for i in range(train_epoch):
        train_l_data, train_l_label, _ = train_l_dataset.__next__()
        train_l_data, train_l_label = train_l_data.to(device), train_l_label.to(device)

        train_u_data, train_u_label, name = train_u_dataset.__next__()
        train_u_data, train_u_label = train_u_data.to(device), train_u_label.to(device)

        optimizer.zero_grad()
        optimizer_op.zero_grad()
        # generate pseudo-labels
        with torch.no_grad():
            if index > 3:  #
                op_module.eval()
                pred_u_or, re_uu, mm_map_1, prototype0 = ema.model(train_u_data, mm=True)
                op_feature_1 = op_module(pred_u_or, mm_map_1)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(op_feature_1, dim=1), dim=1)
                prototype = copy.deepcopy(prototype0).detach()
            else:
                pred_u, re_uu = ema.model(train_u_data)
                pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u, dim=1), dim=1)

                # random scale images first
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_data, pseudo_labels, pseudo_logits,
                                data_loader.crop_size, data_loader.scale_size, apply_augmentation=False)

            # apply mixing strategy: cutout, cutmix or classmix
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data(train_u_aug_data, train_u_aug_label, train_u_aug_logits, mode=config["apply_aug"])

            # apply augmentation: color jitter + flip + gaussian blur
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                batch_transform(train_u_aug_data, train_u_aug_label, train_u_aug_logits,
                                data_loader.crop_size, (1.0, 1.0), apply_augmentation=True)

        # generate labelled and unlabelled data loss
        pred_l, rep_l, MM, _ = model(train_l_data, mm=True)
        pred_l_large = pred_l
        pred_u, rep_u = model(train_u_aug_data.float())
        pred_u_large = pred_u
        rep_all = torch.cat((rep_l, rep_u))
        pred_all = torch.cat((pred_l, pred_u))

        # supervised-learning loss
        sup_ce_loss2 = F.cross_entropy(pred_l, train_l_label,ignore_index=-1).mean()
        outputs_soft = F.softmax(pred_l, dim=1)
        sup_dice_loss2 = dice_loss(outputs_soft[:, 1, :, :, :], train_l_label== 1).mean()##only foreground

        sup_ce_loss3 = F.cross_entropy(MM, train_l_label, ignore_index=-1).mean()
        outputs_soft3 = F.softmax(MM, dim=1)
        sup_dice_loss3 = dice_loss(outputs_soft3[:, 1, :, :, :], train_l_label == 1).mean()  ##only foreground

        sup_loss = 0.5 * (sup_ce_loss2 + sup_dice_loss2+sup_ce_loss3 + sup_dice_loss3)
        # unsupervised-learning loss
        unsup_loss = compute_unsupervised_loss(pred_u_large, train_u_aug_label, train_u_aug_logits,
                                                config["strong_threshold"])
        with torch.no_grad():
            train_u_aug_mask = train_u_aug_logits.ge(config["weak_threshold"]).float()
            mask_all = torch.cat(((train_l_label.unsqueeze(1) >= 0).float(), train_u_aug_mask.unsqueeze(1)))
            mask_all = F.interpolate(mask_all, size=pred_all.shape[2:], mode='nearest')
            label_l = F.interpolate(label_onehot(train_l_label, data_loader.num_segments),
                                    size=pred_all.shape[2:], mode='nearest')
            label_u = F.interpolate(label_onehot(train_u_aug_label, data_loader.num_segments),
                                    size=pred_all.shape[2:], mode='nearest')
            label_all = torch.cat((label_l, label_u))
            prob_l = torch.softmax(pred_l, dim=1)
            prob_u = torch.softmax(pred_u, dim=1)
            prob_all = torch.cat((prob_l, prob_u))
        if index > 3:
            cps_loss = compute_cps_loss(rep_all, label_all, mask_all, prob_all, cls=prototype,
                                        strong_threshold=config["strong_threshold"], temp=config["temp"],
                                        num_queries=config["num_queries"], num_negatives=config["num_negatives"],
                                        con_xi=config["xi"])
        else:
            cps_loss = compute_cps_loss(rep_all, label_all, mask_all, prob_all,
                                        strong_threshold=config["strong_threshold"], temp=config["temp"],
                                        num_queries=config["num_queries"], num_negatives=config["num_negatives"])

        loss = sup_loss + 0.5 * unsup_loss + 0.5 * cps_loss
        loss.backward()
        optimizer.step()
        ema.update(model)

        # if index > 0:
        model.eval()
        op_module.train()
        with torch.no_grad():
            Pred_l, _, Mm_map_l,_ = model(train_l_data, mm=True)
            Pred_ll=Pred_l.detach().clone()
            Mm_map_ll = Mm_map_l.detach().clone()

        Op_feature = op_module(Pred_ll, Mm_map_ll)
        sup_ce_loss_op = F.cross_entropy(Op_feature, train_l_label, ignore_index=-1).mean()
        outputs_soft_op = F.softmax(Op_feature, dim=1)
        sup_dice_loss_op = dice_loss(outputs_soft_op[:, 1, :, :, :], train_l_label == 1).mean()
        op_loss = 0.5 * (sup_ce_loss_op + sup_dice_loss_op)
        op_loss.backward()
        optimizer_op.step()
        model.train()

        cost[0] = sup_loss.item()
        cost[1] = unsup_loss.item()
        cost[2] = cps_loss.item()
        cost[3] = op_loss.item()
        avg_cost[index, :4] += cost / train_epoch
        iteration += 1

    with torch.no_grad():
        metric_record = 0.0
        ema.model.eval()
        op_module.eval()
        dataloader = iter(val_loader)
        tbar = range(len(val_loader))
        tbar = tqdm(tbar, ncols=135)
        for batch_idx in tbar:
            x, y, _ = next(dataloader)
            y=y.squeeze(0)
            y_tilde, y_hat = test_single_case(ema.model,  op_module, x,
                                                stride_xy=16, stride_z=16,
                                                patch_size=(96, 96, 96), num_classes=data_loader.num_segments)
            if np.sum(y_tilde) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(numpy.array(y_tilde),
                                                            numpy.array(y[:]))

            metric_record += np.asarray(single_metric)

        metric_record = metric_record / len(val_loader)
    scheduler.step()
    scheduler_op.step()

    if metric_record[0]>= best_dice:
        best_dice=metric_record[0].copy()
        torch.save(ema.model.state_dict(), save_best)
        torch.save(op_module.state_dict(), save_best_op)
    if index>198:
        torch.save(ema.model.state_dict(), save_last)
        torch.save(op_module.state_dict(), save_last_op)


    print( 'EPOCH: {:04d} ITER: {:04d} | TRAIN [Loss]: {:.4f} {:.4f} {:.4f} {:.4f} '.format(index, iteration, avg_cost[index][0], avg_cost[index][1], avg_cost[index][2],avg_cost[index][3]))
    print('Top: best dice {:.4} dice {:.4f} jaccard {:.4f}  hd95 {:.4f}  asd {:.4f}'.format(best_dice,metric_record[0], metric_record[1], metric_record[2],metric_record[3]))

    wandb.log({"loss_sup": avg_cost[index][0],
                "epoch": index})
    wandb.log({"loss_un": avg_cost[index][1],
                "epoch": index})
    wandb.log({"loss_reco": avg_cost[index][2],
                "epoch": index})
    wandb.log({"op_reco": avg_cost[index][3],
                "epoch": index})

    wandb.log({"dice": metric_record[0],
                "epoch": index})
    wandb.log({"jaccard": metric_record[1],
                "epoch": index})
    wandb.log({"hd95": metric_record[2],
                "epoch": index})
    wandb.log({"asd": metric_record[3],
                "epoch": index})

line = "\n".join(lines)
results_log.write(line)
results_log.write('\n')
results_log.flush()
results_log.close()

if config["Test"]:
    results_log = open(config["record_dir"], 'a')
    model_path = os.path.join(config["model_path"], 'model_last.pth')
    op_model_path = os.path.join(config["model_path"], 'model_op_last.pth')

    model = load_model(model, model_path)
    op_module = load_model(op_module, op_model_path)
    model.eval()
    op_module.eval()
    with torch.no_grad():
        metric_record = 0.0
        dataloader = iter(test_loader)
        tbar = range(len(test_loader))
        tbar = tqdm(tbar, ncols=135)
        for batch_idx in tbar:
            x, y, _ = next(dataloader)
            y=y.squeeze(0)
            y_tilde, y_hat = test_single_case(ema.model,  op_module, x,
                                                stride_xy=16, stride_z=16,
                                                patch_size=(96, 96, 96), num_classes=data_loader.num_segments)
            if np.sum(y_tilde) == 0:
                single_metric = (0, 0, 0, 0)
            else:
                single_metric = calculate_metric_percase(numpy.array(y_tilde),
                                                            numpy.array(y[:]))

            metric_record += np.asarray(single_metric)

        metric_record = metric_record / len(test_loader)

        lines = []
        lines.append('dice: {}%  jaccard: {}%  hd95: {}% asd: {}%  '.format(metric_record[0], metric_record[1],metric_record[2],metric_record[3]))
        line = "\n".join(lines)
        results_log.write(line)
        results_log.write('\n')
        results_log.flush()
        results_log.close()
