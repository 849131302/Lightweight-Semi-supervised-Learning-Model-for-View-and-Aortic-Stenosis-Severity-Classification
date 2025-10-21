import argparse
import ast
import json
import math
import random
import numpy as np
import timm
from sklearn.metrics import accuracy_score,classification_report,balanced_accuracy_score
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler,SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from timm.utils import accuracy as acc, AverageMeter, ModelEma, ModelEmaV2
import warnings


from models.lwn import LWN_4VIEW, SLW_Net3, reparameterize_model

warnings.filterwarnings("ignore",category=UserWarning)

from augmennt import TargetedStrongAugment
import logging
import os
import time
import torch
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from matplotlib import pyplot as plt
from timm.data import Mixup
from timm.utils import accuracy as acc, AverageMeter, ModelEma
from torch.nn import CrossEntropyLoss
from tqdm import tqdm


class Threshold():
    def __init__(self, num_classes, momentum=0.999,use_quantile=True, clip_thresh=True): #use_quantile存储是否使用分位数进行更新。    clip_thresh是否进行剪切操作，将其限制在指定的范围内，通常在0.0到0.95之间。
        self.num_classes = num_classes
        self.m = momentum
        self.use_quantile = use_quantile
        self.clip_thresh= clip_thresh

        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()

    def entropy_loss(self,mask, logits_s):
        mask = mask.bool()
        # select samples
        logits_s = logits_s[mask]

        prob_s = logits_s.softmax(dim=-1)
        _, pred_label_s = torch.max(prob_s, dim=-1)

        hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
        hist_s = hist_s / hist_s.sum()

        # modulate prob model
        prob_model = self.p_model.reshape(1, -1)
        label_hist = self.label_hist .reshape(1, -1)
        # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
        prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
        mod_prob_model = prob_model * prob_model_scaler
        mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

        # modulate mean prob
        mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
        # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
        mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
        mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

        loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
        loss = loss.sum(dim=1)
        return loss.mean(), hist_s.mean()

    @torch.no_grad()
    def update(self, probs_x_ulb):
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1, keepdim=True)

        if self.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs, 0.8)
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        if self.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)
        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype)
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())
    @torch.no_grad()
    def masking(self, logits_x_ulb, softmax_x_ulb=True):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)
        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()
        self.update(probs_x_ulb)
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        return mask

class PseudoLabel():
    def __init__(self, use_hard_label=False, T=1.0, softmax=True, label_smoothing=0.0):
        """
        Initialize PseudoLabelingHook with parameters

        Args:
            use_hard_label: flag of using hard labels instead of soft labels
            T: temperature parameters
            softmax: flag of using softmax on logits
            label_smoothing: label_smoothing parameter
        """
        self.use_hard_label = use_hard_label
        self.T = T
        self.softmax = softmax
        self.label_smoothing = label_smoothing
    @torch.no_grad()
    def gen_ulb_targets(self, logits):
        """
        Generate pseudo-labels from logits/probs
        Args:
            logits: logits (or probs, need to set softmax to False)
        """
        logits = logits.detach()
        if self.use_hard_label:
            # return hard label directly
            pseudo_label = torch.argmax(logits, dim=-1)
            if self.label_smoothing:
                pseudo_label = self.smooth_targets(logits, pseudo_label, self.label_smoothing)
            return pseudo_label

        # return soft label
        if self.softmax:
            pseudo_label = torch.softmax(logits / self.T, dim=-1)
        else:
            # inputs logits converted to probabilities already
            pseudo_label = logits
        return pseudo_label
    def smooth_targets(self,logits, labels, smoothing):
        with torch.no_grad():
            true_dist = torch.zeros_like(logits)
            true_dist.fill_(smoothing / (logits.shape[-1] - 1))
            true_dist.scatter_(1, labels.data.unsqueeze(1), (1 - smoothing))
        return true_dist




class TBLog:
    def __init__(self, tb_dir, file_name, use_tensorboard=False):
        self.tb_dir = tb_dir
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            self.writer = SummaryWriter(os.path.join(self.tb_dir, file_name))

    def update(self, log_dict, it, suffix=None):
        if suffix is None:
            suffix = ''
        if self.use_tensorboard:
            for key, value in log_dict.items():
                self.writer.add_scalar(suffix + key, value, it)

def seed_everything(seed):   #设置随机因子  设置了固定的随机因子，再次训练的时候就可以保证图片的加载顺序不会发生变化
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
def get_config():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    def parse_numpy_array(s):
        return np.array(ast.literal_eval(s))
    parser = argparse.ArgumentParser(description="Semi-Supervised Learning")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0,help='the ratio of unloss to loss  in each mini-batch')#无标签数据损失和有标签数据损失之间的比例。
    parser.add_argument("--use_cat",type=str2bool, default=False, help='use cat operation in algorithms')  #STARING TO BOOL是否使用拼接输入，将有标签数据、弱无标签数据和强无标签数据拼接在一起
    parser.add_argument("--use_amp", type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument("--clip_grad", type=float, default=5.0)  #梯度裁剪的阈值 默认059
    parser.add_argument("--save_name", type=str, default='FreeMatch-DEV165VIEW-200EPOCH1E-3spli0-2RATIO-CUTMIX')#softmatch_fastvit
    parser.add_argument("--save_dir",  type=str, default='./FreeMatch-VIEW')
    parser.add_argument("--resume", type=str,default=None)#''/home/CV/AS/checkpoints/不归一 无预训练，TMED自己训练/model_35_78.306.pth'
    parser.add_argument("--EMA_decay", type=float,default=0.999)  # 是用于控制概率分布的 EMA 的 momentum 参数,该参数通常用于计算无标签数据的伪标签（pseudo labels）的概率分布。0.999

    # PseudoLabel arguments
    parser.add_argument("--T", type=float, default=1.0) #0.5 0.1 6 0.95[0.5，0.95]
    parser.add_argument("--use_hard_label", type=str2bool, default=False)
    parser.add_argument("--momtum", type=float, default=0.999)#0.999
    #Mask arguments
    parser.add_argument("--ent_loss_ratio", type=float, default=1e-2)#1e-2
    parser.add_argument("--use_quantile", type=str2bool, default=True)#true
    parser.add_argument("--clip_thresh", type=str2bool, default=False)
    # common utils arguments
    parser.add_argument("--distributed", type=str2bool,default=False,  help='**node rank** for distributed training')  #是否进行分布式训练
    parser.add_argument("--world_size", type=int, default=1, help='number of nodes for distributed training')  #分布式训练的节点总数
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    #dataset arguments
    parser.add_argument("--use_cutmix", type=str2bool, default=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--ulb_ratio", type=int, default=2,help='the ratio of unlabeled data to labeled data in each mini-batch')
    #optimizer argument
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    #model arguments
    parser.add_argument("--use_pretrain",  type=str2bool,default=False) #TRUE
    parser.add_argument("--use_ema", type=str2bool, default=True)  #FALSE



    args = parser.parse_args()
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    def save_args_to_json(args, file_path):
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(vars(args), f, indent=4, default=convert_numpy_types)
    save_args_to_json(args,file_path=os.path.join(args.save_dir,args.save_name,'config.json'))
    return args


def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val


def ce_loss(logits, targets, reduction='none'):

    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
def consistency_loss(logits, targets, name='ce', mask=None):
    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = F.cross_entropy(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask
    return loss.mean()

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=89./180.,
                                    #num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def count_parameters(model):
    # count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




def FreeMatchSSLtrain_one_epoch(args,model, train_loader1,train_loader2,optimizer,scheduler,epoch,model_ema):
    if epoch == 1:
        flops = FlopCountAnalysis(model, inp)
        print(flop_count_table(flops, max_depth=5))
    model.train()
    all_true_labels = []
    all_pred_labels = []
    trainloss_meter = AverageMeter()
    accury_meter = AverageMeter()
    start_time = time.time()
    #accuracy = 0.0
    # total_loss = 0.0
    labeled_dataloader_iter = iter(train_loader1)
    unlabeled_dataloader_iter = iter(train_loader2)
    #Aug =Mixup(mixup_alpha=0, cutmix_alpha=1.0, cutmix_minmax=None,prob=0.1, switch_prob=0.5, mode='batch', label_smoothing=0.1, num_classes=args.num_classes)# label_smoothing=0.1
    Aug = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=1.0, switch_prob=0.5, mode='batch',label_smoothing=0.1, num_classes=args.num_classes)
    with tqdm(total=num_batches,desc=f'Epoch {epoch}/{args.epoch}') as pbar:
        for batch_idx in range(num_batches):
            try:
                labeled_batch= next((labeled_dataloader_iter))
            except StopIteration:
                labeled_dataloader_iter = iter(labeled_dataloader)
                labeled_batch=next(labeled_dataloader_iter)

            try:
                unlabeled_batch = next(unlabeled_dataloader_iter)
            except StopIteration:
                unlabeled_dataloader_iter = iter(unlabeled_dataloader)
                unlabeled_batch = next(unlabeled_dataloader_iter)
            x_lb, y_lb=labeled_batch
            x_lb, y_lb= x_lb.to(device),y_lb.to(device)
            # 在每个批次中添加真实标签
            all_true_labels.extend(y_lb.cpu().numpy())
            num_lb = y_lb.shape[0]  # 获取有标签数据的数量
            if args.use_cutmix:
                x_lb, y_lb = Aug(x_lb, y_lb)
            (x_ulb_w, x_ulb_s), y_ulb =unlabeled_batch
            x_ulb_w, x_ulb_s= x_ulb_w.to(device),x_ulb_s.to(device)

            # inference and calculate sup/unsup losses
            if args.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))  # 将有标签数据、弱无标签数据和强无标签数据拼接成一个输入张量。
                outputs = model(inputs)
                logits_x_lb = outputs[:num_lb]  # 获取有监督数据的预测结果。
                logits_x_ulb_w, logits_x_ulb_s = outputs[num_lb:].chunk(2)  # 将模型对无标签数据的输出 (outputs['logits'][num_lb:]) 切分成两个部分，分别赋值给 logits_x_ulb_w 和 logits_x_ulb_s
            else:
                logits_x_lb = model(x_lb)
                logits_x_ulb_s = model(x_ulb_s)
                with torch.no_grad():
                    logits_x_ulb_w = model(x_ulb_w)     #logits_x_ulb_w是未经过softmatch的
                # print(logits_x_lb.shape)
            '''使用弱增强数据生成伪标签，然后使用强增强数据计算无监督损失'''
            accuracy = (acc(logits_x_lb,torch.max(y_lb,1)[1], topk=(1,)))[0].item() if args.use_cutmix else (acc(logits_x_lb, y_lb, topk=(1,)))[0].item()
            #accuracy = (acc(logits_x_lb, y_lb, topk=(1,)))[0].item()
            accury_meter.update(accuracy, num_lb)
            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            # 在每个批次中添加预测标签
            _, pred = torch.max(probs_x_lb, 1)
            # print(pred)
            all_pred_labels.extend(pred.cpu().numpy())
            # calculate weight
            mask = Mask.masking(logits_x_ulb=logits_x_ulb_w)
            # print(mask)
            # generate unlabeled targets using pseudo label hook     call_hook调用所有 hooks 中的特定函数
            pseudo_label = Pseudo.gen_ulb_targets(logits=logits_x_ulb_w)  # Temperature 参数，用于控制生成的伪标签的分布的“软化”程度)
            # print(pseudo_label)

            # calculate loss 有标签损失函数+一致性正则化损失函数
            if args.use_amp:
                with torch.cuda.amp.autocast():  # 自动混合精度的计算
                    sup_loss = criterion_train(logits_x_lb, y_lb)
                    #sup_loss = torch.nan_to_num(F.cross_entropy(logits_x_lb, y_lb, reduction='mean'))
                    sup_loss.backward(retain_graph=True)
                    labeled_grads = []
                    for name, param in model.named_parameters():
                        try:
                            labeled_grads.append(param.grad.view(-1))
                        except:
                            continue
                    labeled_grads = torch.cat(labeled_grads)#将 unlabeled_grads 中的梯度张量在默认的维度（通常是0维度，即沿着行的方向）上进行连接。
                    model.zero_grad()
                    unsup_loss = torch.nan_to_num(consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask))
                    # 获取无标签样本的梯度
                    unsup_loss.backward(retain_graph=True)  # 保留计算图以计算无标签样本的梯度
                    unlabeled_grads = []
                    for name, param in model.named_parameters():
                        try:
                            unlabeled_grads.append(param.grad.view(-1))
                        except:
                            continue
                    unlabeled_grads = torch.cat(unlabeled_grads)
                    model.zero_grad()
                    optimizer.zero_grad()
                    gradient_dot = torch.dot(labeled_grads, unlabeled_grads)
                    if mask.sum() > 0:  # 计算熵损失
                        ent_loss, _ = Mask.entropy_loss(mask=mask, logits_s=logits_x_ulb_s)
                    else:
                        ent_loss = 0.0
                    if gradient_dot < 0:
                        total_loss = sup_loss #+ args.ent_loss_ratio * ent_loss  # 不包含无标签损失
                    else:
                        total_loss = sup_loss + args.ulb_loss_ratio * unsup_loss + args.ent_loss_ratio * ent_loss  # 包含无标签损失
                    del logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, probs_x_lb
                    optimizer.zero_grad()######
                    scaler.scale(total_loss).backward()  # scaler.scale(loss) 用于对损失进行梯度缩放
                    torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)  # torch.nn.utils.clip_grad_norm_，梯度裁剪，防止梯度爆炸。  softmatch_algorithm.model
                    scaler.step(optimizer)
                    scaler.update()
                    #scheduler.step()
                    scheduler.step_update((epoch - 1) * num_batches + batch_idx)
                    if model_ema is not None:
                        model_ema.update(model)
                    model.zero_grad()
            else:
                sup_loss = criterion_train(logits_x_lb, y_lb)
                sup_loss.backward(retain_graph=True)
                labeled_grads = []
                for name, param in model.named_parameters():
                    try:
                        labeled_grads.append(param.grad.view(-1))
                    except:
                        continue
                labeled_grads = torch.cat(labeled_grads)
                model.zero_grad()
                unsup_loss = consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)
                # 获取无标签样本的梯度
                unsup_loss.backward(retain_graph=True)  # 保留计算图以计算无标签样本的梯度
                unlabeled_grads = []
                for name, param in model.named_parameters():
                    try:
                        unlabeled_grads.append(param.grad.view(-1))
                    except:
                        continue
                unlabeled_grads = torch.cat(unlabeled_grads)
                model.zero_grad()
                gradient_dot = torch.dot(labeled_grads, unlabeled_grads)
                '''print("SUP",sup_loss)
                print("UNSUP",unsup_loss)'''
                if mask.sum() > 0:   #计算熵损失
                    ent_loss, _ = Mask.entropy_loss(mask=mask, logits_s=logits_x_ulb_s)
                else:
                    ent_loss = 0.0
                if gradient_dot < 0:
                    #total_loss = sup_loss+ args.ent_loss_ratio*ent_loss  # 不包含无标签损失
                    total_loss = sup_loss
                    #print("T")
                else:
                    total_loss = sup_loss + args.ulb_loss_ratio * unsup_loss+args.ent_loss_ratio*ent_loss # 包含无标签损失
                    #print("F")
                #total_loss = sup_loss + args.ulb_loss_ratio * unsup_loss + args.ent_loss_ratio * ent_loss
                del logits_x_lb, logits_x_ulb_w, logits_x_ulb_s, probs_x_lb
                optimizer.zero_grad()  # 梯度清零，把loss关于weight的导数变成0
                total_loss.backward()  # Calculate the gradient.
                optimizer.step()  # 更新权重
                scheduler.step_update((epoch-1)*num_batches+batch_idx)
                if model_ema is not None:
                    model_ema.update(model)
                model.zero_grad()

            # print("sup", sup_loss)
            # print("unsup", unsup_loss)
            #print("ent", args.ent_loss_ratio*ent_loss)
            # print("total",total_loss)
            trainloss_meter.update(total_loss.item(), num_lb)
            # if model_ema is not None:
            #     model_ema.update(model)
            pbar.update()
        #scheduler.step()#############
        classification_rep = classification_report(all_true_labels, all_pred_labels,target_names=labeled_dataset.classes)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        accuracy = accury_meter.avg
        total_loss = trainloss_meter.avg
        end_time = time.time()
        elapsed_time = end_time - start_time
        # 使用 tqdm 输出训练信息
        tqdm.write(f"第 {epoch}/{args.epoch} 个周期，训练时间: {elapsed_time:.2f}s，"f"学习率: {lr:.8f},"f"训练准确率: {accuracy:.4f}%,"f"训练损失: {total_loss:.4f}")
        print(classification_rep)
        # 添加训练信息
        logger.info(f"Epoch {epoch}/{args.epoch}, Training Time: {elapsed_time:.2f}s,lr: {lr:.8f}, Accuracy: {accuracy:.4f}%，total_loss: {total_loss:.4f}")
        pbar.close()

        tb_log = TBLog(save_path, "tensorboard", use_tensorboard=True)
        log_dict = {'train_Accuracy': accuracy, 'train_Loss': total_loss}
        tb_log.update(log_dict, epoch)
        tb_log.writer.close()
        return accuracy, total_loss

@torch.no_grad()
def val(model, val_loader,epoch):
    rep_model = reparameterize_model(model)
    if epoch == 1:
        flops = FlopCountAnalysis(rep_model, inp)
        print(flop_count_table(flops, max_depth=5))
    rep_model.eval()
    global bestacc
    global Best_ACC
    true_labels = []
    predicted_labels = []
    val_list = []
    pred_list = []
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    for val_batch in tqdm(val_loader, total=val_batches, desc=f'Epoch {epoch}/{args.epoch}'):
        val_data, labels = val_batch
        for l in labels:
            val_list.append(l.data.item())
        val_data, labels = val_data.to(device), labels.to(device)
        # 模型前向传播
        logits_x_lb = rep_model(val_data)
        _, pred = torch.max(logits_x_lb, 1)  # 第一个张量 _ 记录了输出张量在维度1上的最大值，即每行的最大值。第二个张量 pred 记录了输出张量在维度1上最大值所对应的索引
        # loss = cross_entropy(logits_x_lb, labels)
        # print(logits_x_lb.shape, labels.shape)
        loss = criterion_val(logits_x_lb, labels)
        # loss = ce_loss(logits_x_lb, labels,reduction='mean')
        # 更新统计信息
        for p in pred:
            pred_list.append(p.data.item())
        acc1 = timm.utils.accuracy(logits_x_lb, labels, topk=(1,))
        loss_meter.update(loss.item(),labels.size(0))  # labels.size(0)是目标数据的大小，通常是批次的大小。这部分代码是用来确保损失值与样本数量相关联。在训练过程中，损失通常是关于整个批次的平均值。
        acc1_meter.update(acc1[0].item(), labels.size(0))
        # 保存真实标签和预测标签
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(pred.cpu().numpy())
    # 计算平均准确度
    acc = acc1_meter.avg
    loss = loss_meter.avg
    ba = balanced_accuracy_score(val_list, pred_list)
    ba=ba*100
    #  打印验证结果
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tbalance_acc:{:.3f}%\n'.format(loss, acc, ba))
    logger.info(f"Epoch {epoch}/{args.epoch}, Val Accuracy: {acc:.4f}%，val_loss: {loss:.4f}")
    tb_log = TBLog(save_path, "tensorboard", use_tensorboard=True)
    log_dict = {'val_Accuracy': acc, 'val_Loss': loss}
    tb_log.update(log_dict, epoch)
    tb_log.writer.close()

    print(classification_report(true_labels, predicted_labels, target_names=labeled_dataset.classes))
    torch.save(rep_model.state_dict(),os.path.join(save_path,f'epoch_{epoch}_acc_{acc:.4f}_model.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch}_acc_{acc:.4f}_NOREPmodel.pth'))
    logger.info("Model saved.")
    if acc >= bestacc:
        bestacc = acc
        torch.save(rep_model.state_dict(), os.path.join(save_path, f'best_epoch_{epoch}_acc_{bestacc:.4f}_model.pth'))
    if ba >= Best_ACC:
        Best_ACC = ba
        torch.save(rep_model.state_dict(), os.path.join(save_path, f'best_epoch_{epoch}_balance_acc_{Best_ACC:.4f}_model.pth'))
    return acc, loss


if __name__ == '__main__':
    seed_everything(seed=1)#42
    args = get_config()


    labeled_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV165(S0)\\train'
    unlabeled_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\unlabeled_set'
    val_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV165(S0)\\val'
    test_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV165(S0)\\test'

    # labeled_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV479\\train'
    # unlabeled_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\\approved_users_only\\unlabeled_set'
    # val_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV479\\val'
    # test_data_path = 'D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV479\\test'



    # 定义数据变换
    transform_labeledtrain = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        # transforms.Normalize(0.061, 0.140)
    ])
    weak_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        # transforms.Normalize(0.061, 0.140)
    ])
    strong_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        #RandAugmentMC(n=2, m=10),
        TargetedStrongAugment(n=6, window_size=56),
        transforms.ToTensor(),
    ])

    class TransformTwice:
        def __init__(self, weak_transform, strong_transform):
            self.weak_transform = weak_transform
            self.strong_transform = strong_transform
        def __call__(self, x):
            ulb_w = self.weak_transform(x)
            ulb_s = self.strong_transform(x)
            return ulb_w, ulb_s

    train_sampler = DistributedSampler if args.distributed else RandomSampler
    val_sampler = SequentialSampler

    # 加载有标签数据集
    labeled_dataset = datasets.ImageFolder(labeled_data_path, transform=transform_labeledtrain)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True,drop_last=True)  # ,shuffle=True,pin_memory=True
    # labeled_dataloader = DataLoader(labeled_dataset, batch_size=args.batch_size,sampler=train_sampler(labeled_dataset),num_workers=args.num_workers,pin_memory=True,drop_last=True)

    # 加载无标签数据集...
    unlabeled_dataset = datasets.ImageFolder(unlabeled_data_path,transform=TransformTwice(weak_transform, strong_transform))
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=(args.batch_size) * (args.ulb_ratio), shuffle=True,num_workers=args.num_workers, pin_memory=True,drop_last=True)  # unlabeled_dataset = [( (out1_weak_transform, out2_strong_transform), -1 )...shuffle=True,
    # unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=(args.batch_size)*(args.ulb_ratio),sampler=train_sampler(unlabeled_dataset),num_workers=args.num_workers,pin_memory=True,drop_last=True)

    val_dataset = datasets.ImageFolder(val_data_path, transform=transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,pin_memory=True, drop_last=False)
    # val_dataloader=DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler(val_dataset), num_workers=args.num_workers,pin_memory=True,drop_last=False)

    num_batches = min(len(labeled_dataloader), len(unlabeled_dataloader))
    val_batches = len(val_dataloader)
    # get_data_loader()
    # 设置模型
    device = torch.device(args.device)
    save_path = os.path.join(args.save_dir, args.save_name)
    #model = fastvit_t8(pretrained=args.use_pretrain)
    model=LWN_4VIEW()
    #model.head = nn.Linear(model.head.in_features, args.num_classes)
    inp = torch.randn(1, 3, 112, 112).to(device)
    model.to(device)
    if args.resume != None:
        state = torch.load(args.resume)
        model.load_state_dict(state['state_dict'])
        print("load from", args.resume)
    if args.use_ema:
        model_ema =ModelEmaV2(model, decay=args.EMA_decay, device=device)
    else:
        model_ema = None
    # 设置loss
    criterion_train=SoftTargetCrossEntropy() if args.use_cutmix else CrossEntropyLoss()
    #criterion_train = CrossEntropyLoss()#weight= (torch.tensor([0.1,0.25,0.65])).cuda(0)
    criterion_val = CrossEntropyLoss()

    # 设置优化器...
    # optimizer = optim.SGD(grouped_parameters, lr=args.lr,momentum=args.momentum, nesterov=True)
    #optimizer = optim.NAdam(params=model.parameters(), lr=args.lr,weight_decay=0.01,decoupled_weight_decay=True)#WD=0.05
    optimizer = optim.AdamW(params=model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-6)
    #scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_epochs * num_batches,num_training_steps=args.epoch * num_batches)

    scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=args.epoch * num_batches,
        lr_min=args.lr*0.01,
        cycle_limit=1,
        warmup_lr_init=0,#args.lr*0.001
        warmup_t=args.warmup_epochs * num_batches,
        t_in_epochs=False,
    )




    # 配置 logger
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = os.path.join(save_path,'logfile.txt')  # 创建了日志文件的路径（含日志文件名）。在这里，日志文件被命名为 'logfile.txt'，并存储在 save_path 目录下。
    logging.basicConfig(filename=log_file,level=logging.INFO)  # filename 参数指定了日志文件的路径，而 level 参数设置了日志记录的级别为 INFO，这意味着只有 INFO 级别及以上的日志信息会被记录。
    logger = logging.getLogger(__name__)
    logger.info(f"Use Device: {args.device} for training")  # 添加 GPU 使用信息
    logger.info(f"Model Architecture: {model}")  # softmatch_algorithm.model # 添加其他重要信息，例如模型结构、数据加载信息等
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info("Model training")

    Mask = Threshold(num_classes=args.num_classes, momentum=args.momtum,use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
    Pseudo = PseudoLabel(use_hard_label=args.use_hard_label, T=args.T, softmax=True,label_smoothing=0.0)  # label_smoothing=0.0

    # START TRAINING of SoftMatch
    # accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=3).to(device)
    print(count_parameters(model))  # softmatch_algorithm.model
    bestacc = 0.0
    Best_ACC=0.0

    log_dir = {}
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []

    for epoch in range(1, args.epoch + 1):
        epoch_list.append(epoch)
        train_acc, train_loss = FreeMatchSSLtrain_one_epoch(args=args, model=model, train_loader1=labeled_dataloader,train_loader2=unlabeled_dataloader, optimizer=optimizer, epoch=epoch,scheduler=scheduler, model_ema=model_ema)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list
        if args.use_ema:
            val_acc, val_loss = val(model=model_ema.module, val_loader=val_dataloader, epoch=epoch)
        else:
            val_acc, val_loss = val(model=model, val_loader=val_dataloader, epoch=epoch)
        # val_acc,val_loss= val(model=model,val_loader=val_dataloader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = bestacc

        # scheduler.step()

        plt.switch_backend('agg')  # 解决多线程绘图出错问题
        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(save_path + "/loss.png")
        plt.close(1)
        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(save_path + "/acc.png")
        plt.close(2)

