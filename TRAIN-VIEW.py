import json
import os
import random
from typing import Optional
import matplotlib.pyplot as plt
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from timm.scheduler import CosineLRScheduler
from timm.utils import accuracy, AverageMeter, ModelEma, ModelEmaV2  # 从timm.utils导入ema模块
from sklearn.metrics import accuracy_score,classification_report,balanced_accuracy_score
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.nn import CrossEntropyLoss
from torchvision.transforms import InterpolationMode

from models.fastvit import fastvit_t8, fastvit_sa24
from torchvision import datasets

from models.lwn import reparameterize_model, LWN, LWN_4VIEW

torch.backends.cudnn.benchmark = False
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES']="0"

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class_weights = (torch.tensor([0.278,0.245,0.148,0.329])).cuda(0)
# 定义训练过程
def train(model, device, train_loader, optimizer,scheduler,epoch,model_ema):
    model.train()
    loss_meter = AverageMeter()  #AverageMeter类来管理一些变量的更新,用于在训练过程中记录准确率等并计算其平均值。
    acc1_meter = AverageMeter()
    balanced_accuracy_meter = AverageMeter()
    total_num = len(train_loader.dataset) #样本数
    train_list = []
    pred_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device,non_blocking=True)  #Variable(target)
        #samples, targets = mixup_fn(data, target)
        train_list.extend(target.cpu().numpy())
        output = model(data)
        #output=model(data)
        _, pred = torch.max(output, 1)
        pred_list.extend(pred.cpu().numpy())
        optimizer.zero_grad()
        '''用了mixup就得用增强后的targets计算损失'''
        if use_amp:
            with torch.cuda.amp.autocast():  # 自动混合精度的计算
                loss = torch.nan_to_num(
                    criterion_train(output, target))  # 计算训练损失，并使用 torch.nan_to_num 将 NaN 替换为 0（防止 NaN 对训练造成问题）。
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)  # torch.nn.utils.clip_grad_norm_，梯度裁剪，防止梯度爆炸。
            # Unscales gradients and calls
            scaler.step(optimizer)
            # Updates the scale for next iteration
            scaler.update()
        else:
            loss = criterion_train(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step_update((epoch - 1) * len(train_loader) // BATCH_SIZE + batch_idx)

        if model_ema is not None:
            model_ema.update(model)
        torch.cuda.synchronize()  # torch.cuda.synchronize()可以用于同步CPU和GPU之间的计算，通常会在需要获取GPU计算结果的时候被使用
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        loss_meter.update(loss.item(), target.size(0))
        # print(target.shape)
        '''acc1 = accuracy(output, torch.max(targets,1)[1], topk=(1,))'''
        acc1 = accuracy(output, target, topk=(1,))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1[0].item(), target.size(0))
        '''loss.item()是当前批次（batch）的损失值。在PyTorch中，loss是一个包含损失值的张量，.item()方法用于获取这个张量中的数值。
        target.size(0) 是目标数据的大小，通常是批次的大小。这部分代码是用来确保损失值与样本数量相关联。在训练过程中，损失通常是关于整个批次的平均值。
        因此，将每个批次的损失值乘以当前批次中的样本数量，可以得到整个批次的总损失。
        loss_meter.update(loss.item(), target.size(0))将当前批次的损失值添加到 loss_meter 对象中，以便记录并计算整个训练过程中的平均损失。'''
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss = loss_meter.avg
    acc = acc1_meter.avg
    print('epoch:{}\tloss:{:.4f}\tacc:{:.4f}%'.format(epoch, ave_loss, acc, ))
    print(classification_report(train_list, pred_list, target_names=dataset_train.classes))
    return ave_loss, acc

# 验证过程
@torch.no_grad()
def val(model, device, val_loader):#= =model.eval()  with torch.no_grad()
    REPmodel=reparameterize_model(model)
    global Best_ACC
    global best_acc
    REPmodel.eval()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    total_num = len(val_loader.dataset)
    print(total_num, len(val_loader))
    val_list = []
    pred_list = []
    for (data, target) in val_loader:
        for t in target:
            val_list.append(t.data.item())
        data, target = data.to(device), target.to(device)
        output = REPmodel(data)
        #loss = F.cross_entropy(output, target)
        loss = criterion_val(output, target)   #output.data 返回 output 中的数据部分。
        _, pred = torch.max(output.data, 1)#第一个张量 _ 记录了输出张量在维度1上的最大值，即每行的最大值。第二个张量 pred 记录了输出张量在维度1上最大值所对应的索引
        for p in pred:
            pred_list.append(p.data.item())
        #print(pred_list)
        acc1= accuracy(output, target, topk=(1,))
        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1[0].item(), target.size(0))
    acc = acc1_meter.avg
    ave_loss = loss_meter.avg
    ba = balanced_accuracy_score(val_list, pred_list)
    ba=ba*100
    print('\nVal set: Average loss: {:.4f}\tAcc1:{:.3f}%\tbalance_acc:{:.3f}%\n'.format(ave_loss, acc, ba ))
    if acc >= Best_ACC:
        Best_ACC = acc
        torch.save(REPmodel.state_dict(), os.path.join(file_dir, f'best_epoch_{epoch}_acc_{Best_ACC:.4f}_model.pth'))
    if ba >= best_acc:
        best_acc = ba
        torch.save(REPmodel.state_dict(), os.path.join(file_dir, f'best_epoch_{epoch}_balance_acc_{best_acc:.4f}_model.pth'))
    if isinstance(REPmodel, torch.nn.DataParallel):
        state = {
            'epoch': epoch,
            'state_dict': REPmodel.module.state_dict(),
            'Best_ACC': Best_ACC
        }
        if use_ema:
            state['state_dict_ema'] = REPmodel.module.state_dict()
        torch.save(state, file_dir + "/" + 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
    else:
        torch.save(REPmodel.state_dict(), os.path.join(file_dir, f'epoch_{epoch}_acc_{acc:.4f}_model.pth'))
        torch.save(model.state_dict(), os.path.join(file_dir, f'epoch_{epoch}_acc_{acc:.4f}_NOREPmodel.pth'))
    return val_list, pred_list, loss_meter.avg, acc

# def seed_everything(seed=42):   #设置随机因子  设置了固定的随机因子，再次训练的时候就可以保证图片的加载顺序不会发生变化
#     os.environ['PYHTONHASHSEED'] = str(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
def seed_everything(seed):   #设置随机因子  设置了固定的随机因子，再次训练的时候就可以保证图片的加载顺序不会发生变化
    os.environ['PYHTONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
'''if  __name__  ==  '__main__':表示如果当前文件作为主程序直接运行，而不是被导入作为模块使用时，执行以下代码块中的内容。
如果该文件被其他文件导入，则代码块中的内容将不会被执行。'''


if __name__ == '__main__':
    #创建保存模型的文件夹
    file_dir =os.path.abspath('checkpoints/TMED-2-VIEW/ADAMW')
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # file_dir = os.path.join(project_root, 'checkpoints/FastVit/')
    if os.path.exists(file_dir):
        print('true')
        os.makedirs(file_dir,exist_ok=True)
    else:
        os.makedirs(file_dir)
    # 设置全局参数
    model_lr = 5e-4#3 4VIEWTRAIN 5E-4
    BATCH_SIZE = 16
    EPOCHS = 200
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    use_amp = False # 是否使用混合精度
    use_dp = False #是否开启dp方式的多卡训练
    classes = 4#3
    resume =None#'D:\cv\CLASSIFICATION\AS Classification by FastVIT\checkpoints\FastVit\model_300_80.539.pth'   #是否从某次训练结果（加载pth文件）继续训练
    CLIP_GRAD = 5.0
    Best_ACC = 0 #记录最高准确率
    best_acc=0
    use_ema=True
    model_ema_decay=0.999    #0.9998
    start_epoch=1
    seed=1
    seed_everything(seed)
    # 数据预处理7
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),

    ])
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.ImageFolder('D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV479\\train', transform=transform)  # target_transform=map_labels(class_to_idx)
    dataset_val = datasets.ImageFolder('D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV479\\val',transform=transform_val)  # ,target_transform=map_labels(class_to_idx)
    with open('class.txt', 'w') as file:  # 将AS严重程度对应数字记为txt和json文件
        file.write(str(dataset_train.class_to_idx))
    with open('class.json', 'w', encoding='utf-8') as file:
        file.write(json.dumps(dataset_train.class_to_idx))
    # 导入数据
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False,pin_memory=True,drop_last=False)

    # 实例化模型并且移动到GPU
    criterion_train = CrossEntropyLoss()#MultiClassFocalLossWithAlpha()#MultiCEFocalLoss(class_num=3, gamma=2, alpha=0.25)#  # SoftTargetCrossEntropy()

    criterion_val =CrossEntropyLoss()#MultiClassFocalLossWithAlpha()#MultiCEFocalLoss(class_num=3, gamma=2, alpha=0.25)#FocalLoss(gamma=2, weight=None)  # torch.nn.CrossEntropyLoss()(weight=(torch.tensor).cuda(0)
    #criterion_val = CrossEntropyLoss()
    #设置模型
    model=LWN_4VIEW()
    #model_ft.load_state_dict(torch.load('D:\\cv\\CLASSIFICATION\\Custom FastViT\\FastVIT_Demo\\checkpoints\\FastVit\\best.pth'))加载训练得到的最佳精度模型
    model.to(DEVICE)
    print(model)
    # 选择简单暴力的Adam优化器，学习率调低
    optimizer = optim.AdamW(model.parameters(),lr=model_lr,)#AdamW weight_decay=5e-4
    #cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    cosine_schedule = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=EPOCHS * len(train_loader)//BATCH_SIZE,
        lr_min=model_lr *0.01,
        cycle_limit=1,
        warmup_t=0,
        t_in_epochs=False,
    )
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    if torch.cuda.device_count() > 1 and use_dp:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    '''if use_ema:
        model_ema = ModelEma(
            model,
            decay=model_ema_decay,
            device=DEVICE,
            resume=resume)
    else:
        model_ema=None'''
    if use_ema:
        model_ema = ModelEmaV2(model, decay=model_ema_decay, device=DEVICE)
    else:
        model_ema = None

    # 训练与验证
    is_set_lr = False
    log_dir = {}
    train_loss_list, val_loss_list, train_acc_list, val_acc_list, epoch_list = [], [], [], [], []
    if resume and os.path.isfile(file_dir+"result.json"):
        with open(file_dir+'result.json', 'r', encoding='utf-8') as file:
            logs = json.load(file)#json.load函数加载文件中的内容，并将其赋值给变量`logs`。
            train_acc_list = logs['train_acc']
            train_loss_list = logs['train_loss']
            val_acc_list = logs['val_acc']
            val_loss_list = logs['val_loss']
            epoch_list = logs['epoch_list']


    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_list.append(epoch)
        log_dir['epoch_list'] = epoch_list
        '''训练并计算训练损失和准确率'''
        train_loss, train_acc = train(model, DEVICE, train_loader, optimizer, cosine_schedule, epoch, model_ema)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        log_dir['train_acc'] = train_acc_list
        log_dir['train_loss'] = train_loss_list
        if use_ema:
            val_list, pred_list, val_loss, val_acc = val(model_ema.module, DEVICE, val_loader)
        else:
            val_list, pred_list, val_loss, val_acc = val(model, DEVICE, val_loader)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        log_dir['val_acc'] = val_acc_list
        log_dir['val_loss'] = val_loss_list
        log_dir['best_acc'] = Best_ACC
        with open(file_dir + '/result.json', 'w', encoding='utf-8') as file:
            file.write(json.dumps(log_dir))
        print(classification_report(val_list, pred_list, target_names=dataset_train.class_to_idx))
        '''if epoch < 600:
            cosine_schedule.step()
        else:
            if not is_set_lr:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = 1e-6
                    is_set_lr = True'''
        plt.switch_backend('agg')  # 解决多线程绘图出错问题
        fig = plt.figure(1)
        plt.plot(epoch_list, train_loss_list, 'r-', label=u'Train Loss')
        # 显示图例
        plt.plot(epoch_list, val_loss_list, 'b-', label=u'Val Loss')
        plt.legend(["Train Loss", "Val Loss"], loc="upper right")
        plt.xlabel(u'epoch')
        plt.ylabel(u'loss')
        plt.title('Model Loss ')
        plt.savefig(file_dir + "/loss.png")
        plt.close(1)
        fig2 = plt.figure(2)
        plt.plot(epoch_list, train_acc_list, 'r-', label=u'Train Acc')
        plt.plot(epoch_list, val_acc_list, 'b-', label=u'Val Acc')
        plt.legend(["Train Acc", "Val Acc"], loc="lower right")
        plt.title("Model Acc")
        plt.ylabel("acc")
        plt.xlabel("epoch")
        plt.savefig(file_dir + "/acc.png")
        plt.close(2)