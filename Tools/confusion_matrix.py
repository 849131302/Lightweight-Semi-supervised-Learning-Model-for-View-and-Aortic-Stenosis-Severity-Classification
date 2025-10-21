from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score,classification_report,balanced_accuracy_score
from timm.utils import AverageMeter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from torch.nn import functional as F
import warnings
from models import fastvit_t8
from models.lwn import SLW_Net, reparameterize_model, SLW_Net_4VIEW
warnings.filterwarnings("ignore",category=UserWarning)



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#model = SLW_Net()
model = SLW_Net_4VIEW()

print(model)
model=reparameterize_model(model)

#resume=('D:\CV\AS\\3VIEWbest_epoch_182_balance_acc_98.0163_model.pth')
resume=('D:\CV\AS\\best_epoch_79_balance_acc_98.3636_model.pth')

state_dict = torch.load(resume)  # load_model_without_module(resume)                 若为加载某一epoch，则state = torch.load(resume)，model_ft.load_state_dict(state['state_dict'])
model.load_state_dict(state_dict)
transform_test = transforms.Compose([
    #transforms.Resize((112,112)),#interpolation=InterpolationMode.LANCZOS     HAMMING95.259  BICUBIC95.187 BILINEAR95.261  LANCZOS95.042
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
dataset_test = datasets.ImageFolder("D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV479\\test", transform=transform_test)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

@torch.no_grad()
def test(model, test_loader):
    model.to(device)
    model.eval()
    global bestacc
    true_labels = []
    predicted_labels = []
    test_list = []
    pred_list = []
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    for test_batch in tqdm(test_loader, total=len(test_loader)):
        test_data, labels = test_batch
        for l in labels:
            test_list.append(l.data.item())
        test_data, labels =test_data.to(device), labels.to(device)
        # 模型前向传播
        logits_x_lb= model(test_data)
        _, pred = torch.max(logits_x_lb, 1)#第一个张量 _ 记录了输出张量在维度1上的最大值，即每行的最大值。第二个张量 pred 记录了输出张量在维度1上最大值所对应的索引
        #print(logits_x_lb.shape, labels.shape)
        loss = F.cross_entropy(logits_x_lb, labels,reduction="mean")
        # 更新统计信息
        for p in pred:
            pred_list.append(p.data.item())
        #print(pred_list)
        acc1 = timm.utils.accuracy(logits_x_lb, labels, topk=(1,))
        loss_meter.update(loss.item(), labels.size(0)) #labels.size(0)是目标数据的大小，通常是批次的大小。这部分代码是用来确保损失值与样本数量相关联。在训练过程中，损失通常是关于整个批次的平均值。
        acc1_meter.update(acc1[0].item(), labels.size(0))
        #ba_meter.update(ba, labels.size(0))
        # 保存真实标签和预测标签
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(pred.cpu().numpy())
    # 计算平均准确度
    ba = balanced_accuracy_score(test_list,pred_list)
    acc = acc1_meter.avg
    loss = loss_meter.avg
    #ba=ba_meter.avg
    #  打印验证结果
    print('\nTest set: Average loss: {:.4f}\tAcc1:{:.3f}%\tBalance ACC:{:.3f}%\n'.format(loss, acc,ba*100))
    return acc,loss,true_labels, predicted_labels

acc,loss,true_labels, predicted_labels=test(model, test_loader)
print(classification_report(true_labels, predicted_labels, target_names=dataset_test.classes))


#classes = ['Other','PLAX', 'PSAX']
classes = ['A2C', 'A4C', 'PLAX', 'PSAX']
confusion_matrix_data = confusion_matrix(true_labels, predicted_labels)
proportion = []
for i in confusion_matrix_data:
    for j in i:
        temp = j / (np.sum(i))
        proportion.append(temp)

pshow = []
for i in proportion:
    pt = "%.1f%%" % (i * 100)
    pshow.append(pt)




# proportion = np.array(proportion).reshape(3, 3)
# pshow = np.array(pshow).reshape(3, 3)
proportion = np.array(proportion).reshape(4, 4)#将数组重新排列为4行4列的形状
pshow = np.array(pshow).reshape(4, 4)

config = {
    "font.family": 'Times New Roman',
}
rcParams.update(config)

# 创建自定义颜色映射
# colors = ['#FFFFFF', '#2D8CFF']  # 从白色到 #92D050
# cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)

plt.figure(figsize=(10, 6), dpi=200)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues,alpha=1)#cmap=plt.cm.Blues
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=12)#fontsize=12 设置刻度标签的字体大小为 12。
plt.yticks(tick_marks, classes, fontsize=12)

thresh = confusion_matrix_data.max() / 2.




# iters = np.reshape([[[i, j] for j in range(3)] for i in range(3)], (confusion_matrix_data.size, 2))
# for i, j in iters:
#     if (i == j):
#         plt.text(j, i - 0.12, format(confusion_matrix_data[i, j]), va='center', ha='center', fontsize=12, color='white', weight=5)
#         plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color='white')
#     else:
#         plt.text(j, i - 0.12, format(confusion_matrix_data[i, j]), va='center', ha='center', fontsize=12)
#         plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12)
# font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}

iters = np.reshape([[[i, j] for j in range(4)] for i in range(4)], (confusion_matrix_data.size, 2))
for i, j in iters:
    if (i == j):
        plt.text(j, i - 0.12, format(confusion_matrix_data[i, j]), va='center', ha='center', fontsize=12, color='white', weight=5)
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color='white')
    else:
        plt.text(j, i - 0.12, format(confusion_matrix_data[i, j]), va='center', ha='center', fontsize=12)
        plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12)
font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}

plt.rc('font', **font)

plt.ylabel('True label', fontsize=16)
plt.xlabel('Predict label', fontsize=16)
plt.tight_layout()
plt.show()

