import numpy as np
import umap
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
import warnings
from models.lwn import SLW_Net, reparameterize_model, SLW_Net_4VIEW,SLW_Net
warnings.filterwarnings("ignore",category=UserWarning)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#model = SLW_Net_4VIEW()
model = SLW_Net()
model = reparameterize_model(model)
resume = 'D:\CV\AS\\3VIEWbest_epoch_182_balance_acc_98.0163_model.pth'
#resume=('D:\CV\AS\\best_epoch_79_balance_acc_98.3636_model.pth')
state_dict = torch.load(resume)
model.load_state_dict(state_dict)
model.to(device)
model=model.eval()

transform_test = transforms.Compose([
    #transforms.Resize((256,256),interpolation=InterpolationMode.BICUBIC),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])
dataset_test = datasets.ImageFolder("D:\CV\AS\TMED\TMED-3VIEW\\test", transform=transform_test)
test_loader = DataLoader(dataset_test, batch_size=16, shuffle=False)

# 定义一个列表用于保存提取的features
extracted_features = []

# 定义钩子函数，在forward方法执行特定层时触发
def hook_fn(module, input, output):
    extracted_features.append(output)

# 注册钩子函数，将其绑定到需要提取features的层
target_layer = model.head[-2]
hook_handle = target_layer.register_forward_hook(hook_fn)

# 正向传播，触发钩子函数并提取features
for test_batch in tqdm(test_loader, total=len(test_loader)):
    test_data, labels = test_batch
    test_data, labels = test_data.to(device), labels.to(device)
    # 模型前向传播
    logits_x_lb = model(test_data)

# 移除钩子
hook_handle.remove()

# 将提取的特征向量拼接起来
extracted_features = torch.cat(extracted_features, dim=0)

# 创建UMAP模型，可以指定降维后的目标维度
#umap_model = umap.UMAP(n_components=2)
# umap_model = umap.UMAP(n_components=2,  metric='cosine')
umap_model = umap.UMAP(n_components=2,min_dist=0.3)

# 将特征矩阵输入UMAP模型，得到降维后的坐标
umap_result = umap_model.fit_transform(extracted_features.cpu().numpy())

# 获取所有数据的标签
all_labels = torch.cat([batch_labels for _, batch_labels in test_loader], dim=0)

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)
plt.figure(figsize=(10, 6), dpi=200)
# 可视化降维结果
#plt.scatter(umap_result[:, 0], umap_result[:, 1], c=all_labels.cpu(), cmap='Spectral',s=5)
scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=all_labels.cpu(), cmap='viridis', s=5)
#plt.title('UMAP Visualization')
# 添加颜色柱体，并设置刻度和范围
cbar = plt.colorbar(scatter)



plt.show()
