import matplotlib.pyplot as plt
import numpy as np


methods = ["LWN Baseline", "Rep LKFFN→Rep SKFFN", "Rep HWD Block→DWConv(S=2)"]
parameters = [0.759, 0.688, 0.686]
flops = [0.154, 0.131, 0.543]
accuracy = [97.5, 97.1, 97.0]

# 设置柱状图的宽度
bar_width = 0.35

# 设置图表的x轴位置
index = np.arange(len(methods))

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)

# 创建一个Figure对象
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=200)

# 在左侧y轴上绘制Parameters (M)的柱状图
ax1.set_ylabel('Parameters (M) & FLOPs (G)')
bars1 = ax1.bar(index - bar_width/2, parameters, bar_width, color='#F8CBAD', alpha=1, label='Parameters (M)')
bars2 = ax1.bar(index + bar_width/2, flops, bar_width, color='#9DC3E6', alpha=1, label='FLOPs (G)')
ax1.tick_params(axis='y')
ax1.set_ylim(0, max(max(parameters), max(flops)) * 1.5)

# 创建右侧y轴，绘制准确率的折线图
ax2 = ax1.twinx()
color = 'gray'
ax2.set_ylabel('Balanced Accuracy (%)')#color=color
ax2.plot(index, accuracy, color=color, marker='o', linestyle='-', linewidth=2, label='Balanced Accuracy (%)')
ax2.tick_params(axis='y')  #labelcolor=color
ax2.set_ylim(95, 98)  # 调整y轴范围，将Accuracy (%)曲线置于柱状图上方
ax2.set_yticks(np.linspace(97, 98, num=3))

# 设置x轴刻度为对应的方法名
ax1.set_xticks(index)
ax1.set_xticklabels(methods)

# 调整图形布局，在标题上方留出空白
fig.tight_layout(rect=[0, 0, 1, 0.95])

# 添加图例
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
# 标题
#plt.title('Model Components Ablation Experiments')
# 显示图表
plt.show()





# methods = ["Efficient SSL", "RandAugment(n=4,m=10)", "Norm Loss"]
# accuracy = [97.8, 97.3, 97.3]
#
# # 设置柱状图的宽度
# bar_width = 0.35
#
# # 设置图表的x轴位置
# index = np.arange(len(methods))
#
# font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}
# plt.rc('font', **font)
#
# # 创建一个Figure对象
# fig, ax = plt.subplots(figsize=(10, 6))
#
# # 定义每个柱体的颜色
# colors = ['tab:red', 'tab:blue', 'gray']
#
# # 在y轴上绘制Accuracy的柱状图
# ax.set_xlabel('Method')
# ax.set_ylabel('Accuracy (%)')
# bars = ax.bar(index, accuracy, bar_width, color=colors, alpha=0.6, label='Accuracy (%)')
# ax.tick_params(axis='y')
# ax.set_ylim(95, 100)
#
# # 设置x轴刻度为对应的方法名
# ax.set_xticks(index)
# ax.set_xticklabels(methods)
#
# # 调整图形布局，在标题上方留出空白
# fig.tight_layout(rect=[0, 0, 1, 0.95])
#
# # 添加图例
# ax.legend(loc='upper right')
#
# # 标题
# plt.title('Comparison of Methods by Accuracy')
#
# # 显示图表
# plt.show()


# 方法名
import numpy as np
import matplotlib.pyplot as plt

# 方法名
methods = ["Improved SSL Baseline", "TSA→RandAugment", "ALF→Conventional Loss Function"]

# 示例数据，假设有每种方法的多个测试准确率结果
data = {
    "Improved SSL Baseline": [97.88, 97.79, 97.78, 97.86, 97.83],
    "TSA→RandAugment": [97.80, 97.52, 97.67, 97.67, 97.42],
    "ALF→Conventional Loss Function": [97.36, 97.17, 97.08, 96.33, 97.23]
}

# 将数据转换为列表
data_list = [data[method] for method in methods]

# 设置颜色
colors = ['#FFCCFF', '#9DC3E6', '#CDE9BB']

font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}
plt.rc('font', **font)
# 创建一个Figure对象
fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

# 绘制箱线图，使用不同的颜色
box = ax.boxplot(data_list, patch_artist=True, widths=0.35)

# 设置箱子的颜色
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# 设置中位数线条颜色为黑色
for median in box['medians']:
    median.set_color('black')

# 设置x轴刻度为对应的方法名
ax.set_xticks(np.arange(1, len(methods) + 1))
ax.set_xticklabels(methods)

# 设置y轴标签
ax.set_ylabel('Balanced Accuracy (%)')

# 设置标题
#plt.title('Efficient SSL Methods Ablation Experiments')

# 显示图表
plt.show()


