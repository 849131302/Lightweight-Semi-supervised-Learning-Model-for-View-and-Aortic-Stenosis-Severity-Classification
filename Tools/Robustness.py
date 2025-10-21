import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def plot(methods, accuracy, image_paths):
    # 创建一个Figure对象
    font = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)


    ax.plot(methods, accuracy, marker='o', linestyle='-', linewidth=2, label='',color='gray')#Balanced Accuracy (%)

    # 设置图表标题

    # 设置图表标题，并增加标题与图表之间的距离
    #plt.title('Model Robustness To Image Quality Degradation', )#pad=20
    # 设置x轴标签
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods)
    ax.set_xlabel('Pixels Removed (%)')

    # 设置y轴范围
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(50, 100, num=6))
    #plt.legend(loc='upper left', bbox_to_anchor=(0, 0.9))
    ax.set_ylabel('Balanced Accuracy (%)')
    # 调整左侧y轴的位置
    # ax.spines['left'].set_position(('outward', 20))
    # ax.spines['right'].set_position(('outward', 20))
    # 调整子图布局，使得折线图所占的空间更小


    # 在每个x轴标签的正下方插入图片
    for i, (method, image_path) in enumerate(zip(methods, image_paths)):
        # 打开灰度图片
        img = Image.open(image_path).convert('L')

        # 创建图像盒子
        imagebox = OffsetImage(img, zoom=0.8, cmap='gray')  # 设置cmap='gray'以确保显示灰度图像

        # 获取x轴标签的位置
        x = i
        y = 17.6# 将图片整体下移一定距离

        # 创建注释盒子并添加到图表中
        ab = AnnotationBbox(imagebox, (x, y), xycoords='data', frameon=False)
        ax.add_artist(ab)

    # 显示图表
    plt.show()

# 调用函数并传入数据和图片路径
methods = ["0", "10", "20", "30", "40", "50"]
accuracy = [97.879, 97.687, 96.320, 95.311, 94.878, 92.311]
image_paths = ["D:\CV\LWM-SSL\TMED\TMED-2\TMED-4VIEW\DEV479\\test\A4C\\582s1_13.png",
               "D:\CV\LWM-SSL\TMED\TMED-2\simulate_ultrasound_dropout\\10%\A4C\\582s1_13_simulate_ultrasound_dropout_.png",
               "D:\CV\LWM-SSL\TMED\TMED-2\simulate_ultrasound_dropout\\20%\A4C\\582s1_13_simulate_ultrasound_dropout_.png",
               "D:\CV\LWM-SSL\TMED\TMED-2\simulate_ultrasound_dropout\\30%\A4C\\582s1_13_simulate_ultrasound_dropout_.png",
               "D:\CV\LWM-SSL\TMED\TMED-2\simulate_ultrasound_dropout\\40%\A4C\\582s1_13_simulate_ultrasound_dropout_.png",
               "D:\CV\LWM-SSL\TMED\TMED-2\simulate_ultrasound_dropout\\50%\A4C\\582s1_13_simulate_ultrasound_dropout_.png"]
plot(methods, accuracy, image_paths)
print(len(methods))
print(len(image_paths))

for method, image_path in zip(methods, image_paths):
    print(method, image_path)