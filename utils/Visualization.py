import numpy as np
import matplotlib.pyplot as plt
import os

def plot_boxes(train_aucs, val_aucs, tasks, save_name=None):
    labels = ['train_aucs', 'val_aucs']
    colors = ['lightgreen', 'lightblue']
    plt.figure(figsize=(10, 6))

    for i in range(len(tasks)):
        data = [train_aucs[:, i], val_aucs[:, i]]
        bplot = plt.boxplot(data, notch=True, patch_artist=True, labels=labels,positions=(i + 1, i + 1.5), widths=0.5) 
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # x_position = [1.25, 2.25, 3.25, 4.25]
    x_position = [1.25, 2.25 ]
    x_position_fmt=tasks
    plt.xticks(x_position, x_position_fmt)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylim(0.5, 1.1)
    plt.ylabel('AUCs')
    plt.grid(linestyle="--", alpha=0.3)  #绘制图中虚线 透明度0.3
    plt.legend(bplot['boxes'],labels,loc='upper right')  #绘制表示框，右下角绘制
    plt.savefig(save_name, dpi=300)
    plt.close()
    # plt.show()



def plot_bar(select_featuresbytimes, save_folder, title=None):
    values = list(select_featuresbytimes.values())
    names = list(select_featuresbytimes.keys())
    plt.figure(figsize=(10, 6))
    plt.bar(names, values)
    plt.xlabel('features')
    plt.ylabel('times')
    plt.xticks(names, names, rotation=90, fontsize=6)
    for a, b in zip(names, values):
        plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=8)  # fontsize表示柱坐标上显示字体的大小
    
    plt.title(title)
    plt.savefig(os.path.join(save_folder, title + '_CV_features.png'), bbox_inches='tight')
    # plt.show()
    plt.close()


def plotCompareFeautures(features1, features2, label, save_folder):
    values1, values2 =[], []
    keys1 = list(features1.keys())
    keys2 = list(features2.keys())
    keys = keys1 + [x for x in keys2 if x not in keys1]

    diff1 = [x for x in keys if x not in keys1]
    diff2 = [x for x in keys if x not in keys2]

    for d in diff1:
        features1[d] = 0

    for d in diff2:
        features2[d] = 0

    values1 = [features1[k] for k in keys]
    values2 = [features2[k] for k in keys]    

    plt.figure(figsize=(10, 6))
    x = np.arange(len(keys))  # x轴刻度标签位置
    width = 0.4  # 柱子的宽度
    # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
    # x - width/2，x + width/2即每组数据在x轴上的位置
    plt.bar(x - width/2, values1, width, label='MT')
    plt.bar(x + width/2, values2, width, label='ST')
    plt.xlabel('features')
    plt.ylabel('times')

    plt.legend()
    plt.xticks(x, keys, rotation=90, fontsize=6)
    plt.title(label)
    plt.savefig(os.path.join(save_folder, label + '_compare_features.png'), bbox_inches='tight')
    # plt.show()
    plt.close()
