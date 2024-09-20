import matplotlib.pyplot as plt
from utils import  make_save_dir
import argparse
import os

def plot_evalu_trends(file_name, save_path, xlim=[], ylim=[]):
    file = open(file_name)  # 打开文档
    lines = file.readlines()  # 读取文档数据
    # epoch = list(1, range(len(lines))+1) #epoch可以直接赋值，不放心的就用下面epoch的代码
    epoch = []
    R2 = []
    MSE = []
    RMSE = []
    MAE = []
    for line in lines:
        # split用于将每一行数据用自定义的符号（逗号）分割成多个对象
        epoch.append(int(line.split(',')[2].split('/')[0].split('[')[1])) # 以,为分隔符第二个中的以/为分隔符第0个里的以[为分隔符的第1个。
        R2.append(float(line.split(',')[9].split(':')[1]))
        MSE.append(float(line.split(',')[10].split(':')[1]))
        RMSE.append(float(line.split(',')[11].split(':')[1]))
        MAE.append(float(line.split(',')[12].split(':')[1]))
    plt.figure()
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('evaluation indicators')  # y轴标签
    plt.title('Trends in evaluation indicators')  # 标题
    # 手动指定 x 轴和 y 轴的起始点
    # plt.xlim(xlim[0], xlim[1])
    # plt.ylim(ylim[0], ylim[1])
    # plt.plot(epoch, R2, linewidth=1, label="R2")
    plt.plot(epoch, R2, linewidth=1, label="R2")
    plt.plot(epoch, MSE, linewidth=1, label="MSE")
    plt.plot(epoch, RMSE, linewidth=1, label="RMSE")
    plt.plot(epoch, MAE, linewidth=1, label="MAE")
    plt.legend()
    plt.savefig(os.path.join(save_path, "Evaluation.png"))
    # plt.show()
    file.close()



if __name__ == "__main__":
    file_name = 'CH4_runs_2579-TECA/32/train/exp6/logs.txt'
    save_path = 'CH4_runs_2579-TECA/32/train/exp6'
    plot_evalu_trends(file_name, save_path, xlim=[0, 50], ylim=[0, 1])
