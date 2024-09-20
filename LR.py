# import math
#
# import matplotlib.pyplot as plt
#
#
# def plot_loss(losses):
#     fig, ax = plt.subplots()
#     ax.set_xlabel('Epoch')
#     ax.set_ylabel('Loss')
#     ax.set_title('Loss over Epochs')
#     # 开启交互模式，允许动态更新图表
#     plt.ion()
#
#     def update_loss(loss):
#         losses.append(loss)
#
#     def plot():
#         ax.clear()
#         ax.plot(losses)
#         plt.pause(0.01)
#
#     def close():
#         # 关闭交互模式
#         plt.ioff()
#         # 显示图表窗口
#         plt.show()
#
#     return update_loss, plot, close
#
#
# # Example usage:
# update_loss, plot, close = plot_loss([])
#
# for epoch in range(100):
#     loss = math.sin(epoch*0.2) # calculate loss for current epoch
#     update_loss(loss)
#     plot()
#
# close()
#



###################################################################################
# 评价指标变化趋势可视化

import matplotlib.pyplot as plt
import os


file_name= 'C:\\Users\\pc\\Desktop\\TCN-OFFICAL\\GUN_TCN\\TCN-last\\SO2_runs_TECA\\256\\train\exp11\\logs.txt'
save_path = 'C:\\Users\\pc\\Desktop\\TCN-OFFICAL\\GUN_TCN\\TCN-last\\SO2_runs_TECA\\256\\train\exp11'
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
plt.xlim(0, 50)
plt.ylim(0, 0.2)
# plt.plot(epoch, R2, linewidth=1, label="R2")
plt.plot(epoch, MSE, linewidth=1, label="MSE")
plt.plot(epoch, RMSE, linewidth=1, label="RMSE")
plt.plot(epoch, MAE, linewidth=1, label="MAE")
plt.legend()
plt.savefig(os.path.join(save_path, "evaluation_trends.png"))
# plt.show()
file.close()











