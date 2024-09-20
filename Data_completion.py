# import pandas as pd
#
# # 读取包含缺失值的数据集
# df = pd.read_csv('data/BJ/zhongxin.csv')
#
# # 计算每一列的均值
# mean_values = df.mean()
#
# # 使用均值填充缺失值
# df_filled = df.fillna(mean_values)
#
# # 保存补全后的数据集到新文件
# df_filled.to_csv('data/BJ/zhongxin1.csv', index=False)



import pandas as pd

# 读取包含缺失值的 CSV 文件
df = pd.read_csv('data/BJ/zhongxin.csv')

# 使用前向填充的平均值来补全缺失值
df_ffill = df.fillna(method='ffill')

# 使用后向填充的平均值来补全缺失值
df_bfill = df.fillna(method='bfill')

# 保存补全后的数据集到新文件
df_ffill.to_csv('data/BJ/zhongxin-ffill.csv', index=False)
df_bfill.to_csv('data/BJ/zhongxin-bfill.csv', index=False)
