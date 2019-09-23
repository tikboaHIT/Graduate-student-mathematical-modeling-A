# huaiwei_math
研究生数学建模A题


## 文件说明
regression.py中包含对特征的构建，预测值的回归
data_path 所有csv文件拼接后的路径，并且将拼接后的csv文件转为h5格式，加快读取速度
save_path 保存模型的路径
输出的oof.csv文件，可以根据其对模型进行评估



## 模型构建
模型构建参考：http://rongzijing.win/index.php/archives/168/  
第一层，特征提取模型，GBDT(lightgbm)  
第二层，回归模型：LR(tensorflow)  
针对GBDT输出的叶子节点的特征，进行回归