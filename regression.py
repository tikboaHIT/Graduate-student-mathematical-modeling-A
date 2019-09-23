import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import json
from time import time
from scipy.stats import norm, skew
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")


data_path = ""
save_path = ""

train_data = pd.read_hdf(data_path,key='data')


"""减少内存"""
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def feature_engineering(data):
    eps = 1e-3

    #构造发射信号到接收的水平距离
    print("构造距离特征-----------")
    data["d"] = ((data["Cell X"] - data["X"]) ** 2 + (data["Cell Y"] - data["Y"]) ** 2) ** (1 / 2)
    A = 1/np.tan(data["Azimuth"]+eps)
    B = -1
    C = (data["Cell Y"]-1/np.tan(data["Azimuth"]+eps)*data["Cell X"])
    data["d_v"] = np.abs(A*data["X"]+B*data["Y"]+C) / np.sqrt((A ** 2) + (B ** 2))

    #判断是否为背面
    print("构造背面特征-----------")
    data["Y_"] = -1*np.tan(data["Azimuth"]+eps) * data["X"] + (data["Cell Y"] + np.tan(data["Azimuth"]+eps) * data["Cell X"])
    data["flag back"] = data["Y"] < data["Y_"]
    data[(90 <= data["Azimuth"]) & (data["Azimuth"] <= 270)]["flag back"] = data[(90 <= data["Azimuth"]) & (data["Azimuth"] <= 270)]["Y_"] < \
                                                                            data[(90 <= data["Azimuth"]) & (data["Azimuth"] <= 270)]["Y"]

    data[((0 <= data["Azimuth"]) & (data["Azimuth"] < 90)) | (( 270 < data["Azimuth"]) & (data["Azimuth"] <= 360))]["flag back"] = \
        data[((0 <= data["Azimuth"]) & (data["Azimuth"] < 90)) | (( 270 < data["Azimuth"]) & (data["Azimuth"] <= 360))]["Y_"] > \
        data[((0 <= data["Azimuth"]) & (data["Azimuth"] < 90)) | (( 270 < data["Azimuth"]) & (data["Azimuth"] <= 360))]["Y"]

    #cell中平均统计特征
    print("构造cell统计特征-----------")
    Altitude_mean = data.groupby(["Cell Index"])["Altitude"].mean()
    Altitude_std = data.groupby(["Cell Index"])["Altitude"].std()
    Altitude_mode = data.groupby(["Cell Index"])["Altitude"].agg(lambda x: np.mean(pd.Series.mode(x)))
    Altitude_median = data.groupby(["Cell Index"])["Altitude"].median()

    data["Altitude_mean_pre_cell"] = data["Cell Index"].map(Altitude_mean)
    data["Altitude_std_pre_cell"] = data["Cell Index"].map(Altitude_std)
    data["Altitude_mode_pre_cell"] = data["Cell Index"].map(Altitude_mode)
    data["Altitude_median_pre_cell"] = data["Cell Index"].map(Altitude_median)

    Building_Height_mean = data.groupby(["Cell Index"])["Building Height"].mean()
    Building_Height_std = data.groupby(["Cell Index"])["Building Height"].std()
    Building_Height_mode = data.groupby(["Cell Index"])["Building Height"].agg(lambda x: np.mean(pd.Series.mode(x)))
    Building_Height_median = data.groupby(["Cell Index"])["Building Height"].median()

    data["Building_Height_mean_pre_cell"] = data["Cell Index"].map(Building_Height_mean)
    data["Building_Height_std_pre_cell"] = data["Cell Index"].map(Building_Height_std)
    data["Building_Height_mode_pre_cell"] = data["Cell Index"].map(Building_Height_mode)
    data["Building_Height_median_pre_cell"] = data["Cell Index"].map(Building_Height_median)

    Clutter_Index_mean = data.groupby(["Cell Index"])["Clutter Index"].mean()
    Clutter_Index_std = data.groupby(["Cell Index"])["Clutter Index"].std()
    Clutter_Index_mode = data.groupby(["Cell Index"])["Clutter Index"].agg(lambda x: np.mean(pd.Series.mode(x)))
    Clutter_Index_median = data.groupby(["Cell Index"])["Clutter Index"].median()

    data["Clutter_Index_mean_pre_cell"] = data["Cell Index"].map(Clutter_Index_mean)
    data["Clutter_Index_std_pre_cell"] = data["Cell Index"].map(Clutter_Index_std)
    data["Clutter_Index_mode_pre_cell"] = data["Cell Index"].map(Clutter_Index_mode)
    data["Clutter_Index_median_pre_cell"] = data["Cell Index"].map(Clutter_Index_median)


    #角度
    print("构造角度特征-----------")
    data["sum_theta"] = data["Electrical Downtilt"] + data["Mechanical Downtilt"]
    data["tan"] = np.tan(data["sum_theta"] * np.pi / 180.0)

    #高度
    print("构造高度特征-----------")
    data["Cell sum high"] = data["Height"] + data["Cell Altitude"]
    data["sum high"] = data["Altitude"]+data["Building Height"]
    data["high diff"] = data["Cell sum high"]-data["sum high"]
    data["theta high"] = data["Cell sum high"]-data["tan"]*data["d"]
    data["signal diff"] = data["high diff"]-data["theta high"]

    #传播路径损耗
    print("构造PL特征-----------")
    data["A"] = 46.3 + 33.9*np.log10(data["Frequency Band"])-13.28*np.log10(data["Cell sum high"])
    data["B"] = 44.9 - 6.65*np.log10(data["Cell sum high"])
    data["PL"] = (data["A"] + data["B"] * np.log10(data["d"]+eps))

    #构造.....特征
    print("构造...特征-----------")
    data["Clutter equal"] = data["Clutter Index"] == data["Cell Clutter Index"]
    data["middle"] = data["Cell Index"].apply(lambda x: int(str(x)[2:-2]))

    print("编码-----------")
    Frequency_Band = {2585.0:0 , 2604.8:1, 2624.6:2}
    data["Frequency_Band"] = data["Frequency Band"].map(Frequency_Band)

    print("目标编码--------")
    # data["RSRP"] = np.log1p(-1 * data["RSRP"])
    #data["RSRP"] = np.log10(-1 * data["RSRP"] / 100)

    return data


def feature_select(data):
    # feature = ['Cell Index', 'Cell X', 'Cell Y', 'Height', 'Azimuth',
    #    'Electrical Downtilt', 'Mechanical Downtilt', 'Frequency Band',
    #    'RS Power', 'Cell Altitude', 'Cell Building Height',
    #    'Cell Clutter Index', 'X', 'Y', 'Altitude', 'Building Height',
    #    'Clutter Index', 'RSRP', 'd', 'd_v', 'Y_', 'flag back',
    #    'Altitude_mean_pre_cell', 'Altitude_std_pre_cell',
    #    'Altitude_mode_pre_cell', 'Altitude_median_pre_cell',
    #    'Building_Height_mean_pre_cell', 'Building_Height_std_pre_cell',
    #    'Building_Height_mode_pre_cell', 'Building_Height_median_pre_cell',
    #    'Clutter_Index_mean_pre_cell', 'Clutter_Index_std_pre_cell',
    #    'Clutter_Index_mode_pre_cell', 'Clutter_Index_median_pre_cell',
    #    'sum_theta', 'tan', 'Cell sum high', 'sum high', 'high diff',
    #    'theta high', 'signal diff', 'A', 'B', 'PL', 'Clutter equal', 'middle']
    feature = ['Cell X', 'Cell Y', 'Height', 'Azimuth', 'Electrical Downtilt',
               'Mechanical Downtilt', 'RS Power', 'Cell Altitude', 'Cell Building Height',
               'X', 'Y', 'Altitude', 'Building Height', 'd', 'd_v', 'Y_', 'flag back',
               'Altitude_mean_pre_cell', 'Altitude_std_pre_cell','Altitude_mode_pre_cell',
               'Altitude_median_pre_cell','Building_Height_mean_pre_cell', 'Building_Height_std_pre_cell',
               'Building_Height_mode_pre_cell', 'Building_Height_median_pre_cell','Clutter_Index_mean_pre_cell',
               'Clutter_Index_std_pre_cell','Clutter_Index_mode_pre_cell', 'Clutter_Index_median_pre_cell',
               'sum_theta', 'tan', 'Cell sum high', 'sum high', 'high diff',
               'theta high', 'signal diff', 'A', 'B', 'PL', 'Clutter equal', 'middle',
               'Cell Clutter Index', 'Clutter Index', 'Frequency_Band']

    X = data[feature]
    Y = data["RSRP"]

    return X, Y

train_data = reduce_mem_usage(train_data)
train_data = feature_engineering(train_data)



n_folds = 10
params = dict(n_estimators=3000,
              metric=["rmse"],
              num_leaves=127,
              min_data_in_leaf=20,
              learning_rate=0.1,
              min_sum_hessian_in_leaf=0.002,
              colsample_bytree=0.8,
              subsample=0.8,
              reg_alpha=0.0,
              reg_lambda=0.0,
              max_bin=511)

folds = KFold(n_folds, shuffle=True, random_state=2019)
oof_df = pd.DataFrame()
eval_result = {}
Cell_Index = train_data["Cell Index"].unique()
for idx, (train_idx, valid_idx) in enumerate(folds.split(Cell_Index)):
    t_cell = Cell_Index[train_idx]
    v_cell = Cell_Index[valid_idx]

    x_train = train_data[train_data["Cell Index"].isin(t_cell)]
    x_valid = train_data[train_data["Cell Index"].isin(v_cell)]

    curr_oof = x_valid[['Unnamed: 0', 'RSRP']]
    t_x, t_y = feature_select(x_train)
    v_x, v_y = feature_select(x_valid)

    model = lgb.LGBMRegressor(**params)
    print("Fold", idx, "-" * 30)
    model.fit(t_x, t_y,
              eval_set=[(t_x, t_y), (v_x, v_y)],
              early_stopping_rounds=100,
              verbose=10,
              callbacks=[lgb.record_evaluation(eval_result)]
              )


    joblib.dump(model, save_path+'model_{}.pkl'.format(idx))
    curr_oof["predict"] = model.predict(v_x, num_iteration=model.best_iteration_)
    oof_df = pd.concat([oof_df, curr_oof])

oof_df.to_csv(save_path + "oof_df.csv")
with open(dir+"log.json", "w") as f:
    json.dump(eval_result, f)


