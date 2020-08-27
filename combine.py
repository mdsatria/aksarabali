import pandas as pd 
import numpy as np

aug_data = pd.read_csv("augmented_data.csv", header=None)
real_data = pd.read_csv("train/gt_train.txt", header=None, delimiter=";")
real_data.drop(columns=[2], inplace=True)
df_full = pd.concat([real_data, aug_data])
df_full.to_csv("df_full.csv", index=False, header=False)