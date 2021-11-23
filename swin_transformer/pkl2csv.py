import numpy as np
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='convert pkl to csv')
    parser.add_argument('pklname', help='your pkl file')
    args = parser.parse_args()
    return args

args = parse_args()
PKL = args.pklname

pred = np.load(PKL, allow_pickle=True)
with open('./data/sample_labels.txt') as f:
    filenames = [x.strip().split(' ')[0].split('.')[0] for x in f.readlines()]
pred_classes = pred['pred_class']

dict1 = {"FileID": filenames, "Type": pred_classes}

# 轉為dataframe再透過pandas轉成csv檔
result_df = pd.DataFrame(dict1)
result_df.to_csv(PKL.split('.')[0]+'csv', index=False)
#print(result_df)