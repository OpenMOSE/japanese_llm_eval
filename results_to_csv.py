#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from glob import glob
import os

model_result_paths = glob("./data/judgements/*/*/*.json")

eval_dataset_dict = {
    "elyza__ELYZA-tasks-100": "ELYZA-tasks-100",
    "yuzuai__rakuda-questions": "Rakuda",
    "lightblue__tengu_bench": "Tengu-Bench",
    "shisa-ai__ja-mt-bench-1shot": "MT-Bench",
}

all_result_dfs = []

for model_result_path in model_result_paths:
    temp_df = pd.read_json(model_result_path, lines=True)
    temp_df["judge_model"] = model_result_path.split("/")[3]
    temp_df["eval_dataset"] = eval_dataset_dict[model_result_path.split("/")[4]]
    temp_df["model_name"] = model_result_path.split("/")[5].replace(".json", "")
    
    all_result_dfs.append(temp_df)

all_result_df = pd.concat(all_result_dfs)
all_result_df["dataset_category"] = all_result_df["eval_dataset"] + " " + all_result_df["Category"]

model_names = all_result_df.model_name.unique()

if not os.path.exists("results"):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs("results")

# 各model_nameごとにデータを分割し、エクセルファイルに書き出す
for model_name in model_names:
    model_df = all_result_df[all_result_df.model_name == model_name]
    with open(f"results/{model_name}_output.csv", mode="w", encoding="cp932", errors="ignore", newline="") as f:
    # pandasでファイルオブジェクトに書き込む
        model_df.to_csv(f, index=False)