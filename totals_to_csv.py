from glob import glob
import os

model_result_paths = glob("./data/judgements/*/*/*.json")


# In[159]:


eval_dataset_dict = {
    "elyza__ELYZA-tasks-100": "ELYZA-tasks-100",
    # "yuzuai__rakuda-questions": "Rakuda",
    "lightblue__tengu_bench": "Tengu-Bench",
    "shisa-ai__ja-mt-bench-1shot": "MT-Bench",
}


# In[160]:


import pandas as pd

all_result_dfs = []

for model_result_path in model_result_paths:
    if not model_result_path.split("/")[4] in eval_dataset_dict:
        continue
    temp_df = pd.read_json(model_result_path, lines=True)
    temp_df["judge_model"] = model_result_path.split("/")[3]
    temp_df["eval_dataset"] = eval_dataset_dict[model_result_path.split("/")[4]]
    temp_df["model_name"] = model_result_path.split("/")[5].replace(".json", "")
    
    all_result_dfs.append(temp_df)


# In[161]:


import pandas as pd

all_result_df = pd.concat(all_result_dfs)

all_result_df["dataset_category"] = all_result_df["eval_dataset"] + " " + all_result_df["Category"]

# In[162]:


eval_dataset_names = all_result_df.eval_dataset.unique()
model_names = all_result_df.model_name.unique()


# In[163]:


eval_corr_results = {}
for eval_dataset_name in eval_dataset_names:
    eval_corr_results[eval_dataset_name] = {}
    for model_name in model_names:
        eval_corr_results[eval_dataset_name][model_name] = all_result_df[(all_result_df.eval_dataset == eval_dataset_name) & (all_result_df.model_name == model_name)].score.mean()


# In[164]:


pd.DataFrame(eval_corr_results).corr().round(4)


# In[166]:


eval_res_df = pd.DataFrame(eval_corr_results)
eval_res_df['ELYZA-tasks-100'] = eval_res_df['ELYZA-tasks-100'] * 2

# データセットごとの重み
weights = {
    # "Rakuda": 40,
    "Tengu-Bench": 120,
    "MT-Bench": 60,
    "ELYZA-tasks-100": 100
}

eval_res_df['mean'] = eval_res_df.mean(axis=1)

# 重み付けされたmean2を計算
weighted_scores = [eval_res_df[dataset] * weight for dataset, weight in weights.items()]
eval_res_df['weighted_mean'] = sum(weighted_scores) / 280

eval_res_df = eval_res_df.sort_values(by='weighted_mean', ascending=False)

if not os.path.exists("results"):
    # ディレクトリが存在しない場合、ディレクトリを作成する
    os.makedirs("results")

# csvで保存
with open("results/totals.csv", mode="w", encoding="cp932", errors="ignore", newline="") as f:
# pandasでファイルオブジェクトに書き込む
    eval_res_df.to_csv(f, index=True)