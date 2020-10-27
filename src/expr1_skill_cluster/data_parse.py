import pandas as pd
import re
import numpy as np


all_data = pd.read_csv('../../data/all_data.csv')

col_skill = all_data['擅长']
all_data_len = len(col_skill)
step = 10000

for idx, skill in col_skill.items():
    try:
        if pd.isnull(skill):
            continue
        if not skill.startswith('擅长'):
            col_skill[idx] = np.nan
            continue

        skill = re.sub(r'擅长|:|：|，|。|,|\.|（.*?）|、|（|!|！', ' ', skill)
        skill = re.sub(r' +', ' ', skill).strip()
        col_skill[idx] = str(skill)

        if idx % step == 0:
            print(f'already finish {idx / all_data_len * 100}%')
    except:
        print(f'col:擅长, row: {idx} content: {skill} error')

all_data = all_data[['回复姓名', '擅长']]
all_data.to_csv('../../data/skill_cluster.csv', index=False)
