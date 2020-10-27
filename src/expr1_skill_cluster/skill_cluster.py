from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# get skill column and transform to narray
skill_cluster_data = pd.read_csv('../../data/expr1/skill_cluster_data.csv')
skill_col = skill_cluster_data['擅长']

skill_arr_index = []
skill_arr = []
for idx, skill in skill_col.items():
    if not pd.isnull(skill):
        skill_arr.append(skill)
        skill_arr_index.append(idx)


# build tfidf matrix
tfidf = TfidfVectorizer(decode_error='ignore')
tfidf_matrix = tfidf.fit_transform(skill_arr)


# cluster skill
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(tfidf_matrix)

# save cluster result to file
kmeans_labels = kmeans.labels_
skill_cluster_result = skill_cluster_data.copy()
skill_cluster_result['cluster'] = -1

for idx, label in zip(skill_arr_index, kmeans_labels):
    skill_cluster_result.loc[[idx], ['cluster']] = label

skill_cluster_result.to_csv('../../data/skill_cluster_result.csv', index=False)
nn