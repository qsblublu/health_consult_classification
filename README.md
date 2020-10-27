### 目录结构

```yaml
data:  # 数据文件
    - all_data.csv  # 初始数据
    - stopwords.txt  # 中文分词停用词
    - expr1:
        - skill_cluster_data.csv  # 聚类数据清理之后文件
        - skill_cluster_result.csv  # 聚类结果
    - expr2:
        - health_consult_data.csv  # 健康咨询分类数据清理之后文件
        - health_consult_data_train.csv  # 训练数据
        - health_consult_data_test.csv  # 测试数据
        - health_consult_dict.json  # 字典
model:  # 模型
    - disease_analysis_word2vec.model  # 疾病分析word2vec模型
    - health_consult_classification.pth  # 健康咨询分类模型
src:  # 源码
```


### Get Started

```bash
# 安装依赖
pip install -r requirements.txt
```

#### Expr1

```bash
# 进入目录
cd your_path/src/expr1_skill_cluster
# 数据清理
python data_parse.py
# 聚类
python skill_cluster.py
```

### Expr2

```bash
# 进入目录
cd your_path/src/expr2_health_consult_classification
# 数据清理
python data_parse.py
# 训练测试模型
python train.py
```


### Expr2配置
可以尝试调节参数来提高准确率

```yaml
dataset:
  train_data_file: ../../data/expr2/health_consult_data_train.csv
  test_data_file: ../../data/expr2/health_consult_data_test.csv
  dict_file: ../../data/expr2/health_consult_dict.json
model:
  embed_num: 41000  # 不建议修改
  embed_dim: 10  # embedding dim
  kernel_sizes: [3, 4, 5]  # conv layer 数量以及每个conv 的kernel size
  class_num: 5  # 分5类不能修改
  dropout_p: 0.3  # dropout layer 概率
train:
  epoch: 10
  print_freq: 1000
  batch_size: 10 
  lr: 0.002
```