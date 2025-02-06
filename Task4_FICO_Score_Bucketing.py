#!/usr/bin/env python
# coding: utf-8

# In[136]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.tree import DecisionTreeClassifier


# In[137]:


# 1. 读取数据
data = pd.read_csv('/Users/huangjiawei/Desktop/JPMORGAN CHASE & CO./3/Task_3_and_4_Loan_Data.csv')


# In[138]:


# 2. 数据预处理
# 确保没有缺失值；如有缺失值，用中位数填充
data['fico_score'] = data['fico_score'].fillna(data['fico_score'].median())
data['default'] = data['default'].fillna(0)  # 假设默认填充为非违约


# In[139]:


# 3. 决策树分箱：自动寻找分箱边界，使每个区间内的违约率更均匀
# 步骤：1使用决策树模型拟合连续特征（如 fico_score）和目标变量（default） 2获取决策树的分裂节点，作为分箱的边界 3根据这些边界将连续特征分箱，生成离散的信用评分等级
# 初始化决策树模型
tree = DecisionTreeClassifier(max_depth=3)  # 限制树深度，避免过拟合


# In[140]:


# 训练决策树模型
# 机器学习模型通常以矩阵形式处理数据，因此，必须确保数据是二维的，即形状为 (样本数, 特征数)
# data['fico_score'].values.reshape(-1, 1)：一维Pandas Series → 一维NumPy数组 → 二维Numpy数组（-1 表示自动计算维度，确保样本数不变）
tree.fit(data['fico_score'].values.reshape(-1, 1), data['default'])


# In[141]:


# 获取分裂点（决策树分裂节点的阈值），并按升序排列
# tree.tree_.threshold：返回决策树中每个节点的分裂点值
# tree.tree_.feature > -2：筛选出有效分裂点（因为无效节点的 feature 索引为 -2）
thresholds = np.sort(tree.tree_.threshold[tree.tree_.feature > -2])  # 提取有效分裂点


# In[142]:


# 将分裂点作为分箱边界，添加起始和结束值
# np.concatenate(...)：将最小值、分裂点、最大值合并成一个完整的边界数组
bins = np.concatenate(([data['fico_score'].min()], thresholds, [data['fico_score'].max()]))
print("分箱边界：", bins)


# In[143]:


# 使用这些边界分箱
labels = list(range(1, len(bins)))  # 创建分箱标签
# 根据 bins 的边界将 fico_score 分箱，生成fico_rating
# include_lowest=True：确保包括第一个区间的最小值
data['fico_rating'] = pd.cut(data['fico_score'], bins=bins, labels=labels, include_lowest=True)


# In[144]:


# 输出分箱后的结果
print(data[['fico_score', 'fico_rating']])


# In[145]:


# 4. 特征和目标变量
# ⚠️data['fico_rating'] 返回一维数组(n_samples,），不符合模型输入要求，解决办法是将 data['fico_rating'] 转换为二维结构，确保形状为 (n_samples, n_features)
# 法1:X = data['fico_rating'].to_frame()
# 法2:X = data['fico_rating'].values.reshape(-1, 1)
# 法3:features = ['fico_rating'] X = data[features]
X = data[['fico_rating']]
y = data['default']


# In[146]:


# 5. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[147]:


# 6. 模型构建（随机森林）
model = RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=100)
model.fit(X_train, y_train)


# In[148]:


# 7. 违约概率预测
y_proba = model.predict_proba(X_test)[:, 1]  # 提取正类（违约）概率
y_pred = model.predict(X_test)


# In[149]:


# 8. 模型评估
roc_auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)

# 打印结果
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("Classification Report:")
print(report)


# In[150]:


# 9. 查看分箱后的数据和实际违约
results = pd.DataFrame({
    'fico_rating': X_test['fico_rating'].reset_index(drop=True),
    'predicted_pd': y_proba,
    'actual_default': y_test.reset_index(drop=True)
})
results.head()


# In[151]:


# 10. 特征重要性
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(importance)

