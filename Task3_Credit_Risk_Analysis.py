#!/usr/bin/env python
# coding: utf-8

# In[76]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


# In[77]:


# 1. 加载数据
data = pd.read_csv('/Users/huangjiawei/Desktop/JPMORGAN CHASE & CO./3/Task_3_and_4_Loan_Data.csv')


# In[78]:


# 2. 数据预处理
# 2.1 检查缺失值并填充（大多数模型（如随机森林）要求输入数据无缺失值，如果有缺失值直接传给模型，可能导致报错或性能下降。）
# 将数据中的缺失值替换为各列的中位数（中位数对异常值不敏感，适合处理数值型数据的缺失值）
data = data.fillna(data.median())


# In[79]:


# 2.2 定义特征和目标变量
features = [
    'credit_lines_outstanding',
    'loan_amt_outstanding',
    'total_debt_outstanding',
    'income',
    'years_employed',
    'fico_score'
]
target = 'default'

X = data[features]
y = data[target]


# In[80]:


# 2.3 数据拆分
# train_test_split：将数据集拆分为训练集和测试集。test_size：测试集占比，默认 0.25。random_state：随机种子，保证结果可复现。stratify：按目标变量的分布比例分层拆分。
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# In[81]:


# 2.4 标准化数值特征（y 是目标变量，表示类别（0/1）或概率值，标准化没有意义）
scaler = StandardScaler()
# fit_transform()：对训练集计算均值和标准差，并进行标准化
X_train = scaler.fit_transform(X_train)
# transform()：使用训练集的均值和标准差对测试集进行标准化（不能重新计算）
# 原因：避免数据泄漏，即测试集的信息提前进入训练过程
X_test = scaler.transform(X_test)


# In[82]:


# 3. 模型训练
# 随机森林：能处理非线性关系；抗噪声，避免过拟合；对特征重要性敏感，适合解释信用风险问题。
# RandomForestClassifier：n_estimators：决策树数量。random_state：随机种子。class_weight="balanced"：处理类别不平衡
model = RandomForestClassifier(random_state=42, class_weight="balanced", n_estimators=500)
model.fit(X_train, y_train)


# In[83]:


# 4. 模型评估
# 4.1 预测违约概率
# 预测概率值，输出二维数组，每行表示一个样本属于各类别（0或1）的概率。
y_proba = model.predict_proba(X_test)[:, 1]
# 预测最终分类结果（0或1）
y_pred = model.predict(X_test)


# In[84]:


# 4.2 打印模型评估指标
print("\n模型评估：")
# AUC-ROC：衡量模型对正负样本的区分能力，越接近 1 越好2
print("AUC-ROC:", roc_auc_score(y_test, y_proba))
# classification_report分类报告：分析模型的分类性能。输出分类指标报告，包括 Precision、Recall、F1-Score
print(classification_report(y_test, y_pred))


# In[85]:


# 5. 违约概率输出示例（将预测概率和测试集数据合并，便于分析）
# 将 X_test 数据（numpy 数组或 pandas DataFrame）重新创建为新的 DataFrame，并指定列名为 features
results = pd.DataFrame(X_test, columns=features)
results['predicted_pd'] = y_proba
# reset_index(drop=True)：重置索引，删除原索引，确保和 results 的行对齐
results['actual_default'] = y_test.reset_index(drop=True)

print("\n违约概率预测示例：")
print(results.head())

