#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
# interp1d是SciPy 的插值函数，用于从离散数据生成连续函数
from scipy.interpolate import interp1d
# SARIMAX 是 statsmodels 中用于时间序列分析的高级模型
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np


# In[8]:


# 读取数据
data = pd.read_csv('/Users/huangjiawei/Desktop/JPMORGAN CHASE & CO./1/Nat_Gas.csv')
# 数据预处理：日期为转为datetime标准格式
data['Dates'] = pd.to_datetime(data['Dates'], format = '%m/%d/%y')
# 日期设为索引
data.set_index('Dates', inplace = True)


# In[42]:


# 生成每日数据并插值
def generate_daily_prices(data):
    daily_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    f = interp1d(data.index.values.astype(float), data['Prices'], kind='linear', fill_value="extrapolate")
    daily_prices = pd.DataFrame({'Prices': f(daily_index.values.astype(float))}, index=daily_index)
    return daily_prices

daily_data = generate_daily_prices(data)


# In[43]:


# 时间序列建模：SARIMA（数据有季节性趋势）
def fit_sarima(data):
    model = SARIMAX(data['Prices'], order = (1, 1, 1), seasonal_order = (1, 1, 0, 12), enforce_stationarity = False, enforce_invertibility = False)
    result = model.fit(disp = False)
    return result

# 拟合 SARIMA 模型
sarima_model = fit_sarima(daily_data)


# In[44]:


# 预测未来价格
def forecast_daily_prices(model, start_date, days):
    future_index = pd.date_range(start=start_date, periods=days, freq='D')
    # 从拟合好的时间序列模型中生成未来 days 步（天）的预测，获取预测的均值（点预测值），即预测的每一天的价格
    forecast = model.get_forecast(steps=days).predicted_mean
    return pd.DataFrame({'Prices': forecast}, index=future_index)


# In[45]:


# 预测从 2024 年 10 月 1 日起未来一年的每日价格
future_prices = forecast_daily_prices(sarima_model, start_date="2024-10-01", days=365)


# In[46]:


# 综合函数
def get_price(date):
    # pd.Timestamp 是 Pandas 用于处理时间的对象，转化为标准的 datetime64 类型
    date = pd.Timestamp(date)
    if date in daily_data.index:
        # .loc返回的是指定日期的价格值。
        return daily_data.loc[date, 'Prices']
    elif date > data.index[-1]:
        if date in future_prices.index:
            return future_prices.loc[date, 'Prices']
        else:
            return 'Date beyond forecast range.'
    else:
        return 'Date beyond forecast range.'


# In[47]:


# ISO 标准日期格式（YYYY-MM-DD），可直接被识别
get_price('2025-9-30')

