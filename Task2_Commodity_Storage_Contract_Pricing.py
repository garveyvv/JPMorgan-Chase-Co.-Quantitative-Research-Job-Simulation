#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


# 调用task1中的价格函数
from Task1 import *


# In[3]:


def price_contract(
    injection_dates,    # 注入日期列表
    withdrawal_dates,   # 撤回日期列表
    injection_rate,     # 每次注入的速率（MMBtu/天）
    withdrawal_rate,    # 每次撤回的速率（MMBtu/天）
    max_storage,        # 最大存储容量（MMBtu）
    storage_cost        # 每天的存储成本（美元/天）
):
    # 初始化变量
    storage_volume = 0  # 当前存储的天然气量
    total_cost = 0      # 总成本
    total_revenue = 0   # 总收入
    injection_cost = withdrawal_cost = 0.01    # 每次注入、撤回成本（美元/MMBtu）
    transporting_cost = 50  # 每次运输成本（美元/次）
    total_storage_cost = 0
    
    is_storage_active = False  # 是否计算存储成本的标志
    last_storage_date = None   # 上次库存非零的日期
    
    # 遍历注入日期并计算成本
    for date in injection_dates:
        date = pd.Timestamp(date)
        injection_price = get_price(date)  # 获取当天价格
        injection_volume = injection_rate  # 每次注入的体积

        # 检查是否超过存储容量
        if storage_volume + injection_volume > max_storage:
            raise ValueError("Injection exceeds max storage capacity!")

        # 更新存储量和成本
        storage_volume += injection_volume
        total_cost += injection_volume * injection_cost + transporting_cost  #注入成本 + 运输成本

        # 开始计算存储费用
        if not is_storage_active:
            is_storage_active = True
            last_storage_date = date

    # 遍历撤回日期并计算收入
    for date in withdrawal_dates:
        date = pd.Timestamp(date)
        withdrawal_price = get_price(date)  # 获取当天价格
        withdrawal_volume = withdrawal_rate  # 每次撤回的体积

        # 检查是否超过存储量
        if storage_volume < withdrawal_volume:
            raise ValueError("Withdrawal exceeds available storage!")

        # 更新存储量和成本
        storage_volume -= withdrawal_volume
        total_cost += withdrawal_volume * withdrawal_cost + transporting_cost  # 撤回成本 + 运输成本
        total_revenue += withdrawal_volume * (injection_price - withdrawal_cost) # 总收入

        # 停止存储费用计算
        if storage_volume == 0:
            is_storage_active = False
            # 计算到当前日期的存储费用
            total_storage_cost += (date - last_storage_date).days * storage_cost
            total_cost += total_storage_cost
                
    # 处理剩余的存储费用
    if is_storage_active:
        total_cost += (pd.Timestamp(max(withdrawal_dates + injection_dates)) - last_storage_date).days * storage_cost
     
    # 返回合同价值
        contract_value = total_revenue - total_cost
        return {
            "contract_value": contract_value
        }


# In[9]:


# 测试代码
injection_dates = ['2020-10-31', '2020-11-2']
withdrawal_dates = ['2025-12-12']
injection_rate = 100000  # 每次注入 100,000 MMBtu
withdrawal_rate = 80000  # 每次撤回 80,000 MMBtu
max_storage = 500000     # 最大存储容量 500,000 MMBtu
storage_cost = 0.05      # 存储成本 $0.05/天

# 计算合同价值
result = price_contract(
    injection_dates, 
    withdrawal_dates, 
    injection_rate,
    withdrawal_rate,
    max_storage,
    storage_cost
)

print(result)

