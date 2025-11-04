# 数据分析智能体测试说明

## 测试概述

本测试套件用于验证数据分析智能体的各种功能，特别是使用sklearn提供的真实数据集进行能力测试。

## 测试内容

测试包括以下10个核心功能：

1. Iris数据集基础分析
2. Iris数据集相关性分析
3. Wine数据集分组分析
4. Breast Cancer数据集统计分析
5. Diabetes数据集可视化
6. California Housing数据集基础分析
7. California Housing数据集相关性分析
8. Iris数据集分类特征分析
9. Wine数据集特征分布
10. Breast Cancer数据集目标变量分析

## 运行测试

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行完整测试

```bash
python test_sklearn_datasets.py
```

### 运行简化测试（不依赖sklearn）

```bash
python test_sklearn_datasets.py --simple
```

## 测试说明

数据分析智能体现在支持以下功能：

1. 从sklearn加载标准数据集（iris, wine, breast_cancer, diabetes, california_housing）
2. 基础数据统计分析
3. 特征相关性分析
4. 分组聚合分析
5. 数据可视化
6. 智能对话交互

## 预期结果

所有测试用例应能成功执行，验证智能体在处理真实数据集时的分析能力。