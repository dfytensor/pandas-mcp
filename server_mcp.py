#!/usr/bin/env python3
"""
基于FastMCP的数据分析服务端
使用HTTP协议提供数据分析功能
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.datasets import fetch_california_housing

# 创建FastMCP实例，支持HTTP协议
mcp = FastMCP("data_analysis_server")


# 存储数据集的字典
datasets = {}
current_dataset_id = None


@mcp.tool()
def load_dataset(dataset_name: str) -> Dict[str, Any]:
    """
    加载sklearn标准数据集
    
    Args:
        dataset_name: 数据集名称 (iris, wine, breast_cancer, diabetes, california_housing)
        
    Returns:
        加载结果字典
    """
    global current_dataset_id
    
    try:
        # 根据数据集名称加载对应的数据
        if dataset_name == 'iris':
            data = load_iris()
        elif dataset_name == 'wine':
            data = load_wine()
        elif dataset_name == 'breast_cancer':
            data = load_breast_cancer()
        elif dataset_name == 'diabetes':
            data = load_diabetes()
        elif dataset_name == 'california_housing':
            data = fetch_california_housing()
        else:
            return {"error": f"不支持的数据集: {dataset_name}"}
        
        # 构建DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        if hasattr(data, 'target'):
            # 添加目标变量列
            df['target'] = data.target
        
        # 存储数据集
        datasets[dataset_name] = df
        current_dataset_id = dataset_name
        
        return {
            "success": True,
            "dataset_id": dataset_name,
            "data_shape": df.shape,
            "columns": list(df.columns),
            "message": f"成功加载数据集: {dataset_name}"
        }
    except Exception as e:
        return {"error": f"加载数据集失败: {str(e)}"}


@mcp.tool()
def get_dataset_info() -> Dict[str, Any]:
    """
    获取当前数据集信息
    
    Returns:
        数据集信息字典
    """
    if not current_dataset_id:
        return {"error": "当前没有加载任何数据集"}
    
    if current_dataset_id not in datasets:
        return {"error": f"数据集 {current_dataset_id} 不存在"}
    
    df = datasets[current_dataset_id]
    return {
        "dataset_id": current_dataset_id,
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }


@mcp.tool()
def get_basic_statistics() -> Dict[str, Any]:
    """
    获取当前数据集的基础统计信息
    
    Returns:
        基础统计信息字典
    """
    if not current_dataset_id:
        return {"error": "当前没有加载任何数据集"}
    
    if current_dataset_id not in datasets:
        return {"error": f"数据集 {current_dataset_id} 不存在"}
    
    df = datasets[current_dataset_id]
    return {
        "dataset_id": current_dataset_id,
        "description": df.describe().to_dict(),
        "info": {
            "shape": df.shape,
            "columns": list(df.columns),
            "missing_values": df.isnull().sum().to_dict()
        }
    }


@mcp.tool()
def get_correlation_matrix() -> Dict[str, Any]:
    """
    计算当前数据集的相关性矩阵
    
    Returns:
        相关性矩阵字典
    """
    if not current_dataset_id:
        return {"error": "当前没有加载任何数据集"}
    
    if current_dataset_id not in datasets:
        return {"error": f"数据集 {current_dataset_id} 不存在"}
    
    df = datasets[current_dataset_id]
    # 只计算数值列的相关性
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {"error": "数据集中没有数值列"}
    
    correlation_matrix = numeric_df.corr()
    return {
        "dataset_id": current_dataset_id,
        "correlation_matrix": correlation_matrix.to_dict(),
        "columns": list(correlation_matrix.columns)
    }


@mcp.tool()
def filter_data(column: str, operator: str, value: float) -> Dict[str, Any]:
    """
    根据条件过滤数据
    
    Args:
        column: 列名
        operator: 操作符 (==, !=, >, <, >=, <=)
        value: 比较值
        
    Returns:
        过滤结果字典
    """
    if not current_dataset_id:
        return {"error": "当前没有加载任何数据集"}
    
    if current_dataset_id not in datasets:
        return {"error": f"数据集 {current_dataset_id} 不存在"}
    
    df = datasets[current_dataset_id]
    
    if column not in df.columns:
        return {"error": f"列 {column} 不存在"}
    
    try:
        # 根据操作符过滤数据
        if operator == "==":
            filtered_df = df[df[column] == value]
        elif operator == "!=":
            filtered_df = df[df[column] != value]
        elif operator == ">":
            filtered_df = df[df[column] > value]
        elif operator == "<":
            filtered_df = df[df[column] < value]
        elif operator == ">=":
            filtered_df = df[df[column] >= value]
        elif operator == "<=":
            filtered_df = df[df[column] <= value]
        else:
            return {"error": f"不支持的操作符: {operator}"}
        
        return {
            "dataset_id": current_dataset_id,
            "original_shape": df.shape,
            "filtered_shape": filtered_df.shape,
            "filtered_count": len(filtered_df),
            "sample_data": filtered_df.head().to_dict('records') if not filtered_df.empty else []
        }
    except Exception as e:
        return {"error": f"过滤数据时出错: {str(e)}"}


@mcp.tool()
def group_by_analysis(group_column: str, agg_column: str, agg_function: str) -> Dict[str, Any]:
    """
    按指定列分组并进行聚合分析
    
    Args:
        group_column: 分组列名
        agg_column: 聚合列名
        agg_function: 聚合函数 (mean, sum, count, min, max)
        
    Returns:
        分组聚合结果字典
    """
    if not current_dataset_id:
        return {"error": "当前没有加载任何数据集"}
    
    if current_dataset_id not in datasets:
        return {"error": f"数据集 {current_dataset_id} 不存在"}
    
    df = datasets[current_dataset_id]
    
    if group_column not in df.columns:
        return {"error": f"分组列 {group_column} 不存在"}
    
    if agg_column not in df.columns:
        return {"error": f"聚合列 {agg_column} 不存在"}
    
    try:
        # 执行分组聚合
        if agg_function == "mean":
            result = df.groupby(group_column)[agg_column].mean()
        elif agg_function == "sum":
            result = df.groupby(group_column)[agg_column].sum()
        elif agg_function == "count":
            result = df.groupby(group_column)[agg_column].count()
        elif agg_function == "min":
            result = df.groupby(group_column)[agg_column].min()
        elif agg_function == "max":
            result = df.groupby(group_column)[agg_column].max()
        else:
            return {"error": f"不支持的聚合函数: {agg_function}"}
        
        return {
            "dataset_id": current_dataset_id,
            "group_column": group_column,
            "agg_column": agg_column,
            "agg_function": agg_function,
            "result": result.to_dict()
        }
    except Exception as e:
        return {"error": f"分组聚合分析时出错: {str(e)}"}


@mcp.tool()
def get_column_unique_values(column: str) -> Dict[str, Any]:
    """
    获取指定列的唯一值及其计数
    
    Args:
        column: 列名
        
    Returns:
        唯一值及计数字典
    """
    if not current_dataset_id:
        return {"error": "当前没有加载任何数据集"}
    
    if current_dataset_id not in datasets:
        return {"error": f"数据集 {current_dataset_id} 不存在"}
    
    df = datasets[current_dataset_id]
    
    if column not in df.columns:
        return {"error": f"列 {column} 不存在"}
    
    try:
        value_counts = df[column].value_counts()
        return {
            "dataset_id": current_dataset_id,
            "column": column,
            "unique_values": value_counts.to_dict(),
            "unique_count": len(value_counts)
        }
    except Exception as e:
        return {"error": f"获取唯一值时出错: {str(e)}"}


@mcp.tool()
def get_data_summary() -> Dict[str, Any]:
    """
    获取当前数据集的综合摘要信息
    
    Returns:
        数据集综合摘要字典
    """
    if not current_dataset_id:
        return {"error": "当前没有加载任何数据集"}
    
    if current_dataset_id not in datasets:
        return {"error": f"数据集 {current_dataset_id} 不存在"}
    
    df = datasets[current_dataset_id]
    
    try:
        # 基本信息
        summary = {
            "dataset_id": current_dataset_id,
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_rows": df.duplicated().sum(),
        }
        
        # 数值列的统计信息
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) > 0:
            summary["numeric_summary"] = df[numeric_columns].describe().to_dict()
        
        # 分类列的唯一值信息
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_columns) > 0:
            categorical_summary = {}
            for col in categorical_columns:
                categorical_summary[col] = {
                    "unique_count": df[col].nunique(),
                    "top_values": df[col].value_counts().head(3).to_dict()
                }
            summary["categorical_summary"] = categorical_summary
        
        return summary
    except Exception as e:
        return {"error": f"生成数据摘要时出错: {str(e)}"}


if __name__ == "__main__":
    print("数据分析MCP服务端启动中...")
    print("可用工具:")
    print("- load_dataset: 加载数据集")
    print("- get_dataset_info: 获取数据集信息")
    print("- get_basic_statistics: 获取基础统计信息")
    print("- get_correlation_matrix: 获取相关性矩阵")
    print("- filter_data: 过滤数据")
    print("- group_by_analysis: 分组聚合分析")
    print("- get_column_unique_values: 获取列的唯一值")
    print("- get_data_summary: 获取数据综合摘要")
    
    # 运行MCP服务器，使用HTTP协议
    mcp.run(transport="http")