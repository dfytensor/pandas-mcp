#!/usr/bin/env python3
"""
基于FastMCP的数据分析服务端
使用HTTP协议提供数据分析功能
"""

import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
from sklearn.datasets import fetch_california_housing
import uuid
import time
import asyncio
import threading
from datetime import datetime
import traceback
from app.config import available_datasets, dataset_timeout, data_type, sqlite_config
import sqlite3
import os

# 创建FastMCP实例，支持HTTP协议
mcp = FastMCP("data_analysis_server")

# 存储数据集的字典，包含时间戳
# 格式: {dataset_id: {"data": DataFrame, "timestamp": timestamp}}
datasets = {}
current_dataset_id = None
# 清理任务引用
cleanup_task = None

# 分析历史记录
analysis_history = []


class PandasBlocksEngine:
    """Pandas积木执行引擎"""

    def load_data(self, data_source: str, data_type_param: str = "auto", dataset_name: str = None) -> Dict[str, Any]:
        """加载数据积木"""
        try:
            df = None
            
            # 如果没有指定数据类型，使用配置中的默认类型
            if data_type_param == "auto":
                data_type_param = data_type

            # 根据数据类型加载数据
            if data_type_param == "sk-learn":
                # 从sklearn加载数据集
                df = self._load_sklearn_dataset(data_source)
            elif data_type_param == "sqlite":
                # 从SQLite加载数据表
                df = self._load_sqlite_table(data_source)
            elif data_type_param == "csv":
                if data_source.startswith(("http://", "https://")):
                    df = pd.read_csv(data_source)
                else:
                    # 尝试解析CSV字符串
                    df = pd.read_csv(io.StringIO(data_source))
            elif data_type_param == "json":
                df = pd.read_json(io.StringIO(data_source))
            elif data_type_param == "excel":
                df = pd.read_excel(io.BytesIO(data_source.encode() if isinstance(data_source, str) else data_source))
            else:
                # 自动检测类型
                try:
                    df = pd.read_csv(io.StringIO(data_source))
                except:
                    try:
                        df = pd.read_json(io.StringIO(data_source))
                    except:
                        return {"error": f"无法自动检测数据类型: {data_type_param}"}

            # 生成数据集ID
            if not dataset_name:
                dataset_name = f"dataset_{len(datasets) + 1}_{datetime.now().strftime('%H%M%S')}"

            datasets[dataset_name] = {"data": df, "timestamp": time.time()}
            global current_dataset_id
            current_dataset_id = dataset_name

            # 记录操作历史
            analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "operation": "load_data",
                "dataset": dataset_name,
                "shape": df.shape
            })

            return {
                "success": True,
                "dataset_id": dataset_name,
                "message": f"数据加载成功: {dataset_name}",
                "data_shape": df.shape,
                "columns": list(df.columns),
                "sample_data": df.head(3).to_dict('records')
            }

        except Exception as e:
            return {"error": f"数据加载失败: {str(e)}"}

    def _load_sklearn_dataset(self, dataset_name: str) -> pd.DataFrame:
        """从sklearn加载数据集"""
        if dataset_name == "iris":
            data = load_iris()
        elif dataset_name == "wine":
            data = load_wine()
        elif dataset_name == "breast_cancer":
            data = load_breast_cancer()
        elif dataset_name == "diabetes":
            data = load_diabetes()
        elif dataset_name == "california_housing":
            data = fetch_california_housing()
        else:
            raise ValueError(f"不支持的sklearn数据集: {dataset_name}")
        
        # 构建DataFrame
        df = pd.DataFrame(data.data, columns=data.feature_names)
        if hasattr(data, 'target'):
            # 添加目标变量列
            target_name = 'target'
            if hasattr(data, 'target_names') and dataset_name != 'california_housing':
                # 对于分类数据集，添加目标名称映射
                df[target_name] = [data.target_names[i] for i in data.target]
            else:
                # 对于回归数据集或其他情况，直接使用target值
                df[target_name] = data.target
        
        return df

    def _load_sqlite_table(self, table_name: str) -> pd.DataFrame:
        """从SQLite数据库加载表"""
        # 获取数据库路径
        db_path = sqlite_config.get("default_path", "data/default.db")
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"数据库文件不存在: {db_path}")
        
        # 连接数据库并读取表
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        finally:
            conn.close()
        
        return df

    def execute_pipeline(self, blocks_config: List[Dict], dataset_id: str = None) -> Dict[str, Any]:
        """执行积木流水线"""
        try:
            # 获取当前数据集
            if not dataset_id:
                dataset_id = current_dataset_id

            if dataset_id not in datasets:
                return {"error": f"数据集不存在: {dataset_id}"}

            df = datasets[dataset_id]["data"].copy()
            execution_results = []
            execution_log = []

            for i, block in enumerate(blocks_config):
                block_type = block.get("type")
                params = block.get("params", {})
                block_name = block.get("name", f"block_{i + 1}")

                execution_log.append(f"执行积木 {i + 1}: {block_type} - {block_name}")

                # 执行单个积木
                result = self._execute_single_block(df, block_type, params, block_name)

                if "error" in result:
                    return {
                        "error": f"积木执行失败: {result['error']}",
                        "failed_block": block_name,
                        "execution_log": execution_log
                    }

                # 更新数据框
                if "data" in result:
                    df = result["data"]

                # 记录结果
                execution_results.append({
                    "block_name": block_name,
                    "block_type": block_type,
                    "result": result.get("summary", "执行成功"),
                    "data_shape_after": df.shape,
                    "details": result.get("details", {})
                })

            # 更新数据集
            datasets[dataset_id] = {"data": df, "timestamp": time.time()}

            # 记录历史
            analysis_history.append({
                "timestamp": datetime.now().isoformat(),
                "operation": "execute_pipeline",
                "dataset": dataset_id,
                "blocks_executed": len(blocks_config),
                "final_shape": df.shape
            })

            return {
                "success": True,
                "execution_results": execution_results,
                "final_data_shape": df.shape,
                "sample_data": df.head(5).to_dict('records'),
                "columns_info": {
                    "names": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
                },
                "execution_log": execution_log
            }

        except Exception as e:
            return {"error": f"积木流水线执行失败: {str(e)}", "traceback": traceback.format_exc()}

    def _execute_single_block(self, df: pd.DataFrame, block_type: str, params: Dict, block_name: str) -> Dict:
        """执行单个积木"""
        try:
            if block_type == "clean":
                return self._clean_block(df, params, block_name)
            elif block_type == "filter":
                return self._filter_block(df, params, block_name)
            elif block_type == "transform":
                return self._transform_block(df, params, block_name)
            elif block_type == "analyze":
                return self._analyze_block(df, params, block_name)
            elif block_type == "groupby":
                return self._groupby_block(df, params, block_name)
            elif block_type == "join":
                return self._join_block(df, params, block_name)
            elif block_type == "visualize":
                return self._visualize_block(df, params, block_name)
            else:
                return {"error": f"不支持的积木类型: {block_type}"}
        except Exception as e:
            return {"error": f"积木执行错误: {str(e)}", "traceback": traceback.format_exc()}

    def _clean_block(self, df: pd.DataFrame, params: Dict, block_name: str) -> Dict:
        """数据清洗积木"""
        original_shape = df.shape
        operations = params.get("operations", [])

        for op in operations:
            method = op.get("method")
            if method == "fillna":
                columns = op.get("columns", df.columns)
                value = op.get("value", 0)
                df[columns] = df[columns].fillna(value)

            elif method == "drop_duplicates":
                subset = op.get("subset", None)
                df = df.drop_duplicates(subset=subset)

            elif method == "drop_columns":
                columns = op.get("columns", [])
                df = df.drop(columns=columns)

            elif method == "rename_columns":
                rename_map = op.get("rename_map", {})
                df = df.rename(columns=rename_map)

            elif method == "correct_types":
                type_map = op.get("type_map", {})
                for col, dtype in type_map.items():
                    if col in df.columns:
                        try:
                            if dtype == "datetime":
                                df[col] = pd.to_datetime(df[col])
                            else:
                                df[col] = df[col].astype(dtype)
                        except:
                            pass  # 类型转换失败时保持原样

        return {
            "data": df,
            "summary": f"数据清洗完成: {original_shape} -> {df.shape}",
            "details": {
                "rows_removed": original_shape[0] - df.shape[0],
                "columns_removed": original_shape[1] - df.shape[1],
                "operations_performed": len(operations)
            }
        }

    def _filter_block(self, df: pd.DataFrame, params: Dict, block_name: str) -> Dict:
        """数据过滤积木"""
        original_shape = df.shape

        # 支持多种过滤条件
        condition = params.get("condition", "")
        query = params.get("query", "")

        if condition:
            # 简单的列条件过滤
            if ">" in condition or "<" in condition or "==" in condition:
                df = df.query(condition)
        elif query:
            # 使用pandas query
            df = df.query(query)
        else:
            # 默认返回原数据
            pass

        return {
            "data": df,
            "summary": f"数据过滤完成: {original_shape} -> {df.shape}",
            "details": {
                "rows_kept": df.shape[0],
                "rows_filtered_out": original_shape[0] - df.shape[0],
                "condition_applied": condition or query
            }
        }

    def _transform_block(self, df: pd.DataFrame, params: Dict, block_name: str) -> Dict:
        """数据转换积木"""
        action = params.get("action", "")

        if action == "select":
            columns = params.get("columns", [])
            df = df[columns]

        elif action == "sort":
            by = params.get("by", [])
            ascending = params.get("ascending", True)
            df = df.sort_values(by=by, ascending=ascending)

        elif action == "create_column":
            column_name = params.get("column_name")
            expression = params.get("expression", "")
            if expression:
                # 简单的表达式计算
                try:
                    df[column_name] = eval(expression, {"df": df, "np": np})
                except:
                    pass

        elif action == "pivot":
            index = params.get("index")
            columns = params.get("columns")
            values = params.get("values")
            if all([index, columns, values]):
                df = df.pivot_table(index=index, columns=columns, values=values, aggfunc='mean')

        return {
            "data": df,
            "summary": f"数据转换完成: {df.shape}",
            "details": {"action_performed": action}
        }

    def _analyze_block(self, df: pd.DataFrame, params: Dict, block_name: str) -> Dict:
        """数据分析积木"""
        analysis_type = params.get("analysis_type", "basic")
        results = {}

        if analysis_type == "basic":
            results["description"] = df.describe().to_dict()
            results["info"] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "null_counts": df.isnull().sum().to_dict()
            }

        elif analysis_type == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                results["correlation_matrix"] = numeric_df.corr().to_dict()

        elif analysis_type == "value_counts":
            column = params.get("column")
            if column and column in df.columns:
                results["value_counts"] = df[column].value_counts().to_dict()

        return {
            "data": df,  # 分析操作通常不改变原数据
            "summary": f"数据分析完成: {analysis_type}",
            "details": results
        }

    def _groupby_block(self, df: pd.DataFrame, params: Dict, block_name: str) -> Dict:
        """分组聚合积木"""
        group_columns = params.get("group_columns", [])
        agg_functions = params.get("agg_functions", {})

        if group_columns and agg_functions:
            # 确保分组列存在
            valid_columns = [col for col in group_columns if col in df.columns]
            if valid_columns:
                grouped = df.groupby(valid_columns)
                result_df = grouped.agg(agg_functions).reset_index()
                return {
                    "data": result_df,
                    "summary": f"分组聚合完成: 按 {valid_columns} 分组",
                    "details": {
                        "group_columns": valid_columns,
                        "aggregations": agg_functions,
                        "result_shape": result_df.shape
                    }
                }

        return {"data": df, "summary": "分组聚合参数不完整，跳过此操作"}

    def _join_block(self, df: pd.DataFrame, params: Dict, block_name: str) -> Dict:
        """表关联积木"""
        try:
            # 获取关联参数
            right_dataset_id = params.get("right_dataset_id")
            join_type = params.get("join_type", "inner")  # inner, left, right, outer
            left_on = params.get("left_on")
            right_on = params.get("right_on")
            left_columns = params.get("left_columns")
            right_columns = params.get("right_columns")
            suffixes = params.get("suffixes", ("_left", "_right"))
            
            # 检查右表是否存在
            if right_dataset_id not in datasets:
                return {"error": f"右表数据集不存在: {right_dataset_id}"}
            
            # 获取右表数据
            right_df = datasets[right_dataset_id]["data"].copy()
            
            # 如果指定了要保留的列，则进行筛选
            if left_columns:
                # 确保指定的列存在于左表中
                valid_left_columns = [col for col in left_columns if col in df.columns]
                df = df[valid_left_columns]
            
            if right_columns:
                # 确保指定的列存在于右表中
                valid_right_columns = [col for col in right_columns if col in right_df.columns]
                right_df = right_df[valid_right_columns]
            
            # 执行关联操作
            if left_on and right_on:
                # 基于指定列进行关联
                result_df = pd.merge(df, right_df, left_on=left_on, right_on=right_on, 
                                   how=join_type, suffixes=suffixes)
            else:
                # 基于索引进行关联
                result_df = pd.merge(df, right_df, left_index=True, right_index=True, 
                                   how=join_type, suffixes=suffixes)
            
            return {
                "data": result_df,
                "summary": f"表关联完成: {join_type} join, {df.shape} + {right_df.shape} -> {result_df.shape}",
                "details": {
                    "join_type": join_type,
                    "left_table_shape": df.shape,
                    "right_table_shape": right_df.shape,
                    "result_shape": result_df.shape,
                    "left_on": left_on,
                    "right_on": right_on
                }
            }
            
        except Exception as e:
            return {"error": f"表关联执行错误: {str(e)}", "traceback": traceback.format_exc()}

    def _visualize_block(self, df: pd.DataFrame, params: Dict, block_name: str) -> Dict:
        """可视化积木"""
        chart_type = params.get("chart_type", "bar")
        title = params.get("title", f"Chart - {block_name}")

        try:
            plt.figure(figsize=params.get("figsize", (10, 6)))

            if chart_type == "bar":
                x_col = params.get("x")
                y_col = params.get("y")
                if x_col and y_col:
                    df.plot.bar(x=x_col, y=y_col, title=title)

            elif chart_type == "line":
                df.plot.line(title=title)

            elif chart_type == "hist":
                column = params.get("column", df.select_dtypes(include=[np.number]).columns[0])
                df[column].hist()
                plt.title(title)

            elif chart_type == "scatter":
                x_col = params.get("x")
                y_col = params.get("y")
                if x_col and y_col:
                    df.plot.scatter(x=x_col, y=y_col, title=title)

            plt.tight_layout()

            # 保存图片到base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.read()).decode()
            plt.close()

            return {
                "data": df,
                "summary": f"可视化完成: {chart_type} 图表",
                "details": {
                    "chart_type": chart_type,
                    "image_base64": img_base64,
                    "image_format": "png"
                }
            }

        except Exception as e:
            return {
                "data": df,
                "summary": f"可视化失败: {str(e)}",
                "details": {"error": str(e)}
            }

    def get_dataset_info(self, dataset_id: str = None) -> Dict[str, Any]:
        """获取数据集信息"""
        if not dataset_id:
            dataset_id = current_dataset_id

        if dataset_id not in datasets:
            return {"error": f"数据集不存在: {dataset_id}"}

        df = datasets[dataset_id]["data"]

        return {
            "dataset_id": dataset_id,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "sample_data": df.head(3).to_dict('records')
        }

    def get_analysis_history(self) -> List[Dict]:
        """获取分析历史"""
        return analysis_history


# 创建全局引擎实例
engine = PandasBlocksEngine()


async def cleanup_expired_datasets_periodically():
    """
    异步定期清理过期的数据集
    """
    while True:
        try:
            current_time = time.time()
            expired_datasets = []
            
            for dataset_id, dataset_info in list(datasets.items()):
                if current_time - dataset_info["timestamp"] > dataset_timeout:
                    expired_datasets.append(dataset_id)
            
            # 删除过期的数据集
            for dataset_id in expired_datasets:
                del datasets[dataset_id]
                # 如果过期的数据集是当前数据集，重置current_dataset_id
                if dataset_id == current_dataset_id:
                    current_dataset_id = None
            
            if len(expired_datasets) > 0:
                print(f"清理了 {len(expired_datasets)} 个过期数据集")
        except Exception as e:
            print(f"清理过期数据集时出错: {e}")
        
        # 等待下一个清理周期
        await asyncio.sleep(min(dataset_timeout, 60))  # 最多每分钟检查一次


def run_cleanup_service():
    """
    在新线程中运行异步清理服务
    """
    async def start_cleanup():
        global cleanup_task
        cleanup_task = asyncio.create_task(cleanup_expired_datasets_periodically())
        await cleanup_task
    
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_cleanup())


def start_cleanup_service():
    """
    启动异步清理服务
    """
    # 在新线程中运行清理服务
    thread = threading.Thread(target=run_cleanup_service, daemon=True)
    thread.start()
    print("数据集清理服务已启动")


def stop_cleanup_service():
    """
    停止异步清理服务
    """
    global cleanup_task
    if cleanup_task and not cleanup_task.done():
        cleanup_task.cancel()
        print("数据集清理服务已停止")


def get_dataset(dataset_id: str):
    """
    获取数据集，如果数据集不存在则返回None
    
    Args:
        dataset_id: 数据集ID
        
    Returns:
        数据集DataFrame或None（如果不存在）
    """
    # 检查数据集是否存在
    if dataset_id in datasets:
        return datasets[dataset_id]["data"]
    return None


# MCP工具定义
@mcp.tool()
def load_dataset(data_content: str, data_type: str = "auto", dataset_name: str = None) -> Dict[str, Any]:
    """
    加载数据集到Pandas积木引擎

    Args:
        data_content: 数据内容（数据集名称、表名、CSV字符串、JSON字符串或URL）
        data_type: 数据类型（sk-learn/sqlite/csv/json/excel/auto）
        dataset_name: 可选的数据集名称
    """
    return engine.load_data(data_content, data_type, dataset_name)


@mcp.tool()
def execute_analysis_pipeline(blocks_config: List[Dict], dataset_id: str = None) -> Dict[str, Any]:
    """
    执行积木流水线分析

    Args:
        blocks_config: 积木配置列表
        dataset_id: 可选的数据集ID（默认为当前数据集）

    Example blocks_config:
        [
            {
                "type": "clean",
                "name": "数据清洗",
                "params": {
                    "operations": [
                        {"method": "fillna", "columns": ["age"], "value": 0},
                        {"method": "drop_duplicates"}
                    ]
                }
            },
            {
                "type": "join",
                "name": "数据关联",
                "params": {
                    "right_dataset_id": "another_dataset_id",
                    "join_type": "inner",
                    "left_on": "id",
                    "right_on": "user_id"
                }
            },
            {
                "type": "analyze",
                "name": "基础分析",
                "params": {
                    "analysis_type": "basic"
                }
            }
        ]
    """
    return engine.execute_pipeline(blocks_config, dataset_id)


@mcp.tool()
def get_dataset_information(dataset_id: str = None) -> Dict[str, Any]:
    """
    获取当前数据集的基本信息
    """
    return engine.get_dataset_info(dataset_id)


@mcp.tool()
def get_basic_statistics(dataset_id: str = None) -> Dict[str, Any]:
    """
    获取数据集的基础统计信息
    
    Args:
        dataset_id: 可选的数据集ID（默认为当前数据集）
    """
    if not dataset_id:
        dataset_id = current_dataset_id
        
    if dataset_id not in datasets:
        return {"error": f"数据集不存在: {dataset_id}"}
        
    df = datasets[dataset_id]["data"]
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {"error": "数据集中没有数值列"}
        
    return {
        "dataset_id": dataset_id,
        "statistics": numeric_df.describe().to_dict()
    }


@mcp.tool()
def get_correlation_matrix(dataset_id: str = None) -> Dict[str, Any]:
    """
    计算数值列之间的相关性矩阵
    
    Args:
        dataset_id: 可选的数据集ID（默认为当前数据集）
    """
    if not dataset_id:
        dataset_id = current_dataset_id
        
    if dataset_id not in datasets:
        return {"error": f"数据集不存在: {dataset_id}"}
        
    df = datasets[dataset_id]["data"]
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {"error": "数据集中没有数值列"}
        
    return {
        "dataset_id": dataset_id,
        "correlation_matrix": numeric_df.corr().to_dict()
    }


@mcp.tool()
def filter_data(column: str, operator: str, value: Any, dataset_id: str = None) -> Dict[str, Any]:
    """
    根据条件过滤数据
    
    Args:
        column: 列名
        operator: 操作符 (==, !=, >, <, >=, <=)
        value: 比较值
        dataset_id: 可选的数据集ID（默认为当前数据集）
    """
    if not dataset_id:
        dataset_id = current_dataset_id
        
    if dataset_id not in datasets:
        return {"error": f"数据集不存在: {dataset_id}"}
        
    df = datasets[dataset_id]["data"]
    
    if column not in df.columns:
        return {"error": f"列 {column} 不存在"}
        
    # 构造过滤条件
    try:
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
            
        # 更新数据集
        datasets[dataset_id] = {"data": filtered_df, "timestamp": time.time()}
        
        return {
            "dataset_id": dataset_id,
            "message": f"成功过滤数据，剩余 {len(filtered_df)} 行",
            "filtered_shape": filtered_df.shape
        }
    except Exception as e:
        return {"error": f"过滤数据时出错: {str(e)}"}


@mcp.tool()
def group_by_analysis(group_column: str, agg_column: str, agg_function: str, dataset_id: str = None) -> Dict[str, Any]:
    """
    按指定列分组并进行聚合分析
    
    Args:
        group_column: 分组列
        agg_column: 聚合列
        agg_function: 聚合函数 (mean, sum, count, min, max)
        dataset_id: 可选的数据集ID（默认为当前数据集）
    """
    if not dataset_id:
        dataset_id = current_dataset_id
        
    if dataset_id not in datasets:
        return {"error": f"数据集不存在: {dataset_id}"}
        
    df = datasets[dataset_id]["data"]
    
    if group_column not in df.columns:
        return {"error": f"分组列 {group_column} 不存在"}
        
    if agg_column not in df.columns:
        return {"error": f"聚合列 {agg_column} 不存在"}
        
    try:
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
            "dataset_id": dataset_id,
            "group_column": group_column,
            "agg_column": agg_column,
            "agg_function": agg_function,
            "result": result.to_dict()
        }
    except Exception as e:
        return {"error": f"分组聚合分析时出错: {str(e)}"}


@mcp.tool()
def get_column_unique_values(column: str, dataset_id: str = None) -> Dict[str, Any]:
    """
    获取列的唯一值
    
    Args:
        column: 列名
        dataset_id: 可选的数据集ID（默认为当前数据集）
    """
    if not dataset_id:
        dataset_id = current_dataset_id
        
    if dataset_id not in datasets:
        return {"error": f"数据集不存在: {dataset_id}"}
        
    df = datasets[dataset_id]["data"]
    
    if column not in df.columns:
        return {"error": f"列 {column} 不存在"}
        
    unique_values = df[column].unique().tolist()
    
    return {
        "dataset_id": dataset_id,
        "column": column,
        "unique_values_count": len(unique_values),
        "unique_values": unique_values
    }


@mcp.tool()
def get_data_summary(dataset_id: str = None) -> Dict[str, Any]:
    """
    获取数据综合摘要
    
    Args:
        dataset_id: 可选的数据集ID（默认为当前数据集）
    """
    if not dataset_id:
        dataset_id = current_dataset_id
        
    if dataset_id not in datasets:
        return {"error": f"数据集不存在: {dataset_id}"}
        
    df = datasets[dataset_id]["data"]
    
    return {
        "dataset_id": dataset_id,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=[np.number]).empty else {},
        "sample_data": df.head().to_dict('records')
    }


@mcp.tool()
def get_analysis_history() -> List[Dict]:
    """
    获取分析操作历史记录
    """
    return engine.get_analysis_history()


if __name__ == "__main__":
    print("数据分析MCP服务端启动中...")
    print("可用工具:")
    print("- load_dataset: 加载数据集")
    print("- execute_analysis_pipeline: 执行分析流水线")
    print("- get_dataset_information: 获取数据集信息")
    print("- get_basic_statistics: 获取基础统计信息")
    print("- get_correlation_matrix: 获取相关性矩阵")
    print("- filter_data: 根据条件过滤数据")
    print("- group_by_analysis: 分组聚合分析")
    print("- get_column_unique_values: 获取列的唯一值")
    print("- get_data_summary: 获取数据综合摘要")
    print("- get_analysis_history: 获取分析历史")
    print("\n支持的积木类型:")
    print("- clean: 数据清洗")
    print("- filter: 数据过滤")
    print("- transform: 数据转换")
    print("- analyze: 数据分析")
    print("- groupby: 分组聚合")
    print("- join: 表关联")
    print("- visualize: 数据可视化")
    
    # 启动清理服务
    start_cleanup_service()
    
    # 运行MCP服务器，使用HTTP协议
    mcp.run(transport="sse")