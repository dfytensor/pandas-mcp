#!/usr/bin/env python3
"""
Pandas积木MCP服务器
让LLM通过简单的积木配置实现复杂的数据分析能力
"""

import pandas as pd
import numpy as np
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from mcp.server.fastmcp import FastMCP
import traceback

# 初始化MCP服务器
mcp = FastMCP("PandasBlocks")


class PandasBlocksEngine:
    """Pandas积木执行引擎"""

    def __init__(self):
        self.datasets = {}  # 存储多个数据集
        self.current_dataset_id = None
        self.analysis_history = []  # 记录分析历史

    def load_data(self, data_source: str, data_type: str = "csv", dataset_name: str = None) -> Dict[str, Any]:
        """加载数据积木"""
        try:
            df = None

            if data_type == "csv":
                if data_source.startswith(("http://", "https://")):
                    df = pd.read_csv(data_source)
                else:
                    # 尝试解析CSV字符串
                    df = pd.read_csv(io.StringIO(data_source))
            elif data_type == "json":
                df = pd.read_json(io.StringIO(data_source))
            elif data_type == "excel":
                df = pd.read_excel(io.BytesIO(data_source.encode() if isinstance(data_source, str) else data_source))
            else:
                # 自动检测类型
                try:
                    df = pd.read_csv(io.StringIO(data_source))
                except:
                    try:
                        df = pd.read_json(io.StringIO(data_source))
                    except:
                        return {"error": f"无法自动检测数据类型: {data_type}"}

            # 生成数据集ID
            if not dataset_name:
                dataset_name = f"dataset_{len(self.datasets) + 1}_{datetime.now().strftime('%H%M%S')}"

            self.datasets[dataset_name] = df
            self.current_dataset_id = dataset_name

            # 记录操作历史
            self.analysis_history.append({
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

    def execute_pipeline(self, blocks_config: List[Dict], dataset_id: str = None) -> Dict[str, Any]:
        """执行积木流水线"""
        try:
            # 获取当前数据集
            if not dataset_id:
                dataset_id = self.current_dataset_id

            if dataset_id not in self.datasets:
                return {"error": f"数据集不存在: {dataset_id}"}

            df = self.datasets[dataset_id].copy()
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
            self.datasets[dataset_id] = df

            # 记录历史
            self.analysis_history.append({
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
            if right_dataset_id not in self.datasets:
                return {"error": f"右表数据集不存在: {right_dataset_id}"}
            
            # 获取右表数据
            right_df = self.datasets[right_dataset_id].copy()
            
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
            dataset_id = self.current_dataset_id

        if dataset_id not in self.datasets:
            return {"error": f"数据集不存在: {dataset_id}"}

        df = self.datasets[dataset_id]

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
        return self.analysis_history


# 创建全局引擎实例
engine = PandasBlocksEngine()


# MCP工具定义
@mcp.tool()
def load_dataset(data_content: str, data_type: str = "csv", dataset_name: str = None) -> Dict[str, Any]:
    """
    加载数据集到Pandas积木引擎

    Args:
        data_content: 数据内容（CSV字符串、JSON字符串或URL）
        data_type: 数据类型（csv/json/excel/auto）
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
def quick_analysis(data_content: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
    """
    快速数据分析（一键式）

    Args:
        data_content: 数据内容
        analysis_type: 分析类型（basic/comprehensive/statistical）
    """
    # 先加载数据
    load_result = engine.load_data(data_content, "csv")
    if "error" in load_result:
        return load_result

    dataset_id = load_result["dataset_id"]

    # 根据分析类型配置积木
    if analysis_type == "basic":
        blocks = [
            {"type": "clean", "name": "基础清洗", "params": {"operations": [
                {"method": "fillna", "value": 0}
            ]}},
            {"type": "analyze", "name": "描述性统计", "params": {"analysis_type": "basic"}}
        ]
    else:  # comprehensive
        blocks = [
            {"type": "clean", "name": "数据清洗", "params": {"operations": [
                {"method": "fillna", "value": 0},
                {"method": "drop_duplicates"}
            ]}},
            {"type": "analyze", "name": "统计分析", "params": {"analysis_type": "basic"}},
            {"type": "analyze", "name": "相关性分析", "params": {"analysis_type": "correlation"}}
        ]

    return engine.execute_pipeline(blocks, dataset_id)


@mcp.tool()
def get_analysis_history() -> List[Dict]:
    """
    获取分析操作历史记录
    """
    return engine.get_analysis_history()


# 运行服务器
if __name__ == "__main__":
    print("Pandas积木MCP服务器启动中...")
    print("可用工具:")
    print("- load_dataset: 加载数据")
    print("- execute_analysis_pipeline: 执行分析流水线")
    print("- get_dataset_information: 获取数据集信息")
    print("- quick_analysis: 快速分析")
    print("- get_analysis_history: 获取分析历史")
    print("\n支持的积木类型:")
    print("- clean: 数据清洗")
    print("- filter: 数据过滤")
    print("- transform: 数据转换")
    print("- analyze: 数据分析")
    print("- groupby: 分组聚合")
    print("- join: 表关联 (新增)")
    print("- visualize: 数据可视化")

    mcp.run(transport="stdio")