#!/usr/bin/env python3
"""
测试智能体自主调用工具回答问题的能力
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(__file__))

from data_analyzer_agent import DataAnalyzerAgent


def test_tool_calling():
    """测试智能体自主调用工具回答问题的能力"""
    print("初始化数据分析智能体...")
    try:
        analyzer = DataAnalyzerAgent()
        print("✓ 智能体初始化成功")
    except Exception as e:
        print(f"✗ 智能体初始化失败: {e}")
        return

    # 加载测试数据集
    print("\n1. 加载Iris数据集用于测试...")
    try:
        result = analyzer.load_data("iris", "iris_test")
        if result.get("success"):
            print(f"✓ 数据加载成功，数据形状: {result['data_shape']}")
            print(f"  列名: {', '.join(result['columns'])}")
        else:
            print(f"✗ 数据加载失败: {result.get('error')}")
            return
    except Exception as e:
        print(f"✗ 数据加载异常: {e}")
        return

    # 测试智能体自主调用工具回答问题
    print("\n2. 测试智能体自主调用工具回答问题...")
    
    test_questions = [
        "当前加载的数据集包含哪些列？",
        "请告诉我数据集的形状信息",
        "当前数据集的基本信息是什么？",
        "数据集中各列的数据类型是什么？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n2.{i} 测试问题: {question}")
        print("   智能体回答:")
        try:
            # 与智能体对话，观察是否能自主调用工具
            analyzer.chat(question)
            print("   ✓ 对话完成")
        except Exception as e:
            print(f"   ✗ 对话失败: {e}")

    print("\n3. 验证工具调用结果...")
    try:
        # 直接调用工具获取信息进行对比
        dataset_info = analyzer.get_current_dataset_info()
        if "error" not in dataset_info:
            print("   直接调用工具获取的数据信息:")
            print(f"   - 数据集ID: {dataset_info['dataset_id']}")
            print(f"   - 数据形状: {dataset_info['shape']}")
            print(f"   - 列名: {', '.join(dataset_info['columns'])}")
            print("   - 数据类型:")
            for col, dtype in dataset_info['data_types'].items():
                print(f"     {col}: {dtype}")
            print("   ✓ 工具调用验证完成")
        else:
            print(f"   ✗ 工具调用验证失败: {dataset_info['error']}")
    except Exception as e:
        print(f"   ✗ 工具调用验证异常: {e}")

    print("\n=== 测试总结 ===")
    print("通过观察智能体的回答内容，可以判断它是否能够:")
    print("1. 理解用户关于数据集结构的问题")
    print("2. 自主调用相应工具获取真实数据信息")
    print("3. 基于真实数据生成准确的回答")
    print("\n请检查上面的智能体回答，确认是否包含真实的列名、数据类型等信息。")


if __name__ == "__main__":
    test_tool_calling()