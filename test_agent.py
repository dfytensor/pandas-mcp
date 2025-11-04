#!/usr/bin/env python3
"""
测试数据分析智能体
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(__file__))

from data_analyzer_agent import DataAnalyzerAgent


def test_agent():
    """测试数据分析智能体"""
    print("初始化数据分析智能体...")
    try:
        analyzer = DataAnalyzerAgent()
        print("✓ 智能体初始化成功")
    except Exception as e:
        print(f"✗ 智能体初始化失败: {e}")
        return
    
    # 创建示例数据
    sample_data = """name,age,income,department,years_experience
Alice,30,50000,Engineering,5
Bob,25,45000,Marketing,2
Charlie,35,60000,Engineering,8
Diana,28,52000,Marketing,3
Eve,32,55000,HR,6
Frank,29,48000,Engineering,4
Grace,27,47000,Marketing,2
Henry,38,70000,Engineering,12"""

    print("\n1. 测试数据加载...")
    try:
        result = analyzer.load_data(sample_data, "employee_data")
        print(f"✓ 加载结果: {result}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
    
    print("\n2. 测试基础数据分析...")
    try:
        analysis_result = analyzer.execute_analysis("请进行基础数据分析")
        print(f"✓ 分析结果: {analysis_result}")
    except Exception as e:
        print(f"✗ 基础数据分析失败: {e}")
    
    print("\n3. 测试相关性分析...")
    try:
        corr_result = analyzer.execute_analysis("分析各数值列之间的相关性")
        print(f"✓ 相关性分析结果: {corr_result}")
    except Exception as e:
        print(f"✗ 相关性分析失败: {e}")
    
    print("\n4. 查看当前上下文...")
    try:
        context = analyzer.get_context()
        print(f"✓ 当前上下文: {context}")
    except Exception as e:
        print(f"✗ 获取上下文失败: {e}")
    
    print("\n5. 测试与智能体对话...")
    print("智能体响应:")
    # 由于这是流式输出，我们只测试调用
    try:
        analyzer.chat("根据当前数据，哪个部门的平均收入最高？")
        print("✓ 对话测试完成")
    except Exception as e:
        print(f"✗ 对话测试失败: {e}")


if __name__ == "__main__":
    test_agent()