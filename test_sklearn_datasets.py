#!/usr/bin/env python3
"""
使用sklearn真实数据集测试数据分析智能体的各种能力
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(__file__))

from data_analyzer_agent import DataAnalyzerAgent


def test_sklearn_datasets():
    """使用sklearn数据集测试数据分析智能体"""
    print("初始化数据分析智能体...")
    try:
        analyzer = DataAnalyzerAgent()
        print("✓ 智能体初始化成功")
    except Exception as e:
        print(f"✗ 智能体初始化失败: {e}")
        return

    # 检查sklearn是否可用
    try:
        from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_diabetes
        from sklearn.datasets import fetch_california_housing
        print("✓ sklearn数据集模块可用")
    except ImportError:
        print("✗ sklearn不可用，请先安装: pip install scikit-learn")
        return

    # 测试用例列表
    test_cases = [
        {
            "name": "Iris数据集基础分析",
            "dataset": "iris",
            "analysis_request": "请进行基础数据分析，展示数据的基本统计信息"
        },
        {
            "name": "Iris数据集相关性分析",
            "dataset": "iris",
            "analysis_request": "分析各特征之间的相关性"
        },
        {
            "name": "Wine数据集分组分析",
            "dataset": "wine",
            "analysis_request": "按类别分组，分析各类别的平均特征值"
        },
        {
            "name": "Breast Cancer数据集统计分析",
            "dataset": "breast_cancer",
            "analysis_request": "提供数据集的完整统计摘要"
        },
        {
            "name": "Diabetes数据集可视化",
            "dataset": "diabetes",
            "analysis_request": "创建特征之间关系的可视化图表"
        },
        {
            "name": "Iris数据集分类特征分析",
            "dataset": "iris",
            "analysis_request": "分析不同类别花的特征差异"
        },
        {
            "name": "Wine数据集特征分布",
            "dataset": "wine",
            "analysis_request": "展示各特征的分布情况"
        },
        {
            "name": "Breast Cancer数据集目标变量分析",
            "dataset": "breast_cancer",
            "analysis_request": "分析目标变量的分布情况"
        }
    ]

    # 执行测试用例
    passed_tests = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. 测试 {test_case['name']}...")
        try:
            # 加载数据集
            print(f"   加载 {test_case['dataset']} 数据集...")
            load_result = analyzer.load_data(test_case['dataset'], test_case['dataset'])
            
            if not load_result.get("success"):
                print(f"   ✗ 数据加载失败: {load_result.get('error', '未知错误')}")
                continue
            
            data_shape = load_result.get('data_shape')
            print(f"   ✓ 数据加载成功，数据形状: {data_shape}")
            
            # 检查数据是否真的加载成功（不是空的）
            if data_shape[0] == 0:
                print(f"   ⚠ 数据形状异常（行数为0），可能存在加载问题")
            
            # 执行分析
            print(f"   执行分析: {test_case['analysis_request']}")
            analysis_result = analyzer.execute_analysis(test_case['analysis_request'])
            
            if "error" in analysis_result:
                print(f"   ✗ 分析执行失败: {analysis_result['error']}")
                continue
            
            print(f"   ✓ 分析执行成功")
            passed_tests += 1
            
        except Exception as e:
            print(f"   ✗ 测试执行失败: {e}")
    
    # 测试与智能体对话功能 - 特别测试获取列信息
    print(f"\n{total_tests + 1}. 测试与智能体对话功能（获取数据集列信息）...")
    try:
        # 确保有一个数据集被加载（重新加载iris数据集）
        print("   重新加载iris数据集用于测试...")
        analyzer.load_data("iris", "iris")
        
        print("   获取当前数据集信息...")
        dataset_info = analyzer.get_current_dataset_info()
        if "error" in dataset_info:
            print(f"   ✗ 获取数据集信息失败: {dataset_info['error']}")
        else:
            print(f"   ✓ 成功获取数据集信息: {dataset_info['shape']}")
            print(f"     列名: {', '.join(dataset_info['columns'])}")
            passed_tests += 1
        
    except Exception as e:
        print(f"   ✗ 获取数据集信息测试失败: {e}")
    
    # 测试工具调用功能
    print(f"\n{total_tests + 2}. 测试智能体工具调用功能...")
    try:
        # 测试获取列信息工具
        columns_info = analyzer.get_dataset_columns()
        if "error" in columns_info:
            print(f"   ✗ 获取列信息失败: {columns_info['error']}")
        else:
            print(f"   ✓ 成功获取列信息，共 {len(columns_info['columns'])} 列")
            passed_tests += 1
            
        # 测试获取形状信息工具
        shape_info = analyzer.get_dataset_shape()
        if "error" in shape_info:
            print(f"   ✗ 获取形状信息失败: {shape_info['error']}")
        else:
            print(f"   ✓ 成功获取形状信息: {shape_info['shape']}")
            passed_tests += 1
            
    except Exception as e:
        print(f"   ✗ 工具调用测试失败: {e}")
    
    # 输出测试结果总结
    print(f"\n测试完成!")
    print(f"通过测试: {passed_tests}/{total_tests + 2}")
    print(f"通过率: {passed_tests / (total_tests + 2) * 100:.1f}%")


# 简化的测试函数，用于快速验证
def simple_test():
    """简化测试，不依赖sklearn"""
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
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        simple_test()
    else:
        test_sklearn_datasets()