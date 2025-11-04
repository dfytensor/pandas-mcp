#!/usr/bin/env python3
"""
安装依赖并测试数据分析智能体
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 安装 {package} 失败: {e}")
        return False

def check_sklearn():
    """检查sklearn是否可用"""
    try:
        import sklearn
        print(f"✓ sklearn 已安装，版本: {sklearn.__version__}")
        return True
    except ImportError:
        print("✗ sklearn 未安装")
        return False

def run_test():
    """运行测试"""
    try:
        # 添加当前目录到Python路径
        sys.path.append(os.path.dirname(__file__))
        
        # 尝试导入测试模块
        import test_sklearn_datasets
        
        print("开始测试...")
        test_sklearn_datasets.test_sklearn_datasets()
        return True
    except Exception as e:
        print(f"测试执行失败: {e}")
        return False

def main():
    print("=== 数据分析智能体测试环境设置 ===")
    
    # 检查sklearn
    if not check_sklearn():
        print("正在安装 scikit-learn...")
        if install_package("scikit-learn"):
            if not check_sklearn():
                print("安装完成但仍然无法导入sklearn")
                return False
        else:
            print("无法安装scikit-learn，测试无法继续")
            return False
    
    print("\n=== 运行测试 ===")
    return run_test()

if __name__ == "__main__":
    main()