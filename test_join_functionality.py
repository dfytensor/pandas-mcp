#!/usr/bin/env python3
"""
测试新增的 join 功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server import PandasBlocksEngine
import pandas as pd
import json


def test_join_functionality():
    """测试表关联功能"""
    # 创建引擎实例
    engine = PandasBlocksEngine()
    
    # 准备测试数据
    # 左表 - 用户信息
    users_data = """user_id,name,age
1,Alice,25
2,Bob,30
3,Charlie,35
4,David,28"""
    
    # 右表 - 订单信息
    orders_data = """order_id,user_id,product,amount
101,1,Laptop,1200
102,2,Mouse,25
103,1,Keyboard,75
104,3,Monitor,300
105,5,Ethernet Cable,15"""
    
    # 加载左表数据
    result1 = engine.load_data(users_data, "csv", "users")
    print("加载用户数据:", result1)
    
    # 加载右表数据
    result2 = engine.load_data(orders_data, "csv", "orders")
    print("加载订单数据:", result2)
    
    # 测试 inner join
    join_params = {
        "type": "join",
        "name": "用户订单关联",
        "params": {
            "right_dataset_id": "orders",
            "join_type": "inner",
            "left_on": "user_id",
            "right_on": "user_id"
        }
    }
    
    # 执行关联操作
    result = engine.execute_pipeline([join_params], "users")
    print("\n关联操作结果:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 验证结果
    if "success" in result and result["success"]:
        print("\n✓ Join 功能测试通过")
        print(f"结果数据形状: {result['final_data_shape']}")
        print("列名:", result['columns_info']['names'])
    else:
        print("\n✗ Join 功能测试失败")
        print("错误信息:", result.get("error", "未知错误"))


if __name__ == "__main__":
    test_join_functionality()