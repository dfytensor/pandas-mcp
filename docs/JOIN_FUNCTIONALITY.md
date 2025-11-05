# 表关联（Join）功能说明

## 功能概述

我们已经为 Pandas 积木 MCP 服务器增加了表与表之间的关联功能。现在可以使用 `join` 类型的积木来合并两个不同的数据集。

## 使用方法

在 [execute_analysis_pipeline](file://E:\pandas-mcp\server.py#L113-L183) 工具中，可以通过添加 `join` 类型的积木来执行表关联操作。

### 积木配置参数

```json
{
  "type": "join",
  "name": "表关联操作名称",
  "params": {
    "right_dataset_id": "另一个数据集的ID",
    "join_type": "inner",
    "left_on": "左侧数据集用于关联的列名",
    "right_on": "右侧数据集用于关联的列名",
    "left_columns": ["可选，指定左侧数据集中需要保留的列"],
    "right_columns": ["可选，指定右侧数据集中需要保留的列"],
    "suffixes": ["_left", "_right"]
  }
}
```

### 参数说明

| 参数 | 类型 | 必填 | 描述 |
|------|------|------|------|
| `right_dataset_id` | string | 是 | 要关联的右侧数据集ID |
| `join_type` | string | 否 | 关联类型：`inner`（默认）、`left`、`right`、`outer` |
| `left_on` | string | 否 | 左侧数据集用于关联的列名 |
| `right_on` | string | 否 | 右侧数据集用于关联的列名 |
| `left_columns` | array | 否 | 指定左侧数据集中需要保留的列 |
| `right_columns` | array | 否 | 指定右侧数据集中需要保留的列 |
| `suffixes` | array | 否 | 列名冲突时的后缀，默认为 `["_left", "_right"]` |

### 示例

#### 1. 基本内关联

```json
{
  "type": "join",
  "name": "用户订单关联",
  "params": {
    "right_dataset_id": "orders",
    "join_type": "inner",
    "left_on": "user_id",
    "right_on": "user_id"
  }
}
```

#### 2. 左关联并指定列

```json
{
  "type": "join",
  "name": "用户及其订单",
  "params": {
    "right_dataset_id": "orders",
    "join_type": "left",
    "left_on": "user_id",
    "right_on": "user_id",
    "left_columns": ["user_id", "name", "email"],
    "right_columns": ["order_id", "product", "amount"]
  }
}
```

## 使用流程

1. 使用 [load_dataset](file://E:\pandas-mcp\server.py#L387-L396) 工具加载至少两个数据集
2. 记录数据集的ID（从加载结果中获得）
3. 在 [execute_analysis_pipeline](file://E:\pandas-mcp\server.py#L113-L183) 中使用 `join` 积木进行关联操作

## 注意事项

- 确保要关联的数据集都已经加载并且ID正确
- 如果不指定 `left_on` 和 `right_on`，系统将基于索引进行关联
- 关联后的结果会替换当前数据集的内容