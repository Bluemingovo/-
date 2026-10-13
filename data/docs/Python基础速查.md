# Python 基础速查

## 常用内置类型

- list：可变有序序列。
- tuple：不可变有序序列。
- dict：键值映射，查找平均 O(1)。
- set：无序去重集合，支持集合运算。

## 列表推导式

```python
squares = [x * x for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

## 函数与参数

- 默认参数：`def f(a, b=1):`
- 可变参数：`*args`、`**kwargs`
- lambda：`lambda x: x + 1`

## 异常处理

```python
try:
    result = 1 / 0
except ZeroDivisionError as e:
    print("除零错误")
finally:
    print("总会执行")
```

## 文件读写

```python
with open("a.txt", "r", encoding="utf-8") as f:
    content = f.read()
```

## 追加内容
列表推导式也可以嵌套使用。

## 追加内容
列表推导式也可以嵌套使用。
