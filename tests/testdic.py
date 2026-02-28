from pydantic import BaseModel


# 定义一个继承自 BaseModel 的类
class Person(BaseModel):
    name: str
    age: int
    city: str = "北京"  # 可以有默认值


# 创建实例
person = Person(name="张三", age=25)

# 方式1: model_dump() - 返回字典（推荐）
print("=== model_dump() ===")
print(person.model_dump())
# 输出: {'name': '张三', 'age': 25, 'city': '北京'}

print()

# 方式2: model_dump_json() - 返回 JSON 字符串
print("=== model_dump_json() ===")
print(person.model_dump_json())
# 输出: {"name":"张三","age":25,"city":"北京"}

print()

# 方式4: json() - 旧版方式（兼容）
print("=== json() ===")
import json
print(json.loads(person.name))
# 输出: {'name': '张三', 'age': 25, 'city': '北京'}

print()

# ===== 嵌套类示例 =====
class Address(BaseModel):
    street: str
    city: str
    country: str = "中国"


class Employee(BaseModel):
    name: str
    age: int
    address: Address  # 嵌套 BaseModel


# 创建嵌套实例
employee = Employee(
    name="李四",
    age=30,
    address=Address(street="中关村大街1号", city="北京")
)

print("=== 嵌套类 model_dump() ===")
print(employee.model_dump())
# 输出: {'name': '李四', 'age': 30, 'address': {'street': '中关村大街1号', 'city': '北京', 'country': '中国'}}

print()

# 嵌套转 JSON
print("=== 嵌套类 model_dump_json() ===")
print(employee.model_dump_json(indent=2))
# 输出格式化的 JSON 字符串


# ===== 从字典创建 =====
print("\n=== 从字典创建实例 ===")
data = {"name": "王五", "age": 28, "city": "上海"}
person2 = Person(**data)
print(person2.model_dump())
