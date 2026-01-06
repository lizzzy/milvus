from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import random

# 1. 连接
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    timeout=60
)
print("Connected to Milvus")

collection_name = "quickstart_collection"

# 2. 删除旧集合
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print(f"Dropped existing collection '{collection_name}'")

# 3. 定义 Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128),   # 向量维度
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50)
]

schema = CollectionSchema(fields=fields)

# 4. 创建集合
collection = Collection(name=collection_name, schema=schema)
print(f"Collection '{collection_name}' created")

# 5. 插入数据
data = [
    [i for i in range(1000)],  # id
    [[random.random() for _ in range(128)] for _ in range(1000)],  # vector
    [f"cat_{i % 3}" for i in range(1000)]  # category
]

collection.insert(data)
print("Inserted 1000 vectors")

# 6. 创建索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",    # 距离度量
    "params": {"M": 16, "efConstruction": 200}
}

collection.create_index(field_name="vector", index_params=index_params)
print("Index created")

# 7. 加载集合
collection.load()
print("Collection loaded")

# 8. 搜索
query_vector = [[random.random() for _ in range(128)]]

results = collection.search(
    data=query_vector,
    anns_field="vector",
    param={"metric_type": "L2", "params": {"ef": 64}},
    limit=5,
    expr="category == 'cat_0'",
    output_fields=["id", "category"]
)

print("\nSearch results:")
for hit in results[0]:
    print(f"ID: {hit.id}, Distance: {hit.distance}, Category: {hit.entity.get('category')}")

# 9. 清理
utility.drop_collection(collection_name)
print("\nCollection dropped")