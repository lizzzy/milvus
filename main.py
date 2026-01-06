from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
import numpy as np
# 2. 连接本地Milvus
connections.connect("default", host="localhost", port=19530)
# 检查连接状态
print(utility.get_server_version())

# 3. 创建集合
# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=256),
]
# 创建集合模式
schema = CollectionSchema(fields=fields, description="Example collection")
collection = Collection(name="test_collection", schema=schema)
print(f"Collection created: {collection.name}")

# 4. 插入数据
# 生成随机向量
vectors = np.random.random(size=(10, 128)).astype(np.float32)
# 插入数据
ids = [i for i in range(10)]
texts = [f"text_{i}" for i in range(10)]
mr = collection.insert([ids, vectors, texts])
print(f"Inserted {mr.insert_count} records")

# 5. 创建索引
# 创建HNSW索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params":{
        "M": 16,
        "efConstruction": 200
    }
}
collection.create_index(field_name="embedding", index_params=index_params)
print("Index Created")


# 加载集合到内存
collection.load()

# 6. 向量检索
# 生成查询向量
query_vector = np.random.rand(1, 128).astype(np.float32)
# 执行相似度搜索
search_param = {
    "metric_type": "L2",
    "params":{
        "ef": 64
    }
}
results = collection.search(
    data=query_vector,
    anns_field="embedding",
    param=search_param,
    limit=3,
    output_fields=["text"]
)
# 打印结果
for res in results[0]:
    print(f"ID: {res.id}, Distance: {res.distance}, Text: {res.entity.get('text')}")