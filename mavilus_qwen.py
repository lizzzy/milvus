from pymilvus import connections, utility, CollectionSchema, FieldSchema, DataType, Collection
from openai import OpenAI
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 初始化百炼客户端
dashscope_client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = os.getenv("DASHSCOPE_BASE_URL"),
)
def get_embedding(text):
    """使用百炼生成向量embedding"""
    response = dashscope_client.embeddings.create(
        model = os.getenv("EMBEDDING_MODEL"),
        input = text,
    )
    return response.data[0].embedding

# 1. 连接Milvus
connections.connect("default", host="localhost", port=19530)
print(f"Milvus version: {utility.get_server_version()}")

# 创建集合
colletcion_name = "ali_milvus_collection"
if utility.has_collection(colletcion_name):
    utility.drop_collection(colletcion_name)

# 获取向量维度
sample_embedding = get_embedding("测试文本")
dim = len(sample_embedding)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
]

schema = CollectionSchema(fields=fields, description="Ali Collection with embedding")
collection = Collection(name=colletcion_name, schema=schema)
print(f"created collection: {collection.name}, vector dim: {dim}")

# 3. 插入真实数据
texts = [
    "Python是一种编程语言",
    "机器学习是人工智能的分支",
    "深度学习使用神经网络",
    "自然语言处理处理文本数据",
    "计算机视觉分析图像",
    "强化学习通过奖励学习",
    "数据科学结合统计和编程",
    "云计算提供按需计算资源",
    "区块链是分布式账本技术",
    "物联网连接智能设备"
]
print("正在生成向量...")
embeddings = [get_embedding(text) for text in texts]
data = [embeddings, texts]
mr = collection.insert(data)
print(f"inserted {mr.insert_count} records")

# 4. 测试不同索引类型
print("\n==== 测试不同索引 ===")

# 4.1 IVF_FLAT索引
print("\n1. IVF_FLAT索引")
index_params_ivf = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 4} # 聚类中心数
}
collection.create_index(field_name="embedding", index_params=index_params_ivf)
collection.load()

query = "什么是AI"
query_embedding = [get_embedding(query)]
search_param = {
    "metric_type": "COSINE",
    "params": {"nprobe": 2},
}
results = collection.search(data=query_embedding,anns_field="embedding", param=search_param, limit=3, output_fields=["text"])
print(f"查询: {query}")
for res in results[0]:
    print(f" - {res.entity.get('text')}, (相似度: {res.distance: .4f})")

# 4.2 HNSW索引
print("\n2. HNSW索引")
collection.release()
collection.drop_index()
index_params_hnsw = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,    # 每层最大连接数
        "efConstruction": 200   # 构建时搜索范围
    },
}
collection.create_index(field_name="embedding", index_params=index_params_hnsw)
collection.load()

search_param_hnsw = {"metric_type": "COSINE", "params": {"ef": 64}}   # 搜索时范围
print(f"查询: {query}")
for res in results[0]:
    print(f" - {res.entity.get('text')}, (相似度: {res.distance: .4f})")

# 5. 参数对比测试
print("\n === HNSW参数对比 ===")
ef_values = [16, 64, 128]
for ef in ef_values:
    search_param = {
        "metric_type": "COSINE",
        "params": {
            "ef": ef,
        }
    }
    results = collection.search(data=query_embedding,anns_field="embedding", param=search_param, limit=3, output_fields=["text"])
    print(f"\n ef={ef}")
    for res in results[0]:
        print(f" - {res.entity.get('text')}, ({res.distance: .4f})")

# 6. 多查询测试
print("\n=== 多查询测试 ===")
queries = ["编程相关", "AI技术", "新兴技术"]
for q in queries:
    qe = [get_embedding(q)]
    results = collection.search(data=qe, anns_field="embedding", param=search_param_hnsw, limit=2, output_fields=["text"])
    print(f"\n查询: {q}")
    for res in results[0]:
        print(f" - {res.entity.get('text')}, ({res.distance: .4f})")

# 清理
# utility.drop_collection(collection_name)
print(f"\n集合保留: {collection.name}")

