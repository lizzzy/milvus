from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from pymilvus.milvus_client import IndexParams
from openai import OpenAI
import os
from dotenv import load_dotenv
# 加载环境变量
load_dotenv()

# 初始化 Milvus 和百炼
client = MilvusClient(uri="http://localhost:19530")
dashscope_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)

# 准备文本数据
texts = [
    "I love machine learning",
    "Deep learning is a subset of AI",
    "Natural language processing is fascinating",
    "Computer vision is transforming industries",
    "AI is the future"
]

# 生成文本嵌入
def get_embedding(text):
    response = dashscope_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding

embeddings = [get_embedding(text) for text in texts]

# 创建集合
collection_name = "semantic_search"
if client.has_collection(collection_name):
    client.drop_collection(collection_name)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
]
schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
client.create_collection(collection_name=collection_name, schema=schema, metric_type="COSINE")

# 插入数据
data = [
    {"id": i, "vector": embeddings[i], "text": texts[i]}
    for i in range(len(texts))
]
client.insert(collection_name=collection_name, data=data)

# 构建索引
index_params = {"field_name": "vector", "index_type": "HNSW", "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200}}
client.create_index(collection_name=collection_name, index_params=IndexParams(**index_params))
client.load_collection(collection_name)

# 搜索
query = "Artificial intelligence advancements"
query_embedding = get_embedding(query)
results = client.search(
    collection_name=collection_name,
    data=[query_embedding],
    limit=3,
    output_fields=["text"]
)

print("Query:", query)
print("Top results:")
for result in results[0]:
    print(f"Text: {result['entity']['text']}, Distance: {result['distance']}")

# 清理
client.drop_collection(collection_name)