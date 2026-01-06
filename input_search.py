from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema
from pymilvus.milvus_client import IndexParams
from openai import OpenAI
from dotenv import load_dotenv
import os
# 加载环境变量
load_dotenv()

# 初始化
client = MilvusClient(uri="http://localhost:19530")
dashscope_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL")
)


def get_embedding(text):
    response = dashscope_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding


# 初始化数据库（只需运行一次）
def init_database():
    collection_name = os.getenv("COLLECTION_NAME")

    # 准备知识库
    texts = [
        "Python是一种编程语言",
        "机器学习用于预测和分类",
        "深度学习是AI的子领域",
        "向量数据库存储嵌入向量",
        "自然语言处理理解人类语言"
    ]

    embeddings = [get_embedding(text) for text in texts]

    # 创建集合
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0]))
    ]
    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
    client.create_collection(collection_name=collection_name, schema=schema, metric_type="COSINE")

    # 插入数据
    data = [{"id": i, "vector": embeddings[i], "text": texts[i]} for i in range(len(texts))]
    client.insert(collection_name=collection_name, data=data)

    # 建索引
    index_params = {"field_name": "vector", "index_type": "HNSW", "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 200}}
    client.create_index(collection_name=collection_name, index_params=IndexParams(**index_params))
    client.load_collection(collection_name)

    print("知识库初始化完成！\n")


# 搜索函数
def search(query, top_k=3):
    query_embedding = get_embedding(query)
    results = client.search(
        collection_name=os.getenv("COLLECTION_NAME"),
        data=[query_embedding],
        limit=top_k,
        output_fields=["text"]
    )
    return results[0]


# 主程序
if __name__ == "__main__":
    init_database()

    print("=== 智能搜索系统 ===")
    print("输入问题搜索知识库，输入 'quit' 退出\n")

    while True:
        query = input("请输入搜索内容: ").strip()

        if query.lower() == 'quit':
            print("再见！")
            break

        if not query:
            print("输入不能为空\n")
            continue

        print(f"\n搜索: {query}")
        print("-" * 50)

        results = search(query, top_k=3)
        for i, hit in enumerate(results, 1):
            print(f"{i}. {hit['entity']['text']}")
            print(f"   相似度: {hit['distance']:.4f}\n")