import os
import numpy as np
from openai import OpenAI

# ===================== 自动填入你的 API Key，绝对不报错 =====================
os.environ["DASHSCOPE_API_KEY"] = "sk-94f98e488ea34b5496d7d54b5c5be57a"
# ==========================================================================

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_MODEL = "text-embedding-v4"

# 测试文本
texts = [
    "我喜欢自然语言处理，尤其是大语言模型。",
    "大模型可以完成文本生成、摘要和问答任务。",
    "今天学校食堂的红烧肉很好吃。",
    "语义向量可以用来计算两个句子的相似度。",
]

def get_embedding(text):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)
    res = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return res.data[0].embedding

def cosine_similarity(vector_a, vector_b):
    a = np.array(vector_a)
    b = np.array(vector_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    # 批量获取向量
    embeddings = [get_embedding(t) for t in texts]

    # 两两相似度
    print("=== 两两相似度 ===")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"句子{i+1} ↔ 句子{j+1}：{sim:.4f}")

    # 检索最相似
    query = "语义向量有哪些作用"
    q_emb = get_embedding(query)
    print("\n=== 最相似句子检索 ===")
    print("查询：", query)

    max_sim = -1
    max_idx = -1
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(q_emb, emb)
        if sim > max_sim:
            max_sim = sim
            max_idx = i

    print(f"最相似：{texts[max_idx]}")
    print(f"相似度：{max_sim:.4f}")