import os
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from openai import OpenAI

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# 填入你的API Key
os.environ["DASHSCOPE_API_KEY"] = "sk-94f98e488ea34b5496d7d54b5c5be57a"

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen-plus"

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    msg = data.get("message", "")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    client = OpenAI(api_key=api_key, base_url=QWEN_BASE_URL)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": msg}]
    )
    # 确保返回的JSON结构是 {"answer": "..."}
    return {"answer": completion.choices[0].message.content}