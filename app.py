import asyncio
import json
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional

import pytz
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI, OpenAIError
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


# -----------------------------------------------------------------------
# 0. 配置
# -----------------------------------------------------------------------
shanghai_tz = pytz.timezone("Asia/Shanghai")

credentials = json.load(open("credentials.json"))
active_provider = credentials.get("active_provider")
if not active_provider:
    raise RuntimeError("请在 credentials.json 中指定 active_provider")

if active_provider not in credentials.get("providers", {}):
    available_providers = list(credentials.get("providers", {}).keys())
    raise RuntimeError(f"指定的 provider '{active_provider}' 不存在。可用的 providers: {available_providers}")

provider_config = credentials["providers"][active_provider]

API_KEY_ENV_NAME = provider_config["API_KEY"]
BASE_URL = provider_config["BASE_URL"]
MODEL_NAME = provider_config["model"]

# 从环境变量获取实际的 API_KEY
API_KEY = os.getenv(API_KEY_ENV_NAME)

if not API_KEY:
    raise RuntimeError(f"请在环境变量 {API_KEY_ENV_NAME} 中配置 API_KEY (当前使用provider: {active_provider})")

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

templates = Jinja2Templates(directory="templates")

# -----------------------------------------------------------------------
# 1. FastAPI 初始化
# -----------------------------------------------------------------------
app = FastAPI(title="AI Animation Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    topic: str
    history: Optional[List[dict]] = None

# -----------------------------------------------------------------------
# 2. 核心：流式生成器 (现在会使用 history)
# -----------------------------------------------------------------------
async def llm_event_stream(
    topic: str,
    history: Optional[List[dict]] = None,
    model: str = None,  # 模型名称将从配置文件读取
) -> AsyncGenerator[str, None]:
    history = history or []
    
    # 如果没有指定模型，使用配置文件中的模型
    if model is None:
        model = MODEL_NAME
    
    # The system prompt is now more focused
    system_prompt = f"""
# 角色：AI 动画工程师

# 任务
你的核心任务是根据用户提供的主题“{topic}”，生成一个完整、独立、且视觉效果惊艳的动画视频。
最终交付物必须是一个单一、可直接在浏览器中运行的 HTML 文件，动画需要自动播放。

# 核心要求
1.  **知识讲解**: 从头到尾，清晰地讲解“{topic}”的核心知识点。
2.  **视觉叙事**: 整个动画要像一个流畅的视频，用视觉语言讲述一个完整的故事。
3.  **旁白与字幕**: 必须包含解说旁白和中英双语字幕，以引导观众理解。
4.  **无交互**: 动画必须是自动播放的，不包含任何需要用户点击的交互元素。

# 艺术风格指南
-   **设计体系**: **请采用现代、简洁的扁平化设计风格 (Flat Design)。** 使用简约的几何图形、清晰的字体和明亮的配色。避免使用复杂的渐变、阴影或拟物效果。
-   **配色方案**:
    -   主色: `#4A90E2` (舒缓蓝)
    -   辅色: `#F5A623` (温暖橙)
    -   点缀色: `#7ED321` (活力绿)
    -   背景色: `#F7F9FA` (淡雅灰)
    -   文字颜色: `#333333` (深灰色)
-   **视觉元素**: 大量运用图标、图形、文字等丰富的视觉元素，使画面生动且信息量充足。

# 技术规格
-   **输出格式**: 一个单一、完整的 HTML5 文件 (`<!DOCTYPE html>`)。
-   **分辨率**: 根 SVG 容器 **必须** 使用 `viewBox="0 0 2560 1440"` 以适配 2K 分辨率。
-   **动画库**: **必须** 使用 GSAP 动画库。请通过以下方式引入: `<script src="https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js"></script>`。
-   **动画风格**:
    -   使用 `gsap.timeline()` 来构建结构化的、精准的动画序列。
    -   为所有动画元素应用平滑的缓动效果，例如 `ease: "power2.inOut"`，使动态看起来更自然。
    -   当多个元素按序列出现时，建议使用交错动画 (`stagger: 0.2`) 创造节奏感。
-   **代码质量**:
    -   **代码结构**: CSS 样式**必须**统一放在 `<head>` 内的 `<style>` 标签中。JavaScript 代码**必须**统一放在 `<body>` 结束前的 `<script>` 标签中。
    -   **代码注释**: 请为关键的 JavaScript 函数和复杂的动画逻辑添加简短的中文注释，解释其作用。
    -   **可读性**: 为 HTML 元素 ID 和 CSS 类名使用有意义的、驼峰式 (camelCase) 或短横线分隔 (kebab-case) 的命名方式。
-   **质量保证**: **至关重要**: 确保没有任何元素发生错误的重叠、穿模，或被字幕遮挡。所有视觉信息都必须清晰、准确地传达。
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": topic},
    ]

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            temperature=0.8, 
        )
    except OpenAIError as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return

    async for chunk in response:
        token = chunk.choices[0].delta.content or ""
        if token:
            payload = json.dumps({"token": token}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            await asyncio.sleep(0.001)

    yield 'data: {"event":"[DONE]"}\n\n'

# -----------------------------------------------------------------------
# 3. 路由 (CHANGED: Now a POST request)
# -----------------------------------------------------------------------
@app.post("/generate")
async def generate(
    chat_request: ChatRequest, # CHANGED: Use the Pydantic model
    request: Request,
):
    """
    Main endpoint: POST /generate
    Accepts a JSON body with "topic" and optional "history".
    Returns an SSE stream.
    """
    accumulated_response = ""  # for caching flow results

    async def event_generator():
        nonlocal accumulated_response
        try:
            async for chunk in llm_event_stream(chat_request.topic, chat_request.history):
                accumulated_response += chunk
                if await request.is_disconnected():
                    break
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


    async def wrapped_stream():
        async for chunk in event_generator():
            yield chunk

    headers = {
        "Cache-Control": "no-store",
        "Content-Type": "text/event-stream; charset=utf-8",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(wrapped_stream(), headers=headers)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse(
        "index.html", {
            "request": request,
            "time": datetime.now(shanghai_tz).strftime("%Y%m%d%H%M%S")})

# -----------------------------------------------------------------------
# 4. 本地启动命令
# -----------------------------------------------------------------------
# uvicorn app:app --reload --host 0.0.0.0 --port 8000


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
