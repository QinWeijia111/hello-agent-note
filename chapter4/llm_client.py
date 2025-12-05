import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict

# 加载 .env 文件中的环境变量
load_dotenv()
# 期望在 .env 或环境中存在以下变量（如未在构造函数显式传入时使用）：
# - LLM_MODEL_ID：要调用的模型标识，例如 "gpt-4o"、"gpt-4o-mini" 或第三方服务的模型名
# - LLM_API_KEY：服务的访问密钥，用于鉴权
# - LLM_BASE_URL：兼容 OpenAI 接口的服务地址，例如 "https://api.openai.com/v1" 或你自建/代理服务
# - LLM_TIMEOUT：网络请求超时时间（单位：秒），可选，默认 60

class HelloAgentsLLM:
    """
    为本书 "Hello Agents" 定制的 LLM 客户端。
    - 目标：以统一的方式调用兼容 OpenAI Chat Completions 接口的服务
    - 适配：OpenAI 官方、第三方代理、本地部署等兼容实现
    - 特性：默认启用流式响应（stream=True），边到边打印并收集模型输出
    - 配置：从 .env 或构造参数读取模型 ID、API 密钥、服务地址与超时
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        参数说明：
        - model：模型标识（如 "gpt-4o-mini" 或其他兼容模型名）
        - apiKey：服务访问密钥；不提供时将读取环境变量 LLM_API_KEY
        - baseUrl：服务地址（兼容 OpenAI 接口）；不提供时读取 LLM_BASE_URL
        - timeout：请求超时秒数；不提供时读取 LLM_TIMEOUT，默认 60
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        # 通过 “参数优先，其次环境变量” 的策略，提升灵活性与复用性
        # 如果关键配置缺失，尽早报错，有助于快速定位问题
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        # 初始化 OpenAI Python 客户端：
        # - api_key：用于鉴权
        # - base_url：指向兼容 OpenAI 接口的服务端（官方或第三方）
        # - timeout：设置网络请求超时，避免调用卡死
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        参数说明：
        - messages：对话消息列表，格式需符合 OpenAI Chat 接口约定
          例如：
          [{"role": "system", "content": "..."},
           {"role": "user", "content": "..."}]
          role 可为 "system" / "user" / "assistant"
        - temperature：采样温度，值越高越发散；0 值更可控、更确定
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            # 发起 chat.completions 请求：
            # - model：选择的模型 ID
            # - messages：多轮对话的消息列表
            # - temperature：控制输出随机性
            # - stream=True：开启流式输出，便于边接收边渲染
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            # 处理流式响应：
            # - 服务端会持续推送 delta（增量）片段
            # - 有些片段的 content 可能是 None，因此使用 “or ''” 兜底
            # - 使用 print(..., end='', flush=True) 即时在终端输出
            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            # 异常捕获：包含网络错误、鉴权失败、配置错误等
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

# --- 客户端使用示例 ---
if __name__ == '__main__':
    try:
        # 创建客户端实例：
        # - 若未显式传入参数，将使用环境变量（LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL / LLM_TIMEOUT）
        # - 请确保 .env 中已配置必要信息
        llmClient = HelloAgentsLLM()
        
        # 构造示例消息：
        # - system 指令用于设定助手的角色或行为准则
        # - user 消息为用户输入
        exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)

    except ValueError as e:
        # 当关键配置缺失时，会在初始化阶段抛出 ValueError
        print(e)
