
# 导入必要的库
import requests  # 用于发起HTTP请求
import json      # 用于处理JSON数据
import os        # 用于操作环境变量
import re        # 用于正则表达式匹配，解析LLM的输出
from tavily import TavilyClient # Tavily搜索客户端，用于搜索实时信息
from openai import OpenAI # OpenAI官方SDK，用于兼容所有OpenAI接口的模型（如DeepSeek, Ollama等）

# ====================================================================================
# 1. 系统提示词 (System Prompt)
# ====================================================================================
# 这是给AI的核心指令，定义了它的角色、可用工具、回复格式和思考方式。
# 这就是"ReAct" (Reasoning + Acting) 模式的核心：先思考(Thought)，再行动(Action)。
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
Thought: [这里是你的思考过程和下一步计划]
Action: [这里是你要调用的工具，格式为 function_name(arg_name="arg_value")]

# 任务完成:
当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `finish(answer="...")` 来输出最终答案。

请开始吧！
"""

# ====================================================================================
# 2. 工具函数定义
# ====================================================================================
# 这些函数是AI的"手"，用来获取外部信息。

def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    参数:
        city: 城市名称 (如 "Beijing")
    返回:
        描述当前天气的字符串
    """
    print(f"  [工具调用] 正在查询 {city} 的天气...") # 添加日志方便观察
    # API端点，我们请求JSON格式的数据 (format=j1)
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status() 
        # 解析返回的JSON数据
        data = response.json()
        
        # 提取当前天气状况 (wttr.in 返回的数据结构比较深，需要逐层提取)
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value'] # 天气描述 (如 Sunny, Cloudy)
        temp_c = current_condition['temp_C'] # 摄氏温度
        
        result = f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
        print(f"  [工具结果] {result}") # 打印结果
        return result
        
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"


def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    参数:
        city: 城市名称
        weather: 天气情况
    返回:
        景点推荐信息
    """
    print(f"  [工具调用] 正在为 {city} 搜索适合 {weather} 的景点...")
    
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 3. 构造一个精确的查询，结合天气情况，让搜索结果更相关
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答 (AI生成的总结)
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. 处理返回结果
        # 优先使用 Tavily 生成的总结 (answer 字段)
        if response.get("answer"):
            print("  [工具结果] 获取到Tavily的智能总结")
            return response["answer"]
        
        # 如果没有总结，则自己格式化搜索到的原始结果 (results 列表)
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "抱歉，没有找到相关的旅游景点推荐。"

        print(f"  [工具结果] 获取到 {len(formatted_results)} 条搜索结果")
        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"

# 将所有工具函数放入一个字典，方便后续根据函数名字符串动态调用
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}


# ====================================================================================
# 3. LLM 客户端封装
# ====================================================================================

class OpenAICompatibleClient:
    """
    一个通用的LLM客户端，用于调用任何兼容OpenAI接口的服务 (如 DeepSeek, Ollama, Moonshot 等)。
    """
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        # 初始化OpenAI客户端
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用LLM API来生成回应。"""
        print("\n>>> [系统] 正在思考中 (调用LLM)...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            # 发起非流式请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False,
                temperature=0.1 # 降低随机性，让工具调用更稳定
            )
            answer = response.choices[0].message.content
            return answer
        except Exception as e:
            print(f"!!! [错误] 调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


# ====================================================================================
# 4. 主程序配置与执行
# ====================================================================================

# --- 4.1 配置LLM客户端 ---
# 这里配置的是本地 Ollama 服务的地址和模型
API_KEY = "sk-14a4b7cadf544a258bfae9ac24fd2813"
BASE_URL = "https://api.deepseek.com"
MODEL_ID = "deepseek-chat"

# 配置 Tavily 搜索 API (用于联网搜索)
TAVILY_API_KEY="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"
os.environ['TAVILY_API_KEY'] = TAVILY_API_KEY

# 初始化 LLM 客户端
llm = OpenAICompatibleClient(
    model=MODEL_ID,
    api_key=API_KEY,
    base_url=BASE_URL
)

# --- 4.2 初始化任务 ---
user_prompt = "你好，请帮我查询一下今天哈尔滨的天气，然后根据天气推荐一个合适的旅游景点。"
# prompt_history 用于维护对话上下文，让模型知道之前发生了什么
prompt_history = [f"用户请求: {user_prompt}"]

print("="*60)
print(f"用户输入: {user_prompt}")
print("="*60)

# --- 4.3 运行 Agent 主循环 (ReAct Loop) ---
# 我们设置最大循环次数为 5，防止死循环
for i in range(5): 
    print(f"\n--- 第 {i+1} 轮思考与行动 ---")
    
    # 1. 构建完整的 Prompt (包含历史对话)
    full_prompt = "\n".join(prompt_history)
    
    # 2. 调用 LLM 获取回复 (包含 Thought 和 Action)
    llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
    
    # --- 处理 DeepSeek R1 可能产生的 <think> 标签 ---
    # DeepSeek R1 等推理模型会输出 <think>...</think> 的思维链，我们需要将其展示出来但不要影响解析
    think_match = re.search(r'<think>(.*?)</think>', llm_output, re.DOTALL)
    if think_match:
        print(f"\n[深度思考]:\n{think_match.group(1).strip()}\n")
        # 移除 <think> 部分，方便后续解析 Action
        llm_output = re.sub(r'<think>.*?</think>', '', llm_output, flags=re.DOTALL).strip()

    # --- 截断多余输出 ---
    # 有时模型会在输出 Action 后继续自言自语，我们只需要第一对 Thought-Action
    match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
    if match:
        truncated = match.group(1).strip()
        if truncated != llm_output.strip():
            llm_output = truncated
            # print("(已截断模型多余的输出)")

    print(f"\n[模型回复]:\n{llm_output}\n")
    prompt_history.append(llm_output) # 将模型的回复加入历史
    
    # 3. 解析 Action (核心逻辑)
    # 使用正则表达式提取 Action 后面的内容
    action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
    if not action_match:
        print("!!! [解析错误] 模型没有输出有效的 Action，结束任务。")
        break
    
    action_str = action_match.group(1).strip()

    # 4. 执行 Action
    
    # 情况 A: 任务完成 (finish)
    if action_str.startswith("finish"):
        # 提取 finish(answer="...") 中的 answer 内容
        final_answer_match = re.search(r'finish\(answer="(.*)"\)', action_str, re.DOTALL)
        if final_answer_match:
            final_answer = final_answer_match.group(1)
            print("="*60)
            print(f"✅ 任务完成，最终答案:\n{final_answer}")
            print("="*60)
            print(full_prompt)
        else:
            print(f"✅ 任务完成 (无法解析最终答案格式): {action_str}")
        break
    
    # 情况 B: 调用工具
    try:
        # 解析函数名和参数: function_name(arg="val")
        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        # 将参数字符串解析为字典
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
        
        print(f">>> [系统] 准备调用工具: {tool_name} | 参数: {kwargs}")

        if tool_name in available_tools:
            # *** 真正执行工具函数 ***
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

    except Exception as e:
        observation = f"错误:解析或执行工具时出错 - {e}"

    # 5. 记录观察结果 (Observation)
    # 将工具的返回结果包装成 "Observation: ..." 格式，这是 ReAct 模式的标准
    observation_str = f"Observation: {observation}"
    print(f"\n[观察结果]:\n{observation}\n")
    print("-" * 40)
    
    # 将观察结果加入历史，这样下一轮 LLM 就能看到工具返回的信息了
    prompt_history.append(observation_str)
