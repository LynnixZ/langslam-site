import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from llm_code import _call_deepseek_with_tools, _call_gemini_with_tools, _call_openai_with_tools  # Import the functions defined in llm_code.py
# 在脚本的最开始加载 .env 文件中的变量
# 这行代码会自动查找当前目录下的 .env 文件并加载
load_dotenv()

# --- 后续代码无需任何改动 ---

def _call_gemini(prompt: str, model_name: str) -> str:
    """
    Calls the Google Gemini API.
    """
    try:
        # os.environ.get 会自动读取由 load_dotenv() 加载的密钥
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not found. Make sure it's in your .env file."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(contents=prompt)
        print(f"Response from {model_name}: {response.text}")
        return response.text
    except Exception as e:
        return f"An error occurred with the Gemini API: {e}"

def _call_openai(prompt: str, model_name: str) -> str:
    """
    Calls the OpenAI GPT API.
    """
    try:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not found. Make sure it's in your .env file."

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the OpenAI API: {e}"

def _call_deepseek(prompt: str, model_name: str) -> str:
    """
    Calls the DeepSeek API.
    """
    try:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            return "Error: DEEPSEEK_API_KEY not found. Make sure it's in your .env file."

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}]
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the DeepSeek API: {e}"

def get_response(prompt: str, model: str, use_tools: bool = False) -> str:
    """
    Gets a response from the specified model (Gemini, GPT, or DeepSeek).
    """
    # ... same as before ...
    print(f"\n{'='*20}\nCalling model: {model} | Use Tools: {use_tools}\n{'='*20}")
    try:
        if not use_tools:
            if "gemini" in model: return _call_gemini(prompt, model)
            if "gpt" in model: return _call_openai(prompt, model)
            if "deepseek" in model: return _call_deepseek(prompt, model)
        else:
            if "gemini" in model: return _call_gemini_with_tools(prompt, model)
            if "gpt" in model: return _call_openai_with_tools(prompt, model)
            if "deepseek" in model: return _call_deepseek_with_tools(prompt, model)
        return "Unsupported model or configuration specified."
    except Exception as e:
        return f"An unexpected error occurred in get_response: {e}"

# --- Example Usage ---
if __name__ == '__main__':
    # 你不再需要在终端手动设置环境变量了！
    # 只需要确保 .env 文件和这个脚本在同一个目录下。

    my_prompt = "你好，请用中文简单介绍一下什么是黑洞。"

    # --- Call Gemini ---



    code_prompt = "请计算1到100所有整数的和，并告诉我最终结果。"
    gemini_tool_model = "gemini-2.5-flash"
    final_response_with_code = get_response(code_prompt, gemini_tool_model, use_tools=True)
    print("\n--- Final Answer (Tools Enabled) ---")
    print(final_response_with_code)
    print("\n" + "="*50)

    code_prompt = "请计算153683*2546。"
    gemini_tool_model = "gemini-2.5-flash"
    final_response_with_code = get_response(code_prompt, gemini_tool_model, use_tools=True)
    print("\n--- Final Answer (Tools Enabled) ---")
    print(final_response_with_code)
    print("\n" + "="*50)