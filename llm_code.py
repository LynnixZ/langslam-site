# -*- coding: utf-8 -*-

import os
import json
import io
import contextlib
import traceback
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
import sys
import logging 

load_dotenv()

# --- 核心工具定义 ---
def execute_python_code(code: str) -> str:
    """Executes Python code and captures output for logging."""
    logging.info(f"--- Executing Code ---\n{code}\n----------------------")
    
    string_io = io.StringIO()
    try:
        with contextlib.redirect_stdout(string_io):
            exec(code, globals())
        
        result = string_io.getvalue()
        if not result:
            local_scope = {}
            exec(code, globals(), local_scope)
            result = str(local_scope.get('result', ''))

        logging.info(f"--- Execution Result (Captured Output) ---\n{result}\n------------------------------------------")
        return f"Execution successful. Output: {result}"
        
    except Exception as e:
        error_info = traceback.format_exc()
        captured_output = string_io.getvalue()
        logging.error(f"--- Execution Failed ---\n{error_info}\n----------------------") # Use logging.error for errors
        return f"Execution failed.\nCaptured Output:\n{captured_output}\nError:\n{error_info}"

# --- 支持工具的函数 ---
def _call_model_with_tools(prompt: str, model_name: str, client: OpenAI, tools: list) -> str:
    messages = [{"role": "user", "content": prompt}]
    logging.info("--- Sending initial prompt to model... ---")
    response = client.chat.completions.create(model=model_name, messages=messages, tools=tools, tool_choice="auto")
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    if not tool_calls:
        return response_message.content
    
    logging.info(f"--- Model requested to call tool: {tool_calls[0].function.name} ---")
    messages.append(response_message)
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        if function_name == "execute_python_code":
            function_response = execute_python_code(code=function_args.get("code"))
        else:
            function_response = f"Error: Unknown function {function_name}"
        messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": function_name, "content": function_response})
    
    logging.info("--- Sending tool execution result back to model... ---")
    second_response = client.chat.completions.create(model=model_name, messages=messages)
    return second_response.choices[0].message.content

# (_call_openai_with_tools and _call_deepseek_with_tools will work automatically with the changes above)
# ... (no changes needed here) ...

def _call_gemini_with_tools(prompt: str, model_name: str) -> str:
    """Calls Gemini with the ability to execute Python code using robust logic."""
    try:
        response_text = None
        while response_text is None:
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key: return "Error: GEMINI_API_KEY not found."
            
            client = genai.Client()
            tool_config = genai.types.ToolConfig(
            function_calling_config=genai.types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["get_current_temperature"]
            )
        )
            config = genai.types.GenerateContentConfig(
                tools=[execute_python_code],
                #tool_config=tool_config,
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            # This log message will now appear in your file
            logging.info(f"Response from {model_name}: {response.text}")
            response_text = response.text
            if not response_text:
                logging.warning("--- Gemini returned an empty response. Retrying... ---")
                continue
        return response_text

    except Exception as e:
        logging.error(f"An error occurred with the Gemini API tool call: {e}") # Log the error
        return f"An error occurred with the Gemini API tool call: {e}"

def _call_openai_with_tools(prompt: str, model_name: str) -> str:
    # ... same as before ...
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key: return "Error: OPENAI_API_KEY not found."
    client = OpenAI(api_key=api_key)
    tools = [{"type": "function", "function": {"name": "execute_python_code", "description": "Executes a string of Python code to perform calculations, data analysis, etc. The code must be self-contained.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The Python code to execute."}}, "required": ["code"]},}}]
    return _call_model_with_tools(prompt, model_name, client, tools)

def _call_deepseek_with_tools(prompt: str, model_name: str) -> str:
    # ... same as before ...
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key: return "Error: DEEPSEEK_API_KEY not found."
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    tools = [{"type": "function", "function": {"name": "execute_python_code", "description": "Executes a string of Python code to perform calculations, data analysis, etc. The code must be self-contained.", "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "The Python code to execute."}}, "required": ["code"]},}}]
    return _call_model_with_tools(prompt, model_name, client, tools)

