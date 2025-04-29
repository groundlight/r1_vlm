DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You may use each tool as many times as needed. You have access to the following tools to help solve problems: 

{tool_descriptions}

For each step:
1. Start by thinking through your reasoning inside <think> tags. Then either return your answer inside <answer> tags, or use a tool inside <tool> tags.
2. If needed, use a tool by writing its arguments inside <tool> tags. Use one line for each argument in the format 'key: value'. The first line must be 'name: <tool_name>'.
3. You will see the tool's output inside <result> tags.
4. Continue until you can give the final answer inside <answer> tags.

Tools expect specific arguments. Follow the examples carefully for the required keys and expected value formats.
Do not make up tools or arguments that aren't listed. 
If the tool includes the argument "image_name", you must provide it the name of an image from this conversation.
"""


SINGLE_TOOL_PROMPT_TEMPLATE = """\
You may call any of the tools exactly one time. You have access to the following tools to help solve problems: 

{tool_descriptions}

For each step:
1. Start by thinking through your reasoning inside <think> tags. Then either return your answer inside <answer> tags, or use a tool inside <tool> tags.
2. If needed, use a tool by writing its arguments inside <tool> tags. Use one line for each argument in the format 'key: value'. The first line must be 'name: <tool_name>'.
3. You will see the tool's output inside <result> tags.
4. Continue until you can give the final answer inside <answer> tags.

Tools expect specific arguments. Follow the examples carefully for the required keys and expected value formats.
Do not make up tools or arguments that aren't listed. 
If the tool includes the argument "image_name", you must provide it the name of an image from this conversation.
"""

SINGLE_OPTIONAL_TOOL_PROMPT_TEMPLATE = """\
You may call any of the tools exactly one time. You have access to the following tools to help solve problems: 

{tool_descriptions}

For each step:
1. Start by thinking through your reasoning inside <think> tags. Then either return your answer inside <answer> tags, or use a tool inside <tool> tags. You are not required to use a tool if you can answer the question without one.
2. If needed, use a tool by writing its arguments inside <tool> tags. Use one line for each argument in the format 'key: value'. The first line must be 'name: <tool_name>'.
3. You will see the tool's output inside <result> tags.
4. Continue until you can give the final answer inside <answer> tags.

Tools expect specific arguments. Follow the examples carefully for the required keys and expected value formats.
Do not make up tools or arguments that aren't listed.  
If the tool includes the argument "image_name", you must provide it the name of an image from this conversation.
"""
