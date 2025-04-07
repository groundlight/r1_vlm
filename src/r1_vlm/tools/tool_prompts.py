DEFAULT_TOOL_PROMPT_TEMPLATE = """\
You may use each tool as many times as needed. You have access to the following tools to help solve problems: 

{tool_descriptions}

For each step:
1. Start by thinking through your reasoning inside <think> tags. Then either return your answer inside <answer> tags, or use a tool by writing a JSON command inside <tool> tags.
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
3. You will see the tool's output inside <result> tags.
4. Continue until you can give the final answer inside <answer> tags.

Tools expect specific JSON input formats. Follow the examples carefully.
Do not make up tools or arguments that aren't listed. 
Only use the named arguments for tools. If a tool has kwargs, do not use them. The user will provide these as necessary. 
If the tool includes the argument "image_name", you must provide it the name of an image from this conversation. The user will provide the relevant image data for that image name. 
"""