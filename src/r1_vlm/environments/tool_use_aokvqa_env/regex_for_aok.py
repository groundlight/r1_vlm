# Regex pattern for the complete <answer>...</answer> block.
# Allows any content within the tags. (?s) makes '.' match newlines.
ANSWER_BLOCK_REGEX = r"(?s)<answer>.*?</answer>"

# --- Tool Call Regexes ---

# Permissive pattern for JSON values, used to avoid validating value contents.
# Matches any characters non-greedily up to the next structural element (comma, brace).
_PERMISSIVE_JSON_VALUE = r".*?"

# Core JSON structure for the zoom tool call.
# Enforces keys and structure (colons, commas, braces) but not value formats.
_ZOOM_TOOL_JSON_CORE = (
    r"\{\s*"
    r"\"name\"\s*:\s*\"zoom\"\s*,"
    r"\s*\"args\"\s*:\s*\{\s*"
    r"\"image_name\"\s*:\s*\"input_image\"\s*,"
    r"\s*\"bbox\"\s*:\s*" + _PERMISSIVE_JSON_VALUE + r"\s*,"
    r"\s*\"magnification\"\s*:\s*" + _PERMISSIVE_JSON_VALUE + r"\s*\}\s*"
    r"\}\s*"
)
# Full regex for the <tool> call using the zoom tool's core JSON structure.
ZOOM_TOOL_CALL_REGEX = rf"<tool>\s*{_ZOOM_TOOL_JSON_CORE}\s*</tool>"

# Core JSON structure for the detect_objects tool call.
# Enforces keys and structure but not the format of the 'classes' value.
_DETECT_OBJECTS_JSON_CORE = (
    r"\{\s*"
    r"\"name\"\s*:\s*\"detect_objects\"\s*,"
    r"\s*\"args\"\s*:\s*\{\s*"
    r"\"image_name\"\s*:\s*\"input_image\"\s*,"
    r"\s*\"classes\"\s*:\s*" + _PERMISSIVE_JSON_VALUE + r"\s*\}\s*"
    r"\}\s*"
)
# Full regex for the <tool> call using the detect_objects tool's core JSON structure.
DETECT_OBJECTS_TOOL_CALL_REGEX = rf"<tool>\s*{_DETECT_OBJECTS_JSON_CORE}\s*</tool>"

# Combined regex matching either a valid zoom OR detect_objects tool call.
EITHER_TOOL_CALL_REGEX = f"(?:{ZOOM_TOOL_CALL_REGEX}|{DETECT_OBJECTS_TOOL_CALL_REGEX})"

# --- Final Combined Regex for Model Output ---

# Matches optional arbitrary leading text (e.g., thoughts) followed by
# either a valid tool call OR an answer block, then optional whitespace.
# This regex is intended for use with vLLM's regex-guided generation.
# (?s) allows '.' to match newlines in the leading arbitrary text.
# The non-greedy .*? ensures it matches up to the first valid tool/answer block.
FINAL_OUTPUT_REGEX = rf"(?s).*?(?:{EITHER_TOOL_CALL_REGEX}|{ANSWER_BLOCK_REGEX})\s*"
