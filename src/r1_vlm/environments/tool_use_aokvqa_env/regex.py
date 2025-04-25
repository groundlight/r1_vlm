# Matches content (at least one character) followed by the closing think tag.
# Assumes the opening <think> tag might be present beforehand (e.g., in the prompt).
# (?s) enables dotall mode ('.' matches newlines).
# .+? matches one or more characters non-greedily.
THINK_CONTENT_CLOSE_TAG_REGEX = r"(?s).+?</think>"

# Matches the full <think>...</think> block including tags.
# (?s) enables dotall mode.
# .*? matches zero or more characters non-greedily.
FULL_THINK_REGEX = r"(?s)<think>.*?</think>"

# Matches either the full think block OR content followed by the closing tag.
# Uses a non-capturing group (?:...) and alternation (|).
EITHER_THINK_REGEX = f"(?:{FULL_THINK_REGEX}|{THINK_CONTENT_CLOSE_TAG_REGEX})"

# Matches the full <answer>...</answer> block including tags.
# (?s) enables dotall mode.
# .*? matches zero or more characters non-greedily.
ANSWER_REGEX = r"(?s)<answer>.*?</answer>"

# --- Tool Regexes ---

# Regex components for JSON values
_JSON_STR = r'\"[^"]*\"'  # Matches a JSON string: "..."
_JSON_INT = r"\d+"  # Matches an integer
_JSON_FLOAT = r"\d+\.?\d*"  # Matches a float or integer (e.g., 1.0, 2, 3.14)
_JSON_BBOX = (
    r"\[\s*"
    + _JSON_INT
    + r"\s*,\s*"
    + _JSON_INT
    + r"\s*,\s*"
    + _JSON_INT
    + r"\s*,\s*"
    + _JSON_INT
    + r"\s*\]"
)  # Matches [int, int, int, int]
# Matches a JSON list of one or more strings: ["str1", "str2", ...]
_JSON_LIST_OF_STRINGS = (
    r"\[\s*"
    + _JSON_STR
    + r"(?:\s*,\s*"
    + _JSON_STR
    + r")*\s*"  # One or more strings, comma-separated
    + r"\]"
)


# Specific regex for the zoom tool call structure
# Assumes order: name, args -> image_name, bbox, magnification
# Added (?s) for potential newlines within the JSON, although unlikely needed with \s*
_ZOOM_TOOL_REGEX_CORE = (
    r"\{\s*"
    r"\"name\"\s*:\s*\"zoom\"\s*,"
    r"\s*\"args\"\s*:\s*\{\s*"
    r"\"image_name\"\s*:\s*\"input_image\"\s*,"
    r"\s*\"bbox\"\s*:\s*" + _JSON_BBOX + r","
    r"\s*\"magnification\"\s*:\s*" + _JSON_FLOAT + r"\s*\}\s*"
    r"\}\s*"
)
ZOOM_TOOL_REGEX = rf"<tool>\s*{_ZOOM_TOOL_REGEX_CORE}\s*</tool>"

# Specific regex for the detect_objects tool call structure
# Assumes order: name, args -> image_name, classes
# Added (?s) for potential newlines within the JSON
_DETECT_OBJECTS_TOOL_REGEX_CORE = (
    r"\{\s*"
    r"\"name\"\s*:\s*\"detect_objects\"\s*,"
    r"\s*\"args\"\s*:\s*\{\s*"
    r"\"image_name\"\s*:\s*\"input_image\"\s*,"
    r"\s*\"classes\"\s*:\s*" + _JSON_LIST_OF_STRINGS + r"\s*\}\s*"
    r"\}\s*"
)
DETECT_OBJECTS_TOOL_REGEX = rf"<tool>\s*{_DETECT_OBJECTS_TOOL_REGEX_CORE}\s*</tool>"


# Matches either the zoom tool OR the detect objects tool structure
# Note: The <tool> tags are now part of the individual tool regexes above
EITHER_TOOL_REGEX = f"(?:{ZOOM_TOOL_REGEX}|{DETECT_OBJECTS_TOOL_REGEX})"

# --- Final Combined Regex ---

# Matches the complete expected output structure:
# ^ (Start)
# EITHER_THINK_REGEX (Think section, full or partial)
# \s* (Optional whitespace)
# (?:EITHER_TOOL_REGEX | ANSWER_REGEX) (Either a valid tool call OR an answer section)
# \s* (Optional whitespace)
# $ (End)
FINAL_OUTPUT_REGEX = (
    rf"^{EITHER_THINK_REGEX}\s*(?:{EITHER_TOOL_REGEX}|{ANSWER_REGEX})\s*$"
)
