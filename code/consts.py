import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_PATH = Path(__file__).parent.parent

load_dotenv(PROJECT_PATH / ".env")

HF_TOKEN = os.getenv("HF_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

EXAMPLE_DATA_PATH = f"{PROJECT_PATH}/example_data"
STORAGE_DIR = "/tmp/pango-converter"

IMAGE_FACTOR = 28
MIN_PIXELS = 100 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 16384 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_RATIO = 200

# [0, 1000) for qwen2vl
MAX_STANDARDIZED_X_COORDINATE = 1000
MAX_STANDARDIZED_Y_COORDINATE = 1000


CUA_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1, y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1, y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1, y1)<|box_end|>')
drag(start_box='<|box_start|>(x1, y1)<|box_end|>', end_box='<|box_start|>(x3, y3)<|box_end|>')
hotkey(key='') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content.
scroll(start_box='<|box_start|>(x1, y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use Chinese in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

GROUNDING_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1, y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1, y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1, y1)<|box_end|>')

## User Instruction
{instruction}"""

STATE_TRANSITION_PROMPT = """You are a GUI agent. You are given two consecutive screenshots of a computer GUI session. You need to determine the action that caused the state transition between the two screenshots.

## Output Format
```
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1, y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1, y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1, y1)<|box_end|>')
"""

UUID_REGEX = (
    r".*([0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}).*"
)
