import ast, base64, math, platform, subprocess
from io import BytesIO
from pathlib import Path
from PIL import Image

from .consts import MAX_PIXELS, MIN_PIXELS, PROJECT_PATH


def download_images(storage_dir: str, actions_path: str, max_concurrent: int = 1000):
    system = platform.system().lower()
    arch = "amd64" if platform.machine().lower() in ["x86_64", "amd64"] else "arm64"
    binary_names = {
        "darwin": f"image_downloader_darwin_{arch}",
        "linux": f"image_downloader_linux_{arch}",
        "windows": f"image_downloader_windows_{arch}.exe",
    }

    binary_path = (
        Path(f"{PROJECT_PATH}/code/scripts/builds/image_downloader")
        / binary_names[system]
    )

    if not binary_path.exists():
        raise FileNotFoundError(f"Binary not found: {binary_path}")

    # make executable on Unix
    if system != "windows":
        binary_path.chmod(0o755)

    subprocess.run(
        [
            str(binary_path),
            "--screenshot_path",
            storage_dir,
            "--actions_path",
            actions_path,
            "--max_concurrent",
            str(max_concurrent),
        ],
        check=True,
    )


def ensure_valid_image(image: Image.Image):
    if image.width * image.height > MAX_PIXELS:
        resize_factor = math.sqrt(MAX_PIXELS / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))
    if image.width * image.height < MIN_PIXELS:
        resize_factor = math.sqrt(MIN_PIXELS / (image.width * image.height))
        width, height = math.ceil(image.width * resize_factor), math.ceil(
            image.height * resize_factor
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pil_to_base64(image: Image.Image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_action(action_str):
    try:
        # 解析字符串为 AST 节点
        node = ast.parse(action_str, mode="eval")

        # 确保节点是一个表达式
        if not isinstance(node, ast.Expression):
            raise ValueError("Not an expression")

        # 获取表达式的主体
        call = node.body

        # 确保主体是一个函数调用
        if not isinstance(call, ast.Call):
            raise ValueError("Not a function call")

        # 获取函数名
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None

        # 获取关键字参数
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            # 处理不同类型的值，这里假设都是常量
            if isinstance(kw.value, ast.Constant):
                value = kw.value.value
            elif isinstance(kw.value, ast.Str):  # 兼容旧版本 Python
                value = kw.value.s
            else:
                value = None
            kwargs[key] = value

        return {"function": func_name, "args": kwargs}

    except Exception as e:
        print(f"Failed to parse action '{action_str}': {e}")
        return None
