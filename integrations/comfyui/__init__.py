"""ComfyUI custom nodes: clients of a local Splash video server.

V3 entry point only (no NODE_CLASS_MAPPINGS). The comfy import is guarded,
so the package also imports outside ComfyUI (tests, tooling).
"""

try:
    from .nodes import SplashExtension
    _IMPORT_ERROR = None
except ImportError as error:  # not running inside ComfyUI
    SplashExtension = None
    _IMPORT_ERROR = error


async def comfy_entrypoint():
    if SplashExtension is None:
        raise ImportError(f"comfyui-splash needs ComfyUI: {_IMPORT_ERROR}") from _IMPORT_ERROR
    return SplashExtension()
