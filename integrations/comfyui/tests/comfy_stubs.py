"""Minimal stand-ins for the ComfyUI modules nodes.py imports.

Each input and output type accepts the keyword names its real constructor
in ComfyUI's comfy_api/latest/_io.py takes (KEYWORDS, read from ComfyUI
62879ac7) and refuses any other, so a misspelled keyword fails here as it
would in ComfyUI. The real schema is checked separately against a running
ComfyUI (/object_info).
"""

from __future__ import annotations

import sys
import types
from abc import ABC, abstractmethod
from enum import Enum

_BASE = {"display_name", "optional", "tooltip", "lazy", "extra_dict", "raw_link", "advanced"}
_WIDGET = _BASE | {"default", "socketless", "force_input"}
KEYWORDS = {
    "STRING": _WIDGET | {"multiline", "placeholder", "dynamic_prompts"},
    "INT": _WIDGET | {"min", "max", "step", "control_after_generate", "display_mode"},
    "BOOLEAN": _WIDGET | {"label_on", "label_off"},
    "COMBO": (_WIDGET - {"force_input"}) | {"options", "control_after_generate", "upload", "image_folder", "remote"},
    "IMAGE": _BASE,
    "VIDEO": _BASE,
    "AUDIO": _BASE,
    "COMFY_AUTOGROW_V3": {"template", "display_name", "optional", "tooltip", "lazy", "extra_dict"},
}
OUTPUT_KEYWORDS = {"id", "display_name", "tooltip", "is_output_list"}
# What a test reads of an input the node left unset.
UNSET = {"optional": False, "advanced": None, "default": None, "options": (), "min": None, "max": None,
         "step": None, "control_after_generate": None, "display_mode": None, "template": None}


class _Recorded:
    """An input as nodes.py declares it: its id and keyword values as attributes."""

    io_type = None

    def __init__(self, id, **values):
        unknown = sorted(set(values) - KEYWORDS[self.io_type])
        if unknown:
            raise TypeError(f"{self.io_type} input got unexpected keyword {unknown[0]!r}")
        self.id = id
        for name, value in {**UNSET, **values}.items():
            setattr(self, name, list(value) if name == "options" else value)


class _Out:
    io_type = None

    def __init__(self, *args, **values):
        unknown = sorted(set(values) - OUTPUT_KEYWORDS)
        if unknown:
            raise TypeError(f"{self.io_type} output got unexpected keyword {unknown[0]!r}")
        self.id = args[0] if args else values.get("id")


def _io_type(name):
    """A ComfyUI type namespace: its io_type and Input/Output constructors."""
    return type(name, (), {
        "io_type": name,
        "Input": type("Input", (_Recorded,), {"io_type": name}),
        "Output": type("Output", (_Out,), {"io_type": name}),
    })


class _Names:
    """Autogrow's names template: the slot input and the names of its slots."""

    def __init__(self, input, names, min=1):
        self.input, self.names, self.min = input, list(names), min


Autogrow = _io_type("COMFY_AUTOGROW_V3")
Autogrow.TemplateNames = _Names
NumberDisplay = Enum("NumberDisplay", {"number": "number", "slider": "slider"}, type=str)
Hidden = Enum("Hidden", {"unique_id": "UNIQUE_ID"}, type=str)


SCHEMA_FIELDS = {"display_name", "category", "inputs", "outputs", "hidden", "description", "is_output_node"}


class Schema:
    def __init__(self, node_id, **fields):
        unknown = sorted(set(fields) - SCHEMA_FIELDS)
        if unknown:
            raise TypeError(f"Schema got unexpected field {unknown[0]!r}")
        self.node_id = node_id
        self.inputs, self.outputs = fields.pop("inputs", []), fields.pop("outputs", [])
        self.__dict__.update(fields)


class NodeOutput:
    def __init__(self, *args, **_):
        self.args = args


class ComfyNode:
    hidden = None


class ComfyExtension(ABC):
    async def on_load(self):
        pass

    @abstractmethod
    async def get_node_list(self):
        ...


class VideoFromFile:
    def __init__(self, file, **_):
        self.file = file


IO = types.SimpleNamespace(
    Schema=Schema,
    NodeOutput=NodeOutput,
    ComfyNode=ComfyNode,
    Hidden=Hidden,
    NumberDisplay=NumberDisplay,
    **{name: _io_type(io_type) for name, io_type in (
        ("String", "STRING"), ("Int", "INT"), ("Boolean", "BOOLEAN"), ("Combo", "COMBO"),
        ("Image", "IMAGE"), ("Video", "VIDEO"), ("Audio", "AUDIO"))},
    Autogrow=Autogrow,
)
InputImpl = types.SimpleNamespace(VideoFromFile=VideoFromFile)


class InterruptProcessingException(BaseException):
    pass


class ModelManagement(types.ModuleType):
    def __init__(self):
        super().__init__("comfy.model_management")
        self.interrupt_processing = False
        self.unloads = 0
        self.empties = 0
        self.InterruptProcessingException = InterruptProcessingException

    def processing_interrupted(self):
        return self.interrupt_processing

    def throw_exception_if_processing_interrupted(self):
        if self.interrupt_processing:
            self.interrupt_processing = False
            raise InterruptProcessingException()

    def unload_all_models(self):
        self.unloads += 1

    def soft_empty_cache(self, force=False):
        self.empties += 1


class ProgressBar:
    """Records each update_absolute(value, total, preview) as (value, preview)."""

    instances = []

    def __init__(self, total, **_):
        self.total = total
        self.current = 0
        self.updates = []
        ProgressBar.instances.append(self)

    def update_absolute(self, value, *rest):
        self.current = min(value, self.total)
        self.updates.append((self.current, rest[1] if len(rest) > 1 else None))


class _PromptServer:
    def __init__(self):
        self.texts = []

    def send_progress_text(self, text, node_id, sid=None):
        self.texts.append((node_id, text))


STUBBED = ("comfy", "comfy.utils", "comfy.model_management", "comfy_api", "comfy_api.latest", "server")


def install():
    """Put the stubs in sys.modules; returns (model_management, utils, prompt_server)."""
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = ProgressBar
    ProgressBar.instances = []
    model_management = ModelManagement()
    comfy.utils, comfy.model_management = utils, model_management
    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.IO, latest.ComfyExtension, latest.InputImpl = IO, ComfyExtension, InputImpl
    comfy_api.latest = latest
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=_PromptServer())
    sys.modules.update({
        "comfy": comfy,
        "comfy.utils": utils,
        "comfy.model_management": model_management,
        "comfy_api": comfy_api,
        "comfy_api.latest": latest,
        "server": server,
    })
    return model_management, utils, server.PromptServer.instance


def uninstall():
    for name in STUBBED:
        sys.modules.pop(name, None)
