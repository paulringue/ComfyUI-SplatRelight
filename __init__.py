"""ComfyUI-SplatRelight - relight animated 3D gaussian splats inside ComfyUI."""

from .nodes import comfy_entrypoint

# Frontend assets (the draggable light gizmo). ComfyUI mounts this directory and
# loads every .js in it as an extension.
WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint", "WEB_DIRECTORY"]
