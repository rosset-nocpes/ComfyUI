import logging
import os

from aiohttp import web
from torchvision.datasets.utils import download_url

import folder_paths
from comfy import model_management
from comfy.sd import load_checkpoint_guess_config
from server import PromptServer

routes = PromptServer.instance.routes


@routes.post("/models/download")
async def download_model(request):
    json_data = await request.json()

    model_url = json_data["url"]
    model_dir = "models/checkpoints"
    filename = str(model_url).split("/")[len(str(model_url).split("/")) - 1]

    filepath = os.path.join(model_dir, filename)
    status = "exists"
    if not os.path.exists(filepath):
        download_url(model_url, model_dir, filename)
        status = "downloaded"

    return web.json_response(
        {"status": status, "filename": filename},
        status=200,
        content_type="application/json",
    )


NODE_CLASS_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS"]
