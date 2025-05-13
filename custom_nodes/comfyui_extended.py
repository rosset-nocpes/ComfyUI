import gc
import logging
import os

from aiohttp import web
from torchvision.datasets.utils import download_url

import folder_paths
from comfy import model_management
from comfy.sd import load_checkpoint_guess_config
from server import PromptServer

routes = PromptServer.instance.routes


def load_model(model_filename: str):
    folder_path = folder_paths.get_folder_paths("checkpoints")[0]
    model_filepath = os.path.join(folder_path, model_filename)
    if not os.path.exists(model_filepath):
        logging.error(f"Model file not found: {model_filepath}")
        return None

    try:
        model_patcher, clip, vae, clipvision = load_checkpoint_guess_config(
            model_filepath,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        logging.info(f"Successfully loaded model: {model_filename}")
        return model_patcher
    except Exception as e:
        logging.error(f"Error loading model {model_filename}: {e}", exc_info=True)
        return None


# doesn't work
def unload_model(model_to_unload):
    """Unloads a specific model."""
    try:
        loaded = model_management.loaded_models()
        # Compare the underlying model object
        models_to_keep = [m for m in loaded if m.model != model_to_unload]
        logging.info(f"Attempting to unload model: {model_to_unload}")
        # Use a large memory_required to ensure the model is unloaded if it's not in models_to_keep
        model_management.free_memory(
            1e30, model_management.get_torch_device(), keep_loaded=models_to_keep
        )
        if model_to_unload not in [
            m.model for m in model_management.current_loaded_models
        ]:
            logging.info(f"Successfully unloaded model: {model_to_unload}")
        else:
            logging.warning(
                f"Model {model_to_unload} might not have been fully unloaded."
            )

        # It's good practice to soft_empty_cache after freeing memory
        model_management.soft_empty_cache()
        gc.collect()

    except Exception as e:
        logging.error(f"Error unloading model {model_to_unload}: {e}", exc_info=True)


@routes.post("/models/download")
async def download_model(request):
    json_data = await request.json()

    model_url = json_data["url"]
    if not isinstance(model_url, str):
        logging.error(f"Invalid model_url type: {type(model_url)}")
        return web.json_response(
            {"code": 400, "message": "model_url must be a string"},
            status=400,
            content_type="application/json",
        )

    model_dir = folder_paths.get_folder_paths("checkpoints")[0]
    filename = str(model_url).split("/")[len(str(model_url).split("/")) - 1]
    if not filename.endswith((".ckpt", ".safetensors")):
        logging.error(f"Invalid model filename extension: {filename}")
        return web.json_response(
            {
                "code": 400,
                "message": "Model filename must end with .ckpt or .safetensors",
            },
            status=400,
            content_type="application/json",
        )

    filepath = os.path.join(model_dir, filename)
    try:
        if not os.path.exists(filepath):
            logging.info(f"Downloading model from {model_url} to {filepath}")
            download_url(model_url, model_dir, filename)
            logging.info(f"Successfully downloaded model: {filename}")
        else:
            logging.info(f"Model already exists: {filepath}")

        return web.json_response(
            {"filename": filename},
            status=200,
            content_type="application/json",
        )
    except Exception as e:
        logging.error(f"Error downloading model {model_url}: {e}", exc_info=True)
        return web.json_response(
            {"code": 500, "message": f"Error downloading model {model_url}: {e}"},
            status=500,
            content_type="application/json",
        )


@routes.post("/models/test")
async def test_model(request):
    json_data = await request.json()

    model_filename = json_data["model_filename"]
    if not isinstance(model_filename, str):
        logging.error(f"Invalid model_filename type: {type(model_filename)}")
        return web.json_response(
            {"code": 400, "message": "model_filename must be a string"},
            status=400,
            content_type="application/json",
        )

    model = load_model(model_filename)

    # Ensure cache is clean before getting stats
    model_management.soft_empty_cache()

    if model is None:
        return web.json_response(
            {"code": 500, "message": f"Failed to load model: {model_filename}"},
            status=500,
            content_type="application/json",
        )

    cpu_device = model_management.torch.device("cpu")
    ram_total = model_management.get_total_memory(cpu_device)
    ram_free = model_management.get_free_memory(cpu_device)
    vram_total = model_management.get_total_memory()
    vram_free = model_management.get_free_memory()

    system_stats = {
        "model_size_on_disk_bytes": model.model_size(),
        "loaded_size_in_vram_bytes": model.loaded_size(),
        "total_ram_bytes": ram_total,
        "free_ram_bytes": ram_free,
        "total_vram_bytes": vram_total,
        "free_vram_bytes": vram_free,
        "used_vram_bytes": vram_total - vram_free,
    }

    unload_model(model)

    return web.json_response(
        {"system_stats": system_stats},
        status=200,
        content_type="application/json",
    )


@routes.post("/models/load")
async def post_load_model(request):
    json_data = await request.json()

    model_filename = json_data["model_filename"]
    if not isinstance(model_filename, str):
        logging.error(f"Invalid model_filename type: {type(model_filename)}")
        return web.json_response(
            {"code": 400, "message": "model_filename must be a string"},
            status=400,
            content_type="application/json",
        )

    model = load_model(model_filename)

    if model is None:
        return web.json_response(
            {"code": 500, "message": f"Failed to load model: {model_filename}"},
            status=500,
            content_type="application/json",
        )

    return web.json_response(
        {"code": 200, "message": f"Successfully loaded model: {model_filename}"},
        status=200,
        content_type="application/json",
    )


NODE_CLASS_MAPPINGS = {}
__all__ = ["NODE_CLASS_MAPPINGS"]
