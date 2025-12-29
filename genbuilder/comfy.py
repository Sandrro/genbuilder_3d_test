"""ComfyUI client utilities used to run the texture pipeline remotely."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import requests
from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class ComfyUIConfig:
    """Configuration for talking to a running ComfyUI instance."""

    base_url: str = "http://127.0.0.1:8188"
    checkpoint: str = "v1-5-pruned-emaonly.ckpt"
    controlnet: str = "control_v11p_sd15_canny.pth"
    sampler: str = "dpmpp_2m"
    scheduler: str = "normal"
    timeout: int = 120


def _build_controlnet_workflow(
    *,
    prompt: str,
    negative_prompt: str,
    control_image: str,
    seed: int,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    config: ComfyUIConfig,
) -> Dict[str, Any]:
    """Build a minimal ComfyUI workflow with SD 1.5 + ControlNet."""

    return {
        "0": {
            "inputs": {
                "ckpt_name": config.checkpoint,
            },
            "class_type": "CheckpointLoaderSimple",
        },
        "1": {
            "inputs": {
                "control_net_name": config.controlnet,
            },
            "class_type": "ControlNetLoader",
        },
        "2": {
            "inputs": {
                "text": prompt,
                "clip": ["0", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "3": {
            "inputs": {
                "text": negative_prompt,
                "clip": ["0", 1],
            },
            "class_type": "CLIPTextEncode",
        },
        "4": {
            "inputs": {
                "image": control_image,
            },
            "class_type": "LoadImage",
        },
        "5": {
            "inputs": {
                "positive": ["2", 0],
                "negative": ["3", 0],
                "control_net": ["1", 0],
                "image": ["4", 0],
                "strength": 1.0,
                "start_percent": 0.0,
                "end_percent": 1.0,
            },
            "class_type": "ControlNetApplyAdvanced",
        },
        "6": {
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1,
            },
            "class_type": "EmptyLatentImage",
        },
        "7": {
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": config.sampler,
                "scheduler": config.scheduler,
                "denoise": 1.0,
                "model": ["0", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "latent_image": ["6", 0],
            },
            "class_type": "KSampler",
        },
        "8": {
            "inputs": {
                "samples": ["7", 0],
                "vae": ["0", 2],
            },
            "class_type": "VAEDecode",
        },
    }


class ComfyUIClient:
    """Tiny helper around the ComfyUI REST API."""

    def __init__(self, config: ComfyUIConfig | None = None) -> None:
        self.config = config or ComfyUIConfig()

    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def upload_image(self, image: Image.Image, name: str) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        files = {"image": (name, buffer, "image/png")}
        resp = requests.post(self._url("/upload/image"), files=files, data={"overwrite": "true"})
        resp.raise_for_status()
        LOGGER.info("Uploaded control image %s to ComfyUI", name)
        return name

    def _queue_prompt(self, workflow: Dict[str, Any]) -> str:
        resp = requests.post(self._url("/prompt"), json={"prompt": workflow}, timeout=self.config.timeout)
        resp.raise_for_status()
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUI did not return a prompt_id")
        return prompt_id

    def _poll_history(self, prompt_id: str) -> Dict[str, Any]:
        deadline = time.time() + self.config.timeout
        while time.time() < deadline:
            resp = requests.get(self._url(f"/history/{prompt_id}"), timeout=self.config.timeout)
            if resp.status_code == 404:
                time.sleep(1)
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get(prompt_id, {}).get("outputs"):
                return data[prompt_id]
            time.sleep(1)
        raise TimeoutError(f"Timed out waiting for ComfyUI results for {prompt_id}")

    def _download_image(self, entry: Dict[str, Any]) -> Image.Image:
        filename = entry.get("filename")
        subfolder = entry.get("subfolder", "")
        filetype = entry.get("type", "output")
        params = {"filename": filename, "subfolder": subfolder, "type": filetype}
        resp = requests.get(self._url("/view"), params=params, timeout=self.config.timeout)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def render_controlnet(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        control_image: Image.Image,
        seed: int,
        steps: int,
        cfg_scale: float,
        width: int,
        height: int,
    ) -> Image.Image:
        filename = f"control_{seed}_{int(time.time())}.png"
        control_name = self.upload_image(control_image, filename)
        workflow = _build_controlnet_workflow(
            prompt=prompt,
            negative_prompt=negative_prompt,
            control_image=control_name,
            seed=seed,
            steps=steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            config=self.config,
        )
        prompt_id = self._queue_prompt(workflow)
        history = self._poll_history(prompt_id)
        for node in history.get("outputs", {}).values():
            images: Iterable[Dict[str, Any]] = node.get("images", [])
            for entry in images:
                return self._download_image(entry)
        raise RuntimeError("ComfyUI returned no images for the workflow")

