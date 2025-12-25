import logging
from pathlib import Path
from typing import Tuple

from huggingface_hub import snapshot_download

LOGGER = logging.getLogger(__name__)

SD15_BASE_REPO = "runwayml/stable-diffusion-v1-5"
CONTROLNET_REPO = "lllyasviel/sd-controlnet-canny"


def _download_if_missing(repo_id: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    marker = target_dir / ".download_complete"
    if marker.exists():
        LOGGER.info("Model for %s already present at %s", repo_id, target_dir)
        return target_dir

    LOGGER.info("Downloading %s to %s", repo_id, target_dir)
    snapshot_download(repo_id=repo_id, local_dir=target_dir, local_dir_use_symlinks=False)
    marker.touch()
    LOGGER.info("Finished download of %s", repo_id)
    return target_dir


def ensure_sd15_controlnet(model_root: Path) -> Tuple[Path, Path]:
    """Ensure SD 1.5 base and ControlNet weights exist locally.

    Downloads the required model snapshots into the repository's model directory
    on first run. Subsequent invocations are skipped thanks to a local marker file.
    """

    base_dir = model_root / "sd-1.5"
    controlnet_dir = model_root / "controlnet-canny-sd15"

    base_path = _download_if_missing(SD15_BASE_REPO, base_dir)
    control_path = _download_if_missing(CONTROLNET_REPO, controlnet_dir)

    return base_path, control_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    models_root = Path(__file__).resolve().parents[1] / "models"
    ensure_sd15_controlnet(models_root)
