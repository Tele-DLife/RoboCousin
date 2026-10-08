# Adapted from Project EmbodiedGen:
# https://github.com/HorizonRobotics/EmbodiedGen/blob/master/embodied_gen/utils/gpt_clients.py
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
# Modified by RoboCousin contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import base64
import logging
import os
from io import BytesIO
from typing import Optional

import openai
import yaml
from openai import AzureOpenAI, OpenAI  # pip install openai
from PIL import Image
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


__all__ = [
    "QWENclient",
]

PRIMARY_CONFIG_FILE = "cousin_layout/configs/default.yaml"
LOCAL_KEYS_FILE = "cousin_layout/configs/local_keys.yaml"
LEGACY_CONFIG_FILE = "envs/utils/qwen_config.yaml"


class QWENclient:
    """A client to interact with QWEN models via OpenAI or Azure API.

    Supports text and image prompts, connection checking, and configurable parameters.

    Args:
        endpoint (str): API endpoint URL.
        api_key (str): API key for authentication.
        model_name (str, optional): Model name to use.
        api_version (str, optional): API version (for Azure).
        check_connection (bool, optional): Whether to check API connection.
        verbose (bool, optional): Enable verbose logging.

    Example:
        ```sh
        export ENDPOINT="https://yfb-openai-sweden.openai.azure.com"
        export API_KEY="xxxxxx"
        export API_VERSION="2025-03-01-preview"
        export MODEL_NAME="yfb-gpt-4o-sweden"
        ```
        ```py
        from embodied_gen.utils.gpt_clients import GPT_CLIENT

        response = GPT_CLIENT.query("Describe the physics of a falling apple.")
        response = GPT_CLIENT.query(
            text_prompt="Describe the content in each image."
            image_base64=["path/to/image1.png", "path/to/image2.jpg"],
        )
        ```
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model_name: str = "yfb-gpt-4o",
        api_version: str = None,
        check_connection: bool = True,
        # verbose: bool = False,
        verbose: bool = True,
    ):
        if api_version is not None:
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            self.client = OpenAI(
                base_url=endpoint,
                api_key=api_key,
            )

        self.endpoint = endpoint
        self.model_name = model_name
        self.image_formats = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        self.verbose = verbose
        if check_connection:
            self.check_connection()

        logger.info(f"Using GPT model: {self.model_name}.")

    @retry(
        retry=retry_if_not_exception_type(openai.BadRequestError),
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(5),
    )
    def completion_with_backoff(self, **kwargs):
        """Performs a chat completion request with retry/backoff."""
        return self.client.chat.completions.create(**kwargs)

    def query(
        self,
        text_prompt: str,
        image_base64: Optional[list[str | Image.Image]] = None,
        system_role: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> Optional[str]:
        """Queries the GPT model with text and optional image prompts.

        Args:
            text_prompt (str): Main text input.
            image_base64 (Optional[list[str | Image.Image]], optional): List of image base64 strings, file paths, or PIL Images.
            system_role (Optional[str], optional): System-level instructions.
            params (Optional[dict], optional): Additional GPT parameters.

        Returns:
            Optional[str]: Model response content, or None if error.
        """
        if system_role is None:
            system_role = "You are a highly knowledgeable assistant specializing in physics, engineering, and object properties."  # noqa

        content_user = [
            {
                "type": "text",
                "text": text_prompt,
            },
        ]

        # Process images if provided
        if image_base64 is not None:
            if not isinstance(image_base64, list):
                image_base64 = [image_base64]
            # Hardcode tmp because of the openrouter can't input multi images.
            # if "openrouter" in self.endpoint:
            #     image_base64 = combine_images_to_grid(image_base64)
            for img in image_base64:
                if isinstance(img, Image.Image):
                    buffer = BytesIO()
                    img.save(buffer, format=img.format or "PNG")
                    buffer.seek(0)
                    image_binary = buffer.read()
                    img = base64.b64encode(image_binary).decode("utf-8")
                elif (
                    len(os.path.splitext(img)) > 1
                    and os.path.splitext(img)[-1].lower() in self.image_formats
                ):
                    if not os.path.exists(img):
                        raise FileNotFoundError(f"Image file not found: {img}")
                    with open(img, "rb") as f:
                        img = base64.b64encode(f.read()).decode("utf-8")

                content_user.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"},
                    }
                )

        payload = {
            "messages": [
                {"role": "system", "content": system_role},
                {"role": "user", "content": content_user},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "top_p": 0.1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "stop": None,
            "model": self.model_name,
        }

        if params:
            payload.update(params)

        response = None
        try:
            response = self.completion_with_backoff(**payload)
            response = response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error QWENclint {self.endpoint} API call: {e}")
            response = None

        if self.verbose:
            logger.info(f"Prompt: {text_prompt}")
            logger.info(f"Response: {response}")

        return response

    def check_connection(self) -> None:
        """Checks whether the GPT API connection is working.

        Raises:
            ConnectionError: If connection fails.
        """
        try:
            response = self.completion_with_backoff(
                messages=[
                    {"role": "system", "content": "You are a test system."},
                    {"role": "user", "content": "Hello"},
                ],
                model=self.model_name,
                temperature=0,
                max_tokens=100,
            )
            content = response.choices[0].message.content
            logger.info(f"Connection check success.")
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to GPT API at {self.endpoint}, "
                f"please check settings in `{PRIMARY_CONFIG_FILE}` / `{LOCAL_KEYS_FILE}` and `README`."
            )


def _safe_load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _deep_merge(base: dict, extra: dict) -> dict:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_merged_config() -> dict:
    primary = _safe_load_yaml(PRIMARY_CONFIG_FILE)
    local = _safe_load_yaml(LOCAL_KEYS_FILE)
    merged = _deep_merge(primary, local)
    if merged:
        return merged
    return _safe_load_yaml(LEGACY_CONFIG_FILE)


def _resolve_agent_config(config: dict) -> tuple[dict, bool]:
    qwen_cfg = config.get("qwen_client")
    if isinstance(qwen_cfg, dict):
        agent_type = qwen_cfg.get("agent_type")
        profiles = qwen_cfg.get("profiles", {}) or {}
        if not isinstance(profiles, dict):
            profiles = {}
        agent_cfg = profiles.get(agent_type, {}) if agent_type else {}
        if not isinstance(agent_cfg, dict):
            agent_cfg = {}
        merged_cfg = dict(agent_cfg)
        if "api_key" in qwen_cfg and qwen_cfg.get("api_key"):
            merged_cfg["api_key"] = qwen_cfg.get("api_key")
        if "verbose" in qwen_cfg:
            merged_cfg["verbose"] = qwen_cfg.get("verbose")
        return merged_cfg, True

    agent_type = config.get("agent_type")
    if isinstance(agent_type, str):
        agent_cfg = config.get(agent_type, {}) or {}
        if isinstance(agent_cfg, dict):
            return agent_cfg, False
    return {}, False


config = _load_merged_config()
agent_config, is_new_format = _resolve_agent_config(config)

# Prefer environment variables, fallback to YAML config
endpoint = os.environ.get("ENDPOINT", agent_config.get("endpoint"))
api_key = os.environ.get("API_KEY", agent_config.get("api_key"))
api_version = os.environ.get("API_VERSION", agent_config.get("api_version"))
model_name = os.environ.get("MODEL_NAME", agent_config.get("model_name"))
verbose = bool(agent_config.get("verbose", True))

QWEN_CLIENT = QWENclient(
    endpoint=endpoint,
    api_key=api_key,
    api_version=api_version,
    model_name=model_name,
    check_connection=False,
    verbose=verbose,
)


if __name__ == "__main__":
    response = QWEN_CLIENT.query("What is the capital of China?")
    print(f"Response: {response}")


