"""Local Ollama service lifecycle helpers."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from agent_rl.rag.embeddings import OllamaEmbeddingProvider


@dataclass(frozen=True)
class OllamaServiceConfig:
    executable: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-embedding:4b"
    models_dir: str = ""
    startup_timeout_s: int = 60
    poll_interval_s: float = 1.0


class OllamaServiceManager:
    """Starts, warms, and stops a local Ollama embedding service."""

    def __init__(self, config: OllamaServiceConfig | None = None) -> None:
        self.config = config or OllamaServiceConfig()
        self.started_process: subprocess.Popen[str] | None = None

    def ensure_running(self) -> None:
        if self.is_healthy():
            return
        self.start()
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self.is_healthy():
                return
            time.sleep(self.config.poll_interval_s)
        raise TimeoutError(f"Ollama did not become healthy at {self.config.base_url}")

    def start(self) -> None:
        env = os.environ.copy()
        if self.config.models_dir:
            env["OLLAMA_MODELS"] = str(Path(self.config.models_dir))
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.started_process = subprocess.Popen(
            [self.config.executable, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=creationflags,
        )

    def warm(self) -> int:
        self.ensure_running()
        embedding = OllamaEmbeddingProvider(base_url=self.config.base_url, model=self.config.model).embed_query("warmup")
        return len(embedding)

    def is_healthy(self) -> bool:
        return OllamaEmbeddingProvider(base_url=self.config.base_url, model=self.config.model).is_healthy()

    def stop_model(self) -> None:
        subprocess.run([self.config.executable, "stop", self.config.model], check=False, capture_output=True, text=True)

    def stop_process(self) -> None:
        if self.started_process is not None and self.started_process.poll() is None:
            self.started_process.terminate()
