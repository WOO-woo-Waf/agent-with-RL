"""Remote RAG service lifecycle helpers."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from agent_rl.config import get_env, get_env_float, get_env_int, load_project_env


@dataclass(frozen=True)
class RemoteRAGServiceConfig:
    """SSH-managed remote embedding/rerank service settings."""

    ssh_host: str = "zjgGroup-A800"
    service_dir: str = "/home/data/nas_hdd/jinglong/waf/novel-embedding-service"
    base_url: str = "http://172.18.36.87:18080"
    cuda_visible_devices: str = "6"
    startup_timeout_s: int = 420
    poll_interval_s: float = 3.0

    @classmethod
    def from_env(cls, env_path: str | Path | None = None, *, base_url: str | None = None) -> "RemoteRAGServiceConfig":
        load_project_env(env_path, start=Path.cwd())
        return cls(
            ssh_host=get_env("RAG_REMOTE_SSH_HOST", cls.ssh_host, aliases=("NOVEL_AGENT_REMOTE_EMBEDDING_SSH_HOST",)),
            service_dir=get_env(
                "RAG_REMOTE_SERVICE_DIR",
                cls.service_dir,
                aliases=("NOVEL_AGENT_REMOTE_EMBEDDING_SERVICE_DIR",),
            ),
            base_url=(base_url or get_env("RAG_EMBEDDING_BASE_URL", cls.base_url, aliases=("NOVEL_AGENT_VECTOR_STORE_URL",))).rstrip("/"),
            cuda_visible_devices=get_env(
                "RAG_REMOTE_CUDA_DEVICES",
                cls.cuda_visible_devices,
                aliases=("NOVEL_AGENT_REMOTE_EMBEDDING_CUDA_DEVICES",),
            ),
            startup_timeout_s=get_env_int(
                "RAG_REMOTE_STARTUP_TIMEOUT_S",
                cls.startup_timeout_s,
                aliases=("NOVEL_AGENT_REMOTE_EMBEDDING_STARTUP_TIMEOUT_S",),
            ),
            poll_interval_s=get_env_float(
                "RAG_REMOTE_POLL_INTERVAL_S",
                cls.poll_interval_s,
                aliases=("NOVEL_AGENT_REMOTE_EMBEDDING_POLL_INTERVAL_S",),
            ),
        )


class RemoteRAGServiceManager:
    """Starts and stops a remote embedding service using SSH scripts."""

    def __init__(self, config: RemoteRAGServiceConfig | None = None) -> None:
        self.config = config or RemoteRAGServiceConfig.from_env()
        self.started_by_manager = False

    def ensure_running(self) -> None:
        if self.is_healthy():
            return
        self.start()
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self.is_healthy():
                return
            time.sleep(self.config.poll_interval_s)
        raise TimeoutError(f"remote RAG service did not become healthy: {self.config.base_url}")

    def start(self) -> None:
        script = (
            f"cd {self.config.service_dir} && "
            f"CUDA_VISIBLE_DEVICES={self.config.cuda_visible_devices} ./run_server.sh"
        )
        self._ssh(script)
        self.started_by_manager = True

    def stop(self) -> None:
        self._ssh(f"cd {self.config.service_dir} && ./stop_server.sh", check=False)

    def is_healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.config.base_url}/health", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return str(payload.get("status", "")).lower() in {"ok", "healthy", "ready"}
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            return False

    def _ssh(self, remote_script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["ssh", "-o", "BatchMode=yes", self.config.ssh_host, remote_script],
            check=check,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
