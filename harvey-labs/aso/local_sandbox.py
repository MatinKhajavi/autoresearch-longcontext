"""LocalSandbox — a drop-in replacement for the Podman `Sandbox`.

Harvey-LAB normally runs agent tool calls inside a rootless Podman container.
That nests badly inside a Modal container (Podman-in-Modal), so for ASO's
fan-out we run tool commands *directly* in the current container instead.

This is the "Approach A" from the design spec: acceptable because tasks use
synthetic data and each Modal container is itself an isolation boundary.

`LocalSandbox` mirrors the public surface that `harness.tools.ToolExecutor`
depends on:
  - attributes `documents_dir`, `output_dir`, `workspace_dir`
  - `start()`, `stop()`
  - `exec(command, *, cwd=WORKSPACE_PATH, timeout=None, env=None) -> ExecResult`

It reuses the real `ExecResult` dataclass so the return shape is identical.

To make the agent's relative bash paths work (`documents/...`, `output/...`)
exactly as they do in the real sandbox — where DOCUMENTS_PATH/OUTPUT_PATH are
subdirs of WORKSPACE_PATH — `start()` symlinks `documents` and `output` under
the workspace directory.
"""

import os
import subprocess
from pathlib import Path

from sandbox.sandbox import DOCUMENTS_PATH, ExecResult, OUTPUT_PATH, WORKSPACE_PATH


class LocalSandbox:
    def __init__(
        self,
        documents_dir,
        output_dir,
        workspace_dir,
        default_timeout: int = 60,
        **_ignored,
    ):
        # Match Sandbox: resolve to absolute paths.
        self.documents_dir = Path(documents_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.default_timeout = default_timeout
        self.container_name = None  # set on start() so callers can detect "running"

    def start(self) -> "LocalSandbox":
        for d in (self.documents_dir, self.output_dir, self.workspace_dir):
            d.mkdir(parents=True, exist_ok=True)
        # Recreate the unified in-container layout: documents/ and output/ live
        # under the workspace so the agent's relative paths resolve.
        self._link("documents", self.documents_dir)
        self._link("output", self.output_dir)
        self.container_name = "local"
        return self

    def stop(self) -> None:
        self.container_name = None

    def _link(self, name: str, target: Path) -> None:
        link = self.workspace_dir / name
        try:
            if link.is_symlink() or link.exists():
                if link.is_symlink() and Path(os.readlink(link)) == target:
                    return
                if link.is_symlink():
                    link.unlink()
                else:
                    return  # a real dir already there — leave it
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            # Symlinks may be unavailable; fall back to leaving workspace as-is.
            pass

    def _to_host(self, sb_path: str) -> Path:
        """Map a sandbox path to its host path (mirrors Sandbox._to_host).

        Order matters: DOCUMENTS_PATH and OUTPUT_PATH are *under* WORKSPACE_PATH,
        so they must be matched before the workspace fallback.
        """
        if sb_path.startswith(DOCUMENTS_PATH):
            return self.documents_dir / sb_path[len(DOCUMENTS_PATH):].lstrip("/")
        if sb_path.startswith(OUTPUT_PATH):
            return self.output_dir / sb_path[len(OUTPUT_PATH):].lstrip("/")
        if sb_path.startswith(WORKSPACE_PATH):
            rel = sb_path[len(WORKSPACE_PATH):].lstrip("/")
            return self.workspace_dir / rel if rel else self.workspace_dir
        # Bare/relative path — resolve against the workspace.
        return self.workspace_dir / sb_path.lstrip("/")

    def _to_host_cwd(self, cwd: str) -> Path:
        """Map a sandbox-relative cwd (e.g. '/workspace[/sub]') to a host path."""
        return self._to_host(cwd)

    # ── file ops ToolExecutor depends on (read/write/edit/glob/grep) ──────
    def exists(self, path: str) -> bool:
        return self._to_host(path).exists()

    def read_file(self, path: str) -> bytes:
        return self._to_host(path).read_bytes()

    def write_file(self, path: str, content: bytes | str) -> None:
        host = self._to_host(path)
        host.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            host.write_text(content, encoding="utf-8")
        else:
            host.write_bytes(content)

    def exec(
        self,
        command: str,
        *,
        cwd: str = WORKSPACE_PATH,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if self.container_name is None:
            raise RuntimeError("sandbox is not running — call start() first")
        timeout = timeout if timeout is not None else self.default_timeout
        run_env = {**os.environ, **(env or {})}
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self._to_host_cwd(cwd)),
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env,
            )
            return ExecResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                returncode=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(
                stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
                stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
                returncode=None,
                timed_out=True,
            )
