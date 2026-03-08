import asyncio
import http
import json
import logging
import time
import traceback
from collections.abc import Mapping

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        log_request_summaries: bool = False,
        log_request_limit: int = 3,
        prompt_preview_chars: int = 120,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._log_request_summaries = log_request_summaries
        self._log_request_limit = max(0, log_request_limit)
        self._prompt_preview_chars = max(0, prompt_preview_chars)
        self._request_count = 0
        self._last_task_name: str | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info(
                "Websocket policy server ready on %s:%s (metadata_keys=%s)",
                self._host,
                self._port,
                sorted(self._metadata.keys()),
            )
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                self._request_count += 1
                task_name = _stringify_scalar(obs.get("task_name"))
                should_log_summary = self._should_log_summary(task_name)
                if should_log_summary:
                    logger.info(
                        "Request %d summary: %s",
                        self._request_count,
                        json.dumps(self._summarize_observation(obs), sort_keys=True),
                    )
                    self._last_task_name = task_name

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                total_time = time.monotonic() - start_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = total_time
                if should_log_summary:
                    logger.info(
                        "Request %d result: task_name=%s output_keys=%s actions=%s infer_ms=%.3f total_ms=%.3f",
                        self._request_count,
                        task_name,
                        sorted(action.keys()),
                        json.dumps(_summarize_leaf(action.get("actions")), sort_keys=True),
                        infer_time * 1000,
                        total_time * 1000,
                    )

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _should_log_summary(self, task_name: str | None) -> bool:
        if not self._log_request_summaries:
            return False
        if self._request_count <= self._log_request_limit:
            return True
        return task_name != self._last_task_name

    def _summarize_observation(self, obs: Mapping) -> dict:
        summary = {
            "top_level_keys": sorted(str(key) for key in obs.keys()),
            "task_name_present": "task_name" in obs,
            "task_name": _stringify_scalar(obs.get("task_name")),
            "prompt_present": "prompt" in obs,
            "prompt_preview": _truncate_text(_stringify_scalar(obs.get("prompt")), self._prompt_preview_chars),
            "state": _summarize_leaf(obs.get("state")),
        }
        if isinstance(obs.get("images"), Mapping):
            summary["image_keys"] = sorted(str(key) for key in obs["images"].keys())
            summary["images"] = {
                str(key): _summarize_leaf(value)
                for key, value in obs["images"].items()
            }
        if isinstance(obs.get("eef"), Mapping):
            summary["eef_keys"] = sorted(str(key) for key in obs["eef"].keys())
            summary["eef"] = {
                str(key): _summarize_leaf(value)
                for key, value in obs["eef"].items()
            }
        return summary


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None


def _stringify_scalar(value) -> str | None:
    if value is None:
        return None
    if hasattr(value, "shape") and getattr(value, "shape", None) == ():
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _truncate_text(value: str | None, max_chars: int) -> str | None:
    if value is None or max_chars <= 0:
        return value
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}..."


def _summarize_leaf(value) -> dict:
    if value is None:
        return {"present": False}
    summary = {"present": True, "type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        summary["shape"] = list(shape)
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        summary["dtype"] = str(dtype)
    if "shape" not in summary and isinstance(value, (list, tuple)):
        summary["len"] = len(value)
    return summary
