from __future__ import annotations

import logging
import threading

import ray
from ray import serve

from app.config import AppConfig
from app.service import LinearModelDeployment


def run() -> None:
    config = AppConfig.from_env()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    ray.init(ignore_reinit_error=True, log_to_driver=True)
    serve.run(
        LinearModelDeployment.bind(config),
        host=config.host,
        port=config.port,
        route_prefix="/",
        name="linear-model-service",
    )
    logging.getLogger(__name__).info(
        "Service started", extra={"host": config.host, "port": config.port}
    )

    # Keep the process alive so the service remains available.
    threading.Event().wait()


if __name__ == "__main__":
    run()
