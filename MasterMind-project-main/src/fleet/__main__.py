"""Fleet worker entrypoint: ``python -m src.fleet``

Boots the Credential Enclave (if ENCLAVE_MASTER_KEY is set) and starts
the blocking worker poll loop.  Can be run locally or as a container
process (Kubernetes, Cloud Run, Fly.io, etc.).

    python -m src.fleet
"""

import os

from .worker import Worker, WorkerConfig
from ..logging_setup import get_logger

log = get_logger("fleet.__main__")


def main() -> None:
    enclave_manager = None
    if os.environ.get("ENCLAVE_MASTER_KEY"):
        try:
            from ..enclave import CredentialStore, LifecycleManager as EnclaveManager
            store = CredentialStore.from_env()
            enclave_manager = EnclaveManager(store)
            log.info("Credential Enclave initialised with %d registered credentials.", len(store._credentials))
        except Exception as exc:
            log.warning("Failed to initialise Credential Enclave: %s — running without JIT injection.", exc)
    else:
        log.warning(
            "ENCLAVE_MASTER_KEY not set — credentials will be read directly from "
            "environment variables. Set ENCLAVE_MASTER_KEY and ENCRYPTED_* vars for "
            "production deployments."
        )

    from ..observability import configure_tracing
    configure_tracing()

    config = WorkerConfig()
    worker = Worker(config=config, enclave_manager=enclave_manager)
    worker.start()
    worker.run()


if __name__ == "__main__":
    main()
