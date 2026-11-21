"""Noisy local Pauli expectations and shared-parameter gradients."""
__version__ = "0.5.0rc3"
__all__ = ["Circuit", "Prepared", "Evaluation", "Truncation", "ConeOverflow", "Topology", "register_backend", "unregister_backend", "available_backends"]

def __getattr__(name):
    if name == "Topology":
        from .topology import Topology
        return Topology
    if name in __all__:
        from . import api
        return getattr(api, name)
    raise AttributeError(name)

from .backends import register_backend, unregister_backend, available_backends
