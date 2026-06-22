import importlib.metadata

try:
    VERSION = importlib.metadata.version("pyelastica-jax")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"
