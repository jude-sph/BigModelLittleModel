"""Big Model Little Model (BMLM) - Hierarchical agents for Android GUI automation."""

import os

# Silence gRPC and absl logging noise
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Suppress absl warnings before they're imported
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

__version__ = "0.1.0"
