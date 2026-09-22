"""Distributed sampler; uses LOCAL_RANK for multi-node support."""
import _bootstrap
from sample import main

if __name__ == "__main__":
    main(distributed=True)
