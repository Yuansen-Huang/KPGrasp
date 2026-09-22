"""Single-GPU entry point sharing the distributed training implementation."""
import _bootstrap
from train_ddp import main

if __name__ == "__main__":
    main()
