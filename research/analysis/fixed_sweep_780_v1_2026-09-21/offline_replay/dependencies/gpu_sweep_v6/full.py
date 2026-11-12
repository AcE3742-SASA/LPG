#!/usr/bin/env python3
"""Full data collection; requires a completed, manually reviewed pilot."""
import sys
from suite import main

if __name__ == "__main__":
    sys.exit(main(["run", "full", *sys.argv[1:]]))
