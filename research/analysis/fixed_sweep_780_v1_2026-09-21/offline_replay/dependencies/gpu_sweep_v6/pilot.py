#!/usr/bin/env python3
"""Small diagnostic run; never starts full.py."""
import sys
from suite import main

if __name__ == "__main__":
    sys.exit(main(["run", "pilot", *sys.argv[1:]]))
