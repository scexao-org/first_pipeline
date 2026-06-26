#! /usr/bin/env python3
# -*- coding: iso-8859-15 -*-
"""
FIRST Pipeline - Auto Process CLI Interface

Command-line interface for automatically walking a base directory, finding
"firstpl" and "darks" directories, and running the pixel map and preprocessing
steps on each of them.
"""

import argparse


def main():
    """
    Main entry point for the auto process script.
    """
    parser = argparse.ArgumentParser(
        description="Find and process firstpl directories under a base directory."
    )
    parser.add_argument(
        "base_directory",
        nargs="?",
        default="/mnt/sdata01/",
        help="Base directory to search for firstpl and darks directories (default: /mnt/sdata01/).",
    )
    args = parser.parse_args()

    from .run_autoProcess import find_and_process_firstpl_directories

    find_and_process_firstpl_directories(args.base_directory)


if __name__ == "__main__":
    main()
