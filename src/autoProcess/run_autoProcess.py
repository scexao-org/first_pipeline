#%%

"""
FIRST Pipeline - Auto Process Core Functions

Core logic for walking a base directory, finding "firstpl" and "darks"
directories, and running the pixel map and preprocessing steps on each of them.
Separated from the CLI interface for interactive use and modularity.
"""

import os
import subprocess


def find_and_process_firstpl_directories(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "firstpl":
                firstpl_dir = os.path.join(root, dir_name)
                print(f"Processing directory: {firstpl_dir}")
                run_scripts_in_directory_only_with_modulation(firstpl_dir)
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "darks":
                firstpl_dir = os.path.join(root, dir_name)
                print(f"Processing directory: {firstpl_dir}")
                run_scripts_in_directory(firstpl_dir)
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name in ("flats", "neons"):
                firstpl_dir = os.path.join(root, dir_name)
                print(f"Processing directory: {firstpl_dir}")
                run_scripts_in_directory(firstpl_dir)
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "preproc":
                firstpl_dir = os.path.join(root, dir_name)
                print(f"Processing directory: {firstpl_dir}")
                run_flat_and_wave(firstpl_dir)
    copy_data_to_gravity()


def run_scripts_in_directory_only_with_modulation(directory):
    try:
        # Execute runPL_create_pixelMap
        subprocess.run(["runPL_create_pixelMap"], cwd=directory, check=True)
        # Execute runPL_make_preproc
        subprocess.run(["runPL_make_preproc", "--only_with_modulation"], cwd=directory, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing {directory}: {e}")


def run_scripts_in_directory(directory):
    try:
        # Execute runPL_create_pixelMap
        # subprocess.run(["runPL_create_pixelMap"], cwd=directory, check=True)
        # Execute runPL_make_preproc
        subprocess.run(["runPL_make_preproc"], cwd=directory, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing {directory}: {e}")


def run_flat_and_wave(directory):
    try:
        # Execute runPL_create_flatMap
        subprocess.run(["runPL_create_flatMap"], cwd=directory, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running runPL_create_flatMap in {directory}: {e}")
    try:
        # Execute runPL_create_waveMap
        subprocess.run(["runPL_create_waveMap"], cwd=directory, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running runPL_create_waveMap in {directory}: {e}")


def copy_data_to_gravity():
    try:
        # Execute sh_copydatatoGravity.sh
        subprocess.run(["sh_copydatatoGravity.sh"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running sh_copydatatoGravity.sh: {e}")
