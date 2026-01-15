import os
import subprocess

def find_and_process_firstpl_directories(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "firstpl":
                firstpl_dir = os.path.join(root, dir_name)
                print(f"Processing directory: {firstpl_dir}")
                run_scripts_in_directory(firstpl_dir)
            if dir_name == "darks":
                firstpl_dir = os.path.join(root, dir_name)
                print(f"Processing directory: {firstpl_dir}")
                run_scripts_in_directory(firstpl_dir)

def run_scripts_in_directory(directory):
    try:
        # Execute runPL_create_pixelMap.py
        # subprocess.run(["runPL_create_pixelMap.py"], cwd=directory, check=True)
        # Execute runPL_make_preproc.py
        subprocess.run(["runPL_make_preproc.py","--only_with_modulation"], cwd=directory, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while processing {directory}: {e}")

if __name__ == "__main__":
    base_directory = "/mnt/datazpool/PL"  # Change this to the desired base directory
    find_and_process_firstpl_directories(base_directory)