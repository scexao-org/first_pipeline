#!/usr/bin/env python3
"""
Setup script for FIRST Pipeline
"""

from setuptools import setup, find_packages
import os

# Read the README file for the long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "FIRST Pipeline for Visible Photonic Lantern data reduction at SUBARU/SCEXAO"

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="first-pipeline",
    version="1.0.0",
    description="FIRST Pipeline for Visible Photonic Lantern data reduction at SUBARU/SCEXAO",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="sylacour",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'runPL_changeKeyword=first_pipeline.runPL_changeKeyword:main',
            'runPL_create_pixelMap=first_pipeline.runPL_create_pixelMap:main',
            'runPL_make_preproc=first_pipeline.runPL_make_preproc:main',
            'runPL_create_flatMap=first_pipeline.runPL_create_flatMap:main',
            'runPL_create_waveMap=first_pipeline.runPL_create_waveMap:main',
            'runPL_create_couplingMap=first_pipeline.runPL_create_couplingMap:main',
            'runPL_make_image=first_pipeline.runPL_make_image:main',
            'runPL_make_astrometry=first_pipeline.runPL_make_astrometry:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="astronomy photonic-lantern interferometry data-reduction",
    project_urls={
        "Documentation": "https://github.com/scexao-org/first_pipeline",
        "Source": "https://github.com/scexao-org/first_pipeline",
    },
)