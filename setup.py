#!/usr/bin/env python3
"""
Setup script for FIRST Pipeline
"""

from setuptools import setup, find_packages
import os
import sys

# Add src directory to path to import version from package structure
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

try:
    from first_pipeline_shared import __version__, __author__, __email__, __description__
except ImportError:
    # Fallback version info if import fails
    __version__ = "1.1.1"
    __author__ = "sylacour"
    __email__ = "sylvestre.lacour@observatoiredeparis.psl.eu"
    __description__ = "FIRST Pipeline for Visible Photonic Lantern data reduction at SUBARU/SCEXAO"

# Configure package discovery for src layout
def find_src_packages(where="src"):
    packages = find_packages(where=where)
    return packages

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
    version=__version__,
    description=__description__,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author=__author__,
    author_email=__email__,
    python_requires=">=3.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    scripts=[
        "runPL_dfits",
        "sh_copydatatoGravity.sh",
    ],
    entry_points={
        'console_scripts': [
            'runPL_changeKeyword=changeKeyword.main:main',
            'runPL_create_pixelMap=createPixelMap.main:main',
            'runPL_make_preproc=makePreproc.main:main',
            'runPL_create_flatMap=createFlatMap.main:main',
            'runPL_create_waveMap=createWaveMap.main:main',
            'runPL_create_couplingMap=createCouplingMap.main:main',
            'runPL_make_image=makeImage.main:main',
            'runPL_make_astrometry=makeAstrometry.main:main',
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