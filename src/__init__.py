# FIRST Pipeline Source Package
"""
FIRST Pipeline for Visible Photonic Lantern data reduction at SUBARU/SCEXAO.

This package provides a complete pipeline for processing data from the 
Visible Photonic Lantern instrument, including pixel mapping, preprocessing,
wavelength calibration, coupling maps, and image reconstruction.
"""

# Import version information
try:
    from .first_pipeline_shared import __version__, __author__, __email__, __description__
except ImportError:
    # Fallback version info
    __version__ = "1.1.0"
    __author__ = "sylacour"
    __email__ = "sylvestre.lacour@observatoiredeparis.psl.eu"
    __description__ = "FIRST Pipeline for Visible Photonic Lantern data reduction at SUBARU/SCEXAO"