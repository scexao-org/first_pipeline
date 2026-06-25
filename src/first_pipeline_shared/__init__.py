# FIRST Pipeline Version and Package Information
"""
Shared version and package information for FIRST Pipeline.
"""

import os

from astropy.utils import iers


def _configure_optional_iers_suppression():
    """Optionally suppress IERS download warnings via environment variable.

    Default behavior keeps Astropy settings unchanged, so warning messages are
    shown when remote IERS downloads fail. Set
    FIRST_PIPELINE_SUPPRESS_IERS_WARNING=1 to use bundled IERS data without
    online refresh attempts.
    """
    suppress = os.environ.get("FIRST_PIPELINE_SUPPRESS_IERS_WARNING", "0") == "1"
    if suppress:
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None


_configure_optional_iers_suppression()

# Version information
__version__ = "1.1.0"
__author__ = "sylacour"
__email__ = "sylvestre.lacour@observatoiredeparis.psl.eu"
__description__ = "FIRST Pipeline for Visible Photonic Lantern data reduction at SUBARU/SCEXAO"

# Package metadata
__license__ = "MIT"
__copyright__ = "Copyright 2024-2026 SCEXAO Team"
__url__ = "https://github.com/scexao-org/first_pipeline"

# All available package information
__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    '__license__',
    '__copyright__',
    '__url__'
]