"""Command-line interface for H-alpha continuum-subtracted imaging."""

import argparse
from first_pipeline_shared.libraries.runPL_library_cli import check_file_options

from .run_makeHalphaImaging import process_halpha_imaging


def main():
    parser = argparse.ArgumentParser(
        description='Fit the H-alpha continuum and correlate line residuals across observed positions.')
    parser.add_argument('files', nargs='*', default=[], help='Preprocessed OBJECT FITS files')
    parser.add_argument('--object_name', help='OBJECT header value to select')
    parser.add_argument('--dark_files', nargs='+', help='Dark preprocessed FITS files')
    parser.add_argument('--flat_files', nargs='+', help='Flat-map FITS file or directory')
    parser.add_argument('--wave_files', nargs='+', help='Wavelength-map FITS file or directory')
    parser.add_argument('--modID', type=int, help='Modulation pattern ID to select')
    parser.add_argument('--modScale', type=int, help='Modulation scale to select')
    parser.add_argument('--wollaston', help='Wollaston configuration: IN or OUT')
    parser.add_argument('--line_center', type=float, default=656.28, help='H-alpha line center in nm')
    parser.add_argument('--line_width', type=float, default=2.0, help='H-alpha integration width in nm')
    parser.add_argument('--polynomial_degree', type=int, default=2, help='Continuum polynomial degree')
    parser.add_argument('--neighbours', type=int, default=12, help='Number of local spatial samples for correlation')
    args = parser.parse_args()
    check_file_options(parser, args.files, dark_files=args.dark_files, flat_files=args.flat_files, wave_files=args.wave_files)

    process_halpha_imaging(
        file_patterns=args.files or ['*.fits'], object_name=args.object_name,
        dark_patterns=args.dark_files,
        flat_patterns=args.flat_files,
        wave_patterns=args.wave_files,
        modID=args.modID, modScale=args.modScale, wollaston=args.wollaston,
        line_center=args.line_center, line_width=args.line_width,
        polynomial_degree=args.polynomial_degree, neighbours=args.neighbours,
    )


if __name__ == '__main__':
    main()