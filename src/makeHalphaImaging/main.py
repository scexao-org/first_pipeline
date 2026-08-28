"""Command-line interface for H-alpha continuum-subtracted imaging."""

import argparse

from .run_makeHalphaImaging import process_halpha_imaging


def main():
    parser = argparse.ArgumentParser(
        description='Fit the H-alpha continuum and correlate line residuals across observed positions.')
    parser.add_argument('files', nargs='*', default=['*.fits'], help='Preprocessed OBJECT FITS files')
    parser.add_argument('--object_name', help='OBJECT header value to select')
    parser.add_argument('--dark_files', help='Dark preprocessed FITS files')
    parser.add_argument('--flat_files', help='Flat-map FITS file or directory')
    parser.add_argument('--wave_files', help='Wavelength-map FITS file or directory')
    parser.add_argument('--modID', type=int, help='Modulation pattern ID to select')
    parser.add_argument('--modScale', type=int, help='Modulation scale to select')
    parser.add_argument('--wollaston', help='Wollaston configuration: IN or OUT')
    parser.add_argument('--line_center', type=float, default=656.28, help='H-alpha line center in nm')
    parser.add_argument('--line_width', type=float, default=2.0, help='H-alpha integration width in nm')
    parser.add_argument('--polynomial_degree', type=int, default=2, help='Continuum polynomial degree')
    parser.add_argument('--neighbours', type=int, default=12, help='Number of local spatial samples for correlation')
    args = parser.parse_args()

    process_halpha_imaging(
        file_patterns=args.files or ['*.fits'], object_name=args.object_name,
        dark_patterns=[args.dark_files] if args.dark_files else None,
        flat_patterns=[args.flat_files] if args.flat_files else None,
        wave_patterns=[args.wave_files] if args.wave_files else None,
        modID=args.modID, modScale=args.modScale, wollaston=args.wollaston,
        line_center=args.line_center, line_width=args.line_width,
        polynomial_degree=args.polynomial_degree, neighbours=args.neighbours,
    )


if __name__ == '__main__':
    main()