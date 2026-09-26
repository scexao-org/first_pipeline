"""
Helpers shared by the command-line interfaces (``*/main.py``) of the pipeline.
"""


def check_file_options(parser, files, **pattern_options):
    """Guard against file options that swallowed the positional input files.

    File options such as ``--dark_files`` take one or more values
    (``nargs='+'``) so that a shell wildcard (``--dark_files dark*.fits``)
    keeps all the expanded files.  argparse is greedy: in
    ``prog --dark_files d1.fits d2.fits data/*.fits`` the data files are also
    read as darks and no input file is left.  When no positional file is given
    and one of these options received several values, stop with a clear
    message instead of silently mixing the lists.

    Parameters
    ----------
    parser : argparse.ArgumentParser
    files : list
        Positional input files as parsed (empty when none were given).
    **pattern_options : list or None
        The parsed values of the file options, keyed by option name.
    """
    if files:
        return
    for name, values in pattern_options.items():
        if values is not None and len(values) > 1:
            parser.error(
                f"--{name} received {len(values)} values and no input file was given: "
                f"the input files were probably read as --{name}. Put the input files "
                f"before the options, quote the pattern (--{name} 'dark*.fits' or "
                f"--{name}=dark*.fits), or end the option list with '--'.")
