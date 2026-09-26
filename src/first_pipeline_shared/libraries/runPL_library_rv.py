"""
FIRST Pipeline - systemic radial velocity of a target from SIMBAD.

The value returned is SIMBAD's "radial velocity" measurement of the object
(heliocentric/barycentric, which differ by ~0.01 km/s), in km/s, together with
its error, quality flag (A = best ... E = worst) and bibliographic reference.

Lookup order:
    1. local cache  (~/.first_pipeline/vsys_cache.json, usable offline)
    2. SIMBAD by name (header OBJECT keyword, e.g. "HD163296")
    3. SIMBAD by position (header D_IMRRA / D_IMRDEC), closest object within
       `radius_arcsec` - needed for header names SIMBAD does not know
       (e.g. "DELSGE", "KAPPEG", "IRAS192050746")

Requires astroquery (pip install astroquery) and network access for 2-3.
"""

import json
import os
import warnings

import numpy as np

CACHE_FILE = os.path.join(os.path.expanduser("~"), ".first_pipeline", "vsys_cache.json")


def _load_cache():
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    try:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=1, sort_keys=True)
    except OSError as e:
        warnings.warn(f"Could not write the vsys cache {CACHE_FILE}: {e}")


def _column(table, *names):
    """Return the first column of `table` whose name matches one of `names` (case-insensitive)."""
    lower = {c.lower(): c for c in table.colnames}
    for n in names:
        if n.lower() in lower:
            return table[lower[n.lower()]]
    return None


def _parse_row(table, i=0):
    """Extract (vsys, err, qual, bibcode, main_id) from a SIMBAD result table row."""
    def get(col):
        if col is None:
            return None
        v = col[i]
        if np.ma.is_masked(v):
            return None
        return v.decode() if isinstance(v, bytes) else v

    vsys = get(_column(table, "rvz_radvel", "RV_VALUE"))
    if vsys is None:
        return None
    err = get(_column(table, "rvz_err", "RVZ_ERROR"))
    return dict(vsys=float(vsys),
                err=None if err is None else float(err),
                qual=get(_column(table, "rvz_qual", "RVZ_QUAL")),
                bibcode=get(_column(table, "rvz_bibcode", "RVZ_BIBCODE")),
                main_id=get(_column(table, "main_id", "MAIN_ID")))


def name_variants(name):
    """
    Spellings of a header OBJECT name to try in SIMBAD, e.g.
    'MWC480' -> ['MWC480', 'MWC 480'], 'HD163296' -> ['HD163296', 'HD 163296'].
    """
    import re
    name = (name or "").strip()
    variants = [name]
    spaced = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", name)
    if spaced != name:
        variants.append(spaced)
    return [v for v in variants if v]


def _simbad():
    from astroquery.simbad import Simbad
    s = Simbad()
    try:                      # astroquery >= 0.4.8 (TAP based)
        s.add_votable_fields("velocity")
    except Exception:         # older astroquery
        s.add_votable_fields("rv_value", "rvz_error", "rvz_qual", "rvz_bibcode")
    return s


def query_systemic_velocity(object_name=None, ra=None, dec=None, radius_arcsec=10.0,
                            use_cache=True):
    """
    Systemic radial velocity of a target (km/s) from SIMBAD.

    Parameters
    ----------
    object_name : str, optional
        Name to resolve (header OBJECT keyword).
    ra, dec : str or float, optional
        Target coordinates ("hh:mm:ss.s", "+dd:mm:ss" or degrees), used when
        the name is unknown to SIMBAD.
    radius_arcsec : float
        Search radius for the position query.
    use_cache : bool
        Read/write the local cache (so a reduction works offline once the
        target has been queried once).

    Returns
    -------
    dict with keys vsys (km/s), err (km/s or None), qual ('A'..'E' or None),
    bibcode, main_id (SIMBAD name) and source.

    Raises
    ------
    LookupError if no velocity is found, RuntimeError if SIMBAD cannot be reached.
    """
    key = (object_name or f"{ra} {dec}").strip().upper()
    cache = _load_cache() if use_cache else {}
    if key in cache:
        result = dict(cache[key], source="cache")
        return result

    try:
        simbad = _simbad()
    except ImportError as e:
        raise RuntimeError("astroquery is needed to query SIMBAD (pip install astroquery)") from e

    result = None
    found_without_rv = []          # SIMBAD objects found, but with no velocity
    try:
        for name in name_variants(object_name):
            table = simbad.query_object(name)
            if table is not None and len(table) > 0:
                result = _parse_row(table)
                if result is not None:
                    break
                main_id = _column(table, "main_id", "MAIN_ID")
                found_without_rv.append(str(main_id[0]) if main_id is not None else name)
        if result is None and ra is not None and dec is not None:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            unit = ("hourangle", "deg") if isinstance(ra, str) else ("deg", "deg")
            coord = SkyCoord(ra, dec, unit=unit)
            table = simbad.query_region(coord, radius=radius_arcsec * u.arcsec)
            if table is not None and len(table) > 0:
                # closest object that has a velocity
                ra_col, dec_col = _column(table, "ra", "RA"), _column(table, "dec", "DEC")
                order = range(len(table))
                if ra_col is not None and dec_col is not None:
                    try:
                        sep = coord.separation(SkyCoord(np.asarray(ra_col, float),
                                                        np.asarray(dec_col, float), unit="deg"))
                        order = np.argsort(sep.arcsec)
                    except Exception:
                        pass
                for i in order:
                    result = _parse_row(table, i)
                    if result is not None:
                        break
                if result is None:
                    main_id = _column(table, "main_id", "MAIN_ID")
                    if main_id is not None:
                        found_without_rv += [str(m) for m in main_id]
    except Exception as e:
        raise RuntimeError(f"SIMBAD query failed for {key!r}: {e}") from e

    if result is None:
        if found_without_rv:
            raise LookupError(f"SIMBAD knows {key!r} (as {', '.join(dict.fromkeys(found_without_rv))}) "
                              f"but lists no radial velocity for it")
        raise LookupError(f"{key!r} not found in SIMBAD (tried the names {name_variants(object_name)}"
                          f"{' and the header coordinates' if ra is not None else ''})")

    if use_cache:
        cache[key] = result
        _save_cache(cache)
    return dict(result, source="SIMBAD")
