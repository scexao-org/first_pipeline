from astropy.io import fits
import numpy as np

class CouplingMap:
    def __init__(self, file, pyramids = False):
        cmap_file = fits.open(file)
        header = cmap_file[0].header
        self.wavelength_bin = header['P_CMWBIN']
        if pyramids:
            add_key = "_P"
        else:
            add_key =  "_T"

        self.flux_2_data = cmap_file['F2DATA'+add_key].data
        self.data_2_flux = cmap_file['DATA2F'+add_key].data
        self.QT = cmap_file['QT'+add_key].data
        self.R = cmap_file['R'+add_key].data
        self.position = cmap_file['XY'+add_key].data
        self.flat = cmap_file['FLAT'].data
        self.ref_spectra = cmap_file['SPECTRA'].data

        self.Npositions = self.position.shape[0]

        cmap_file.close()

    def compute_broadband_QR(self, wmin, wmax):
        """
        Compute broadband QR matrices over a wavelength range.
        typically for Nqr=6
        Parameters
        ----------
        wmin : int
            Start index for wavelength range
        wmax : int
            End index for wavelength range

        Returns
        -------
        QT_broadband : np.ndarray
            Shape (Ntriangles, Nqr, wmax-wmin, Nqr)
        R_broadband : np.ndarray
            Shape (Ntriangles, Nqr, Nqr)
        """
        Ntriangles, Nwave, Nqr, _ = self.R.shape
        if wmin < 0 or wmax > Nwave or wmin >= wmax:
            raise ValueError("Invalid wavelength range")

        QT_broadband = np.zeros((Ntriangles, Nqr, (wmax-wmin) * Nqr))
        R_broadband = np.zeros((Ntriangles, Nqr, Nqr))

        for t in range(Ntriangles):

            R_stack = np.vstack(self.R[t,wmin:wmax])        # (num_wave*6, 6)
            Q_new, R_new = np.linalg.qr(R_stack, mode='reduced')     # Q_intermediate: (num_wave*6,6), R_new: (6,6)

            R_broadband[t] = R_new
            QT_broadband[t]= Q_new.T

        return QT_broadband, R_broadband