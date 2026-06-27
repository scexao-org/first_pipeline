import os
import numpy as np
from astroplan import Observer
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.ndimage import uniform_filter1d

from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

subaru = Observer.at_site("Subaru")

class DataCube:
    """
    A class to represent a data cube.
    Attributes:
        data (numpy.ndarray): The data cube.
        variance (numpy.ndarray): The variance of the data cube.
        header (astropy.io.fits.Header): The header information.
    """

    def __init__(self, preproc, dark_preproc=None):
        """
        Build a DataCube from a Preproc object and an optional dark Preproc.

        The DataCube does not read any file itself: the object data, header and
        modulation information are taken from `preproc`, and the dark frame (when
        available) from `dark_preproc`. Dark subtraction and variance estimation
        are performed here.

        Args:
            preproc (Preproc): Loaded Preproc object for the science/object file,
                providing data, filename, header and modulation data.
            dark_preproc (Preproc, optional): Loaded Preproc object for the dark
                file. If None, default dark values are derived from the header.
        """
        self.preproc = preproc
        self.dark_preproc = dark_preproc
        filename = preproc.filename
        header = preproc.header
        self.dirname = os.path.dirname(filename)
        self.basename = os.path.basename(filename)
        self.filename = filename
        self.header = header
        self.dit = header.get('EXPTIME', 0.0)
        self.gain = header.get('GAIN', 1.0)

        # cast data to double, subtract the dark and estimate the variance
        data = np.double(preproc.data)
        self.data, self.variance, self.dark, self.dark_variance = \
            self._subtract_dark_and_variance(data, dark_preproc)

        self.Ndit = self.data.shape[0]
        self.Noutput = self.data.shape[1]
        self.Nwave = self.data.shape[2]
        self.modID = int(header.get('X_FIRMID', 0))
        self.modScale = int(header.get('X_FIRMSC', 1))
        self.object_name = header.get('OBJECT', 'Unknown')
        self.wollaston = header.get('X_FIRWOL', 'IN')

        self.x_object = header.get('X_FIROBX', 0.0)
        self.y_object = header.get('X_FIROBY', 0.0)
        self.target_ra = header.get('D_IMRRA', '21:15:49.440')
        self.target_dec = header.get('D_IMRDEC', '+05:14:52.41')
        self.pupil_PA = header.get('D_IMRPAD', -233.206)
        self.date = header.get('DATE-OBS', '2025-07-14')
        self.ut_str = header.get('UT-STR', "11:52:44.20")
        self.ut_end = header.get('UT-END', "11:53:20.76")
        self.time_start = Time(f"{self.date} {self.ut_str}")
        self.time_end = Time(f"{self.date} {self.ut_end}")
        # Handle case where time_end is before time_start (observation crosses midnight)
        if self.time_end < self.time_start:
            self.time_end += TimeDelta(1, format='jd')

        # getting parameters of metrology glitches
        self.glitch_on=header.get('X_FIRGON', 0)
        self.glitch_frame=header.get('X_FIRGFR', 0)
        self.glitch_delay=header.get('X_FIRGEX', 0) # in ms

        # # computing the position angle (PA) of each frame
        # THETA_OFFSET = 102.2  # degrees
        # # if after September 2025, change value of THETA_OFFSET
        # self.PAangle = -1 * (180.0 - THETA_OFFSET - self.get_parallactic_angle())/180*np.pi

        # new values correct since 2026-06-22:
        THETA_OFFSET = 129.44 - 180 - 37
        self.PAangle = (THETA_OFFSET + self.get_parallactic_angle())/180*np.pi


        
        # print(f"Image-rotation angle range: {self.PAangle.min()*180/np.pi} to {self.PAangle.max()*180/np.pi} degrees")
        self.wave_label = "Pixel Index"
        self.wave = np.arange(self.Nwave)

        self.add_modulation()

    def _subtract_dark_and_variance(self, data, dark_preproc):
        """
        Subtract the dark frame from the data and estimate the variance.

        Args:
            data (numpy.ndarray): Object data already cast to double.
            dark_preproc (Preproc or None): Loaded dark Preproc object, or None
                to fall back on default dark values derived from the header.

        Returns:
            tuple: (data, variance, dark, dark_variance)
        """
        header = self.header

        if dark_preproc is not None:
            data_dark = dark_preproc.data
            if len(data_dark) == 1:
                data_dark = data_dark[0]
                data_dark_std = data_dark[0] * 0 + 12
            else:
                data_dark = data_dark.mean(axis=0)
                data_dark_std = data_dark.std(axis=0)
        else:
            # using default values if we do not know the dark
            data_dark = header["DETBIAS"] * (2 + 2 * header["PIX_WIDE"])
            data_dark_std = 12 * np.sqrt(2 + 2 * header["PIX_WIDE"])

        data -= data_dark
        dark_variance = data_dark_std ** 2
        variance = dark_variance + self.gain * np.abs(data)  # +0.05*np.abs(data)**2
        variance[np.abs(data) > 2 ** 16] = np.inf  # saturating values

        return data, variance, data_dark, dark_variance

    def add_modulation(self):
        """ 
        Adds modulation information to the data cube.
        Reads the modulation data from the Preproc object and extracts xmod and ymod arrays.
        If the modulation data is not available, initializes xmod and ymod to zeros.
        """

        # Read modulation information from the Preproc object
        modulation_data = self.preproc.modulation_data
        if modulation_data is None:
            print(f"WARNING: 'MODULATION' extension not found in {self.filename}")
            xmod = np.zeros(1)
            ymod = np.zeros(1)
        elif self.header.get('X_FIRMID', -1) < 0:
            xmod = np.zeros(1)
            ymod = np.zeros(1)
        else:
            # reading modulation data
            xmod = np.double(modulation_data['xmod'])
            ymod = np.double(modulation_data['ymod'])
            # Ensure xmod and ymod are arrays, even if they are scalars
            if np.isscalar(xmod):
                xmod = np.array([xmod])
            if np.isscalar(ymod):
                ymod = np.array([ymod])

        # Fix known issue with ymod[373] if necessary (only for 2024-2025 data)
        try:
            year = int(self.date.split('-')[0])
            if year >= 2024 and len(xmod) == 595:
                if ymod[373] < 1e-5:
                    ymod[373] = ymod[372]
        except (ValueError, IndexError):
            pass

        # Modulation-to-frame shift measured from the metrology glitch. It is
        # applied when copying the data into the padded cube below, so that the
        # frames shifted out are filled with padding (NaN / inf) instead of
        # wrapping around like np.roll would.
        frame_shift = int(round(self.header.get('X_FIRGSH', 0)))
        if len(xmod) <= 1:
            frame_shift = 0

        self.xmod = xmod
        self.ymod = ymod
        self.Nmod = len(xmod)
        self.Ncube = self.Ndit//self.Nmod
        if (self.Ncube*self.Nmod)!=self.Ndit:
            self.Ncube += 1
            print(f"WARNING, CUBE not multiple of modulation pattern (Ncube={self.Ncube}, Nmod={self.Nmod}, Ndit={self.Ndit})")
            print("filling with zeros file: ",self.filename)

        size_new = (self.Ncube,self.Nmod,self.Noutput,self.Nwave)
        size_old = int(np.prod((self.Ndit,self.Noutput,self.Nwave)))
        size_total = int(np.prod(size_new))

        # Frame shift expressed in flattened-array elements (one frame is
        # Noutput * Nwave elements).
        frame_offset = frame_shift * self.Noutput * self.Nwave

        # Reshape data and variance arrays accordingly to datacube size,
        # applying the frame shift and padding with NaN / inf where needed.
        if (size_total != size_old) or (frame_offset != 0):
            data_padded = np.full(size_total, np.nan)
            variance_padded = np.full(size_total, np.inf)
            PAangle_padded = np.full(self.Ncube * self.Nmod, np.nan)

            # Non-cyclic shift: a positive frame_shift drops the leading frames
            # and appends padding at the end; a negative one prepends padding.
            src_start = max(frame_offset, 0)
            dst_start = max(-frame_offset, 0)
            src_start_PAangle = max(frame_shift, 0)
            dst_start_PAangle = max(-frame_shift, 0)

            n = min(size_old - src_start, size_total - dst_start)
            if n > 0:
                data_padded[dst_start:dst_start + n] = self.data.ravel()[src_start:src_start + n]
                variance_padded[dst_start:dst_start + n] = self.variance.ravel()[src_start:src_start + n]

            # PAangle holds one value per frame, so its shift/count is expressed
            # in frames rather than in flattened data elements.
            n_PAangle = min(self.Ndit - src_start_PAangle,
                            self.Ncube * self.Nmod - dst_start_PAangle)
            if n_PAangle > 0:
                PAangle_padded[dst_start_PAangle:dst_start_PAangle + n_PAangle] = \
                    self.PAangle.ravel()[src_start_PAangle:src_start_PAangle + n_PAangle]

            # The padded frames carry no data (NaN above) and are masked out
            # downstream, but PAangle is a geometric quantity used in
            # (non-NaN-tolerant) matrix inversions. Fill the padded slots with
            # the nearest valid angle so the geometry stays finite.
            valid_mask = ~np.isnan(PAangle_padded)
            if valid_mask.any():
                valid_idx = np.where(valid_mask)[0]
                PAangle_padded[:valid_idx[0]] = PAangle_padded[valid_idx[0]]
                PAangle_padded[valid_idx[-1] + 1:] = PAangle_padded[valid_idx[-1]]

            self.data = data_padded.reshape(size_new)
            self.variance = variance_padded.reshape(size_new)
            self.PAangle = PAangle_padded.reshape(size_new[0], size_new[1])
        else:
            self.data = self.data.reshape(size_new)
            self.variance = self.variance.reshape(size_new)
            self.PAangle = self.PAangle.reshape(size_new[0], size_new[1])


        self.xmod=np.zeros((size_new[0],size_new[1]))
        self.xmod[:]=xmod

        self.ymod=np.zeros((size_new[0],size_new[1]))
        self.ymod[:]=ymod

        return
    
    def get_parallactic_angle(self):
        """
        Calculate the parallactic angle using the Subaru observer from astroplan.
        Returns:
            float: Parallactic angle in degrees.
        """
        target = SkyCoord(self.target_ra, self.target_dec, unit=('hourangle', 'deg'))
        # we could aslo use FRATE, but I am not sure it is better in triggered mode
        frame_sampling = (self.time_end - self.time_start).sec / self.Ndit

        times = self.time_start + (frame_sampling/2 + np.linspace(0, frame_sampling * self.Ndit, self.Ndit) ) * u.s
        par_angles = subaru.parallactic_angle(times, target).deg
        if False:
            par_angle = subaru.parallactic_angle(self.time_start, target).deg
            from_header = self.pupil_PA
            print("Difference between computed and header parangle: ", par_angle - from_header)

        return par_angles

    ## calculate the projection of the offset in the field based on the current parangle and delta-coordinates of target */
    ## note: imversion of function given by mathias nowak:
    # def project_offsets(dra, ddec):
        # proj_offsets = np.zeros(2)
        # derotangle = (THETA_OFFSET + get_parallactic_angle())/180*M_PI
        # proj_offsets[0] = np.sin(derotangle) * ddec - np.cos(derotangle) * dra
        # proj_offsets[1] = np.cos(derotangle) * ddec + np.sin(derotangle) * dra
        # return proj_offsets
        
    def project_on_sky(self, x_mod, y_mod):
        ra_dec = np.zeros((*x_mod.shape, 2))
        PAangle = self.PAangle[..., None]
        dra  = -np.cos(PAangle) * x_mod + np.sin(PAangle) * y_mod
        ddec =  np.sin(PAangle) * x_mod + np.cos(PAangle) * y_mod
        ra_dec[..., 0] = dra
        ra_dec[..., 1] = ddec
        ra_dec[..., 0] += self.x_object
        ra_dec[..., 1] += self.y_object
        return ra_dec

    def compute_xy_sky(self,couplingMap = None):
        """
            use PA to project on sky the modulation
        """
        if couplingMap is not None:
            xmod_2 = self.xmod[:,:,None] - couplingMap.position[:,0]
            ymod_2 = self.ymod[:,:,None] - couplingMap.position[:,1]
        else:
            xmod_2 = self.xmod[:,:,None]
            ymod_2 = self.ymod[:,:,None]

        self.ra_dec = self.project_on_sky(xmod_2, ymod_2)
        if couplingMap is None:
            self.ra_dec = self.ra_dec[:,:,0]

        return self.ra_dec

    def compute_flux(self):
        """
        Compute the common-mode flux by averaging over the outputs (axis 2).
        Stores the result in self.flux with shape (Ncube, Nmod, Nwave).
        """
        self.flux = self.data.mean(axis=(2))


    def center_flux_outputs(self):
        """
        """
        if not hasattr(self, 'flux'):
            self.compute_flux()

        self.data -= self.flux[:, :, None]

        
    def smooth(self, Nsmooth):
        """
        Smooth the data cube.
        """

        self.data = uniform_filter1d(self.data, size=Nsmooth, axis=-1, mode='nearest')
        self.variance = uniform_filter1d(self.variance, size=Nsmooth, axis=-1, mode='nearest')

    def bin(self, Nbin):
        """
        Bin the data cube.
        """
        Nwave = self.data.shape[3]

        self.data = self.data[:, :, :, :(Nwave // Nbin) * Nbin]
        self.variance = self.variance[:, :, :, :(Nwave // Nbin) * Nbin]

        self.data = self.data.reshape((self.Ncube, self.Nmod, self.Noutput, Nwave // Nbin, Nbin)).sum(axis=-1)
        self.variance = self.variance.reshape((self.Ncube, self.Nmod, self.Noutput, Nwave // Nbin, Nbin)).sum(axis=-1)
        
        self.wave = self.wave[:(Nwave // Nbin) * Nbin].reshape((Nwave // Nbin, Nbin)).mean(axis=-1)
        self.Nwave = self.data.shape[3]

    def get_triangles(self,quiet=False):
    
        xmod=self.xmod[0]
        ymod=self.ymod[0]

        # Combine xmod and ymod into a 2D array of points
        points = np.array([xmod, ymod]).T

        # Perform Delaunay triangulation
        delaunay_triangles = Delaunay(points)

        # Extract the triangles
        triangles = delaunay_triangles.simplices
        # Filter triangles to keep only equatorial ones
        good_triangles = []
        for triangle in triangles:
            # Get the y-coordinates of the vertices
            x_coords = points[triangle, 0]
            y_coords = points[triangle, 1]
            l1=np.sqrt((x_coords[0]-x_coords[1])**2+(y_coords[0]-y_coords[1])**2)
            l2=np.sqrt((x_coords[1]-x_coords[2])**2+(y_coords[1]-y_coords[2])**2)
            l3=np.sqrt((x_coords[2]-x_coords[0])**2+(y_coords[2]-y_coords[0])**2)
            # Check if the triangle is equilateral within a tolerance
            lenghts_triangle = np.array([l1, l2, l3])
            l_max = np.max(lenghts_triangle)
            l_min = np.min(lenghts_triangle)

            # good only if l_max/l_min < (1+1.5**2)**.5
            # to avoid edge triangles
            if l_max/l_min < 1.8:
                good_triangles.append(triangle)

        indexes = np.array(good_triangles)

        if not quiet:
            print(f"Computed {len(triangles)} triangles for the given positions, with {len(good_triangles)} good triangles.")

        center_triangles = points[indexes].mean(axis=1)

        orders = center_triangles[:, 0] + center_triangles[:, 1] * 1e5
        indexes = indexes[np.argsort(orders)]

        return indexes 
    
    def get_pyramids(self):

        xmod=self.xmod[0]
        ymod=self.ymod[0]

        # Combine xmod and ymod into a 2D array of points
        points = np.array([xmod, ymod]).T

        indexes_triangles , center_triangles = self.get_triangles(quiet=True)

        # Compute the center of each triangle
        center_triangles = points[indexes_triangles].mean(axis=1)

        # Create a KDTree for efficient nearest neighbor search
        tree = cKDTree(points)
        distances, indices = tree.query(center_triangles, k=6)
        l_mean = np.mean(distances)

        # Filter triangles based on distance criteria to the center (only keep pyramids)
        delta_xy_triangles = (points[indices] - center_triangles[:, None, :]).mean(axis=1)
        delta_triangles = np.sqrt((delta_xy_triangles**2).sum(axis=-1))

        good_pyramids = indices[delta_triangles < l_mean / 10]

        print(f"Computed {len(good_pyramids)} good pyramids.")

        return good_pyramids

