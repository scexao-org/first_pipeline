import os
import numpy as np
from astropy.io import fits
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

    def __init__(self, data, variance, dark, dark_variance, filename, header):
        self.data = data
        self.variance = variance
        self.dark = dark
        self.dark_variance = dark_variance
        self.dirname = os.path.dirname(filename)
        self.basename = os.path.basename(filename)
        self.filename = filename
        self.header = header
        self.dit = header.get('EXPTIME', 0.0)
        self.gain = header.get('GAIN', 1.0)
        self.Ndit = data.shape[0]
        self.Noutput = data.shape[1]
        self.Nwave = data.shape[2]
        self.modID = int(header.get('X_FIRMID', 0))
        self.modScale = int(header.get('X_FIRMSC', 1))
        self.object_name = header.get('OBJECT', 'Unknown')
        self.wollaston = header.get('X_FIRWOL', 'IN')
        self.add_modulation()

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

        # computing the position angle (PA) of each frame
        THETA_OFFSET = 102.2  # degrees
        self.PAangle = -1 * (180.0 - THETA_OFFSET - self.get_parallactic_angle())[:,:,None]/180*np.pi
        # print(f"Image-rotation angle range: {self.PAangle.min()*180/np.pi} to {self.PAangle.max()*180/np.pi} degrees")

    def add_modulation(self):
        """ 
        Adds modulation information to the data cube.
        Reads the 'MODULATION' extension from the FITS file and extracts xmod and ymod arrays.
        If the extension does not exist, initializes xmod and ymod to zeros.
        """

        # Check if 'MODULATION' extension exists in the FITS file
        with fits.open(self.filename) as hdul:
            if 'MODULATION' not in hdul:
                print(f"WARNING: 'MODULATION' extension not found in {self.filename}")
                xmod = np.zeros(1)
                ymod = np.zeros(1)
            elif hdul[0].header.get('X_FIRMID', -1) < 0:
                xmod = np.zeros(1)
                ymod = np.zeros(1)
            else:
                # reading modulation data
                modulation_data = hdul['MODULATION'].data
                xmod = np.double(modulation_data['xmod'])
                ymod = np.double(modulation_data['ymod'])
                # Ensure xmod and ymod are arrays, even if they are scalars
                if np.isscalar(xmod):
                    xmod = np.array([xmod])
                if np.isscalar(ymod):
                    ymod = np.array([ymod])

            # Fix known issue with ymod[373] if necessary
        if len(xmod) == 595:
            if ymod[373]<1e-5:
                ymod[373]=ymod[372]

        self.xmod = xmod
        self.ymod = ymod
        self.Nmod = len(xmod)
        self.Ncube = self.Ndit//self.Nmod
        if (self.Ncube*self.Nmod)!=self.Ndit:
            print(f"WARNING, CUBE not multiple of modulation pattern (Ncube={self.Ncube}, Nmod={self.Nmod}, Ndit={self.Ndit})")
            print("filling with zeros file: ",self.filename)
            self.Ncube += 1

        size_new = (self.Ncube,self.Nmod,self.Noutput,self.Nwave)
        size_old = np.prod((self.Ndit,self.Noutput,self.Nwave))

        if np.prod(size_new) != size_old:
            data_padded=np.zeros(np.prod(size_new))
            data_padded[:size_old]=self.data.ravel()[:size_old]
            self.data=data_padded.reshape(size_new)

            variance_padded=np.zeros(np.prod(size_new))
            variance_padded[:size_old]=self.variance.ravel()[:size_old]
            self.variance=variance_padded.reshape(size_new)
        else:
            self.data = self.data.reshape(size_new)
            self.variance = self.variance.reshape(size_new)


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

        times = self.time_start + (frame_sampling/2 + np.linspace(0, frame_sampling * self.Ncube * self.Nmod, self.Ncube * self.Nmod) ) * u.s
        par_angles = subaru.parallactic_angle(times, target).deg
        if False:
            par_angle = subaru.parallactic_angle(self.time_start, target).deg
            from_header = self.pupil_PA
            print("Difference between computed and header parangle: ", par_angle - from_header)

        return par_angles.reshape(self.Ncube, self.Nmod)

    ## calculate the projection of the offset in the field based on the current parangle and delta-coordinates of target */
    def project_offsets(self, x_sky, y_sky):
        proj_offsets = np.zeros((*x_sky.shape, 2))
        proj_offsets[..., 0] = np.sin(self.PAangle) * y_sky + np.cos(self.PAangle) * x_sky
        proj_offsets[..., 1] = np.cos(self.PAangle) * y_sky - np.sin(self.PAangle) * x_sky
        proj_offsets[..., 0] += self.x_object
        proj_offsets[..., 1] += self.y_object

        return proj_offsets

    def compute_xy_sky(self,couplingMap):
        """
            usa PA to project on sky the modulation
        """
        x_sky = couplingMap.position[:,0] - self.xmod[:,:,None]
        y_sky = couplingMap.position[:,1] - self.ymod[:,:,None]
        self.ra_dec = self.project_offsets(x_sky,y_sky)

        return self.ra_dec

    def normalize_with_flat(self, flat):
        """
        Normalize the data cube by a flat field.
        Args:
            flat (numpy.ndarray): The flat field to normalize the data cube.
        """
        self.data /= flat
        self.variance /= flat**2

    def compute_flux(self):
        """
        Get the mean spectra of the 38 outputs.
        Returns:
            numpy.ndarray: The mean spectra of the 38 outputs.
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
        center_triangles = points[indexes].mean(axis=1)

        return indexes , center_triangles
    
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
        center_pyramids = center_triangles[delta_triangles < l_mean / 10]

        print(f"Computed {len(good_pyramids)} good pyramids.")

        return good_pyramids, center_pyramids


def extract_datalist(files_with_dark, Nsmooth = 1, Nbin = 1, flat = None, center = True):
    """
    Extracts and processes data cubes from the input files.
    Subtracts dark files, applies wavelength smoothing, and calculates variance.
    Returns the processed data cubes, variance cubes, and a header to save.
    If Nsmooth > 1, the data is smoothed along its wavelength dimension by Nsmooth values.
    If Nbin > 1, the data is binned along its wavelength dimension by Nbin values.
    """

    datalist=[]

    for data_file,dark_file  in files_with_dark.items():

        # reading header data
        header=fits.getheader(data_file)
        # important to cast the data in double!
        data=np.double(fits.getdata(data_file))

        if dark_file is not None:
            data_dark=fits.getdata(dark_file)
            if len(data_dark)==1:
                data_dark=data_dark[0]
                data_dark_std=data_dark[0]*0+12
            else:
                data_dark=data_dark.mean(axis=0)
                data_dark_std=data_dark.std(axis=0)
        else:
            # using default values if we do not know the dark
            data_dark=header["DETBIAS"]*(1+2*header["PIX_WIDE"])
            data_dark_std=12*np.sqrt(1+2*header["PIX_WIDE"])

        data-=data_dark
        gain=header['GAIN']
        data_dark_var=data_dark_std**2
        data_var=data_dark_var+gain*np.abs(data)

        dataCube = DataCube(data, data_var, data_dark, data_dark_var, data_file, header)

        # Normalize the data cube by the flat field if provided
        if flat is not None:
            dataCube.normalize_with_flat(flat)

        # If smoothing and binning is required
        if Nsmooth > 1:
            dataCube.smooth(Nsmooth)
        if Nbin > 1:
            dataCube.bin(Nbin)

        # If centering flux is required, do it after smoothing and binning
        dataCube.compute_flux()
        if center == True:
            dataCube.center_flux_outputs()

        datalist += [dataCube]

    return datalist
