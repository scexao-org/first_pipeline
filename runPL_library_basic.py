from astropy.io import fits
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit

from scipy.interpolate import griddata

class PixelMap:
    def __init__(self, file):
        self.file = file
        self.header = fits.getheader(file)
        self.traces_loc = fits.getdata(file)
        self.pixel_min = self.header.get('PIX_MIN', 100)
        self.pixel_max = self.header.get('PIX_MAX', 1600)
        self.pixel_wide = self.header.get('PIX_WIDE', 2)
        self.output_channels = self.header.get('OUT_CHAN', 38)

def preprocess_cutData(data, pixelMap, dark_calculation=False):
    """
    Preprocesses and extracts specific pixel data from the input data array based on the provided pixel map.
    This function cuts out pixel data from the input `data` array according to the pixel locations and 
    configurations defined in the `pixelMap` object. It also optionally calculates dark pixels based on 
    the traces' locations.
    Args:
        data (numpy.ndarray): The input data array of shape (Nimages, height, width) or (height, width).
                              If 2D, it is automatically expanded to 3D with a single image.
        pixelMap (object): An object containing the following attributes:
            - pixel_min (int): The minimum pixel index to consider.
            - pixel_max (int): The maximum pixel index to consider.
            - pixel_wide (int): The width of the pixel window to extract.
            - output_channels (int): The number of output channels.
            - traces_loc (numpy.ndarray): A 2D array specifying the trace locations for each pixel.
        dark_calculation (bool, optional): If True, calculates dark pixels for adjacent traces. 
                                           Defaults to False.
    Returns:
        tuple: A tuple containing:
            - data_cut_pixels (numpy.ndarray): A 4D array of shape 
              (Nimages, output_channels, Nwave, window_size) containing the extracted pixel data.
            - data_dark_pixels (numpy.ndarray): A 3D array of shape 
              (Nimages, output_channels - 1, Nwave) containing the calculated dark pixels. 
              This is returned only if `dark_calculation` is True; otherwise, it is an empty array.
    Notes:
        - `Nwave` is calculated as `pixel_max - pixel_min`.
        - `window_size` is calculated as `(pixel_wide * 2 + 1)`.
        - The function ensures that pixel indices do not go out of bounds when accessing the `data` array.
    """

    pixel_min = pixelMap.pixel_min
    pixel_max = pixelMap.pixel_max
    pixel_wide = pixelMap.pixel_wide
    output_channels = pixelMap.output_channels
    traces_loc = pixelMap.traces_loc

    Nwave = pixel_max - pixel_min
    window_size = (pixel_wide * 2 + 1)

    add_dimension_for_cubelike_data = False
    if len(data.shape) == 2:
        add_dimension_for_cubelike_data = True
        data = data[None]

    Nimages = data.shape[0] 

    data_cut_pixels = np.zeros((Nimages, output_channels, Nwave, window_size), dtype='uint16')
    data_dark_pixels = np.zeros((Nimages, output_channels - 1, Nwave), dtype='uint16')
    for x in range(Nwave):
        for i in range(output_channels):
            for w in range(pixel_wide*2+1):
                t=traces_loc[x + pixel_min, i]+w-pixel_wide
                if t<0:
                    t=0
                if t>=data.shape[1]:
                    t=data.shape[1]-1
                data_cut_pixels[:,i,x,w] = data[:, t, x + pixel_min]
            if (i > 0)&(dark_calculation):
                t=(traces_loc[x + pixel_min, i-1]+traces_loc[x + pixel_min, i])//2+w-pixel_wide
                data_dark_pixels[:,i-1,x] = data[:, t, x + pixel_min]
    
    if add_dimension_for_cubelike_data:
        data_cut_pixels = data_cut_pixels[0]
        data_dark_pixels = data_dark_pixels[0]

    return data_cut_pixels, data_dark_pixels


class CouplingMap:
    def __init__(self, file):

        cmap_file=fits.open(file)
        header = cmap_file[0].header
        self.wavelength_bin = header['WL_BIN']

        self.flux_2_data=cmap_file['F2DATA'].data
        self.data_2_flux=cmap_file['DATA2F'].data  # the only matrix usefull for the image reconstruction
        self.fluxtiptilt_2_data=cmap_file['FTT2DATA'].data
        self.data_2_fluxtiptilt=cmap_file['DATA2FTT'].data

        self.xpos=cmap_file['POSITIONS'].data.field('X_POS')
        self.ypos=cmap_file['POSITIONS'].data.field('Y_POS')
        self.Npositions=cmap_file['POSITIONS'].header['NAXIS2']

        self.xtri=cmap_file['TRIANGLES'].data.field('X_TRI')
        self.ytri=cmap_file['TRIANGLES'].data.field('Y_TRI')
        self.Ntriangles=cmap_file['TRIANGLES'].header['NAXIS2']

        cmap_file.close()


def make_image_source_removal(datacube,arg_triangle,couplingMap):
    """
    Removes the source contribution from a datacube using a coupling map and 
    returns the residual datacube and the flux-tip-tilt (FFT) fit.
    Parameters:
    -----------
    datacube : numpy.ndarray
        The input data cube. It can have dimensions of 2D, 3D, or 4D:
        - 2D: (Nwave, Noutput)
        - 3D: (Nwave, Noutput, Ncube)
        - 4D: (Nwave, Noutput, Ncube, Nmod)
        The function will adjust the shape internally to ensure compatibility.
    arg_triangle : numpy.ndarray
        A 2D array of indices with shape (Ncube, Nmod) that maps each cube and 
        mode to the corresponding transformation in the coupling map.
    couplingMap : object
        An object containing the following attributes:
        - fluxtiptilt_2_data: A mapping matrix to transform flux-tip-tilt (FFT) 
            back to the data space.
        - data_2_fluxtiptilt: A mapping matrix to transform data into the 
            flux-tip-tilt (FFT) space.
    Returns:
    --------
    residual : numpy.ndarray
        The residual datacube after source removal. Its shape matches the 
        adjusted input datacube dimensions.
    fft_fit : numpy.ndarray
        The flux-tip-tilt (FFT) fit for the datacube. Its shape matches the 
        adjusted input datacube dimensions.
    Notes:
    ------
    - The function ensures that the input datacube is reshaped to 4D for 
        processing and reshaped back to its original dimensions before returning.
    - The residual is computed by subtracting the reconstructed data (from FFT) 
        from the original datacube.
    """
    
    size_cube = len(datacube.shape)
    if size_cube == 3:
        datacube = datacube[:, :, :, np.newaxis]
    if size_cube == 2:
        datacube = datacube[:, :, np.newaxis, np.newaxis]

    Nwave = datacube.shape[0]
    Noutput = datacube.shape[1]
    Ncube = datacube.shape[2]
    Nmod = datacube.shape[3]

    fluxtiptilt_2_data = couplingMap.fluxtiptilt_2_data
    data_2_fluxtiptilt = couplingMap.data_2_fluxtiptilt

    residual = datacube.copy()
    fft_fit = np.zeros((Nwave,3,Ncube,Nmod))
    for c in range(Ncube):
        for m in range(Nmod):
            i = arg_triangle[c,m]
            fft = np.matmul(data_2_fluxtiptilt[i],datacube[:,:,c,m,None]) #flux tip tilt
            fft_fit[:,:,c,m] = fft[:,:,0]
            residual[:,:,c,m] -= np.matmul(fluxtiptilt_2_data[i],fft)[:,:,0]
    
    if size_cube == 2:
        residual = residual[:,:,0,0]
        fft_fit = fft_fit[:,:,0,0]
    if size_cube == 3:
        residual = residual[:,:,:,0]
        fft_fit = fft_fit[:,:,:,0]
    
    return residual, fft_fit

def make_image_grid(couplingMap, Npixels, xmod=[0], ymod=[0]):
    """
    Generate a grid for image reconstruction based on the coupling map positions.
    This function creates a 2D grid for interpolation, which can be used to reconstruct
    an image from the coupling map data. The grid is defined based on the x and y 
    positions of the coupling map, with optional modifications using `xmod` and `ymod`.
    Parameters:
        couplingMap (object): An object containing `xpos` and `ypos` attributes, which 
                                represent the x and y positions of the coupling map.
        Npixels (int): The number of pixels along each dimension of the grid.
        xmod (float or array-like, optional): Modifications to apply to the x positions. 
                                                Defaults to 0.
        ymod (float or array-like, optional): Modifications to apply to the y positions. 
                                                Defaults to 0.
    Returns:
        tuple: A tuple containing two 2D arrays (`grid_x`, `grid_y`) representing the 
                x and y coordinates of the grid.
    """

    # Define the grid for interpolation
    # calcul de la grille de l'image que l'on souhaite reconstruire
    # if it is for a quick look of the real time display, use xmod=ymod=0
    xpos = couplingMap.xpos
    ypos = couplingMap.ypos

    # Define the grid for interpolation
    xmin, xmax   = np.min(xpos)-np.max(xmod), np.max(xpos)-np.min(xmod)
    ymin, ymax   = np.min(ypos)-np.max(ymod), np.max(ypos)-np.min(ymod)
    grid_x, grid_y = np.mgrid[xmin:xmax:Npixels*1j, ymin:ymax:Npixels*1j]

    return grid_x, grid_y


# Interpolate the fluxes onto the grid
def make_image_maps(datacube, couplingMap, grid_x, grid_y, xmod= [0], ymod= [0], wavelength = False):
    """
    Generate flux maps by interpolating fluxes from a datacube onto a specified grid.
    Parameters:
    -----------
    datacube : numpy.ndarray
        The input data cube containing flux values. It can have dimensions of 
        (Nwave, Noutput, Ncube, Nmod) or fewer, which will be reshaped accordingly.
    couplingMap : object
        An object containing the coupling map information. It must have the following attributes:
        - xpos : numpy.ndarray
            Array of x-positions for the coupling map.
        - ypos : numpy.ndarray
            Array of y-positions for the coupling map.
        - Npositions : int
            Number of positions in the coupling map.
        - data_2_flux : numpy.ndarray
            Transformation matrix to convert data to flux values.
    grid_x : numpy.ndarray
        The x-coordinates of the grid onto which the fluxes will be interpolated.
    grid_y : numpy.ndarray
        The y-coordinates of the grid onto which the fluxes will be interpolated.
    wavelength : bool, optional
        If False (default), the fluxes are averaged over all wavelengths. If True, 
        the fluxes are computed for each wavelength separately.
    xmod : list or numpy.ndarray, optional
        Modifications to apply to the x positions. Defaults to [0].
    ymod : list or numpy.ndarray, optional
        Modifications to apply to the y positions. Defaults to [0].
    Returns:
    --------
    flux_maps_sum : numpy.ndarray
        The summed flux maps across all modes. If the input datacube has 2 dimensions, 
        the output will be a single flux map.
    flux_maps : numpy.ndarray
        The flux maps for each cube, mode, and wavelength. The shape is 
        (Ncube, Nmod, Nwave, len(grid_x), len(grid_y)).
    Notes:
    ------
    - The function uses cubic interpolation to map the fluxes onto the specified grid.
    - If the input datacube has fewer than 4 dimensions, it is reshaped to ensure compatibility.
    - The function handles NaN values by summing flux maps with `np.nansum`.
    """

    # Interpolate the fluxes onto the grid
    # datacube_cleaned : 625, 38, 100
    # couplingMap : 625, 38, 100
    # grid_x, grid_y : 100, 100

    size_cube = len(datacube.shape)
    if size_cube == 3:
        datacube = datacube[:, :, :, np.newaxis]
    if size_cube == 2:
        datacube = datacube[:, :, np.newaxis, np.newaxis]

    Nwave = datacube.shape[0]
    Noutput = datacube.shape[1]
    Ncube = datacube.shape[2]
    Nmod = datacube.shape[3]

    xpos = couplingMap.xpos
    ypos = couplingMap.ypos
    Npositions = couplingMap.Npositions

    data_2_flux = couplingMap.data_2_flux

    fluxes = np.matmul(data_2_flux, datacube.reshape((Nwave,Noutput,Ncube*Nmod)))
    fluxes = fluxes.reshape((Nwave,Npositions,Ncube,Nmod))

    # developement mode, or quick look mode :
    if wavelength == False:
        fluxes = fluxes.mean(axis=0, keepdims=True)
        Nwave = 1

    flux_maps = []
    if wavelength == False:
        for c in range(Ncube):
            for m in range(Nmod):
                for w in range(Nwave):
                    # Interpolate the fluxes onto the grid
                    flux_map = griddata((xpos-xmod[m], ypos-ymod[m]), fluxes[w,:,c,m], (grid_x, grid_y), method='cubic')
                    flux_maps += [flux_map]
    else:
        for c in tqdm(range(Ncube)):
            for m in range(Nmod):
                for w in range(Nwave):
                    # Interpolate the fluxes onto the grid
                    flux_map = griddata((xpos-xmod[m], ypos-ymod[m]), fluxes[w,:,c,m], (grid_x, grid_y), method='cubic')
                    flux_maps += [flux_map]
    flux_maps = np.array(flux_maps).reshape((Ncube,Nmod,Nwave,len(grid_x),len(grid_y)))
    flux_maps_sum = np.nansum(flux_maps,axis=1)
    if size_cube == 2:
        flux_maps_sum = flux_maps_sum[0]
    
    return flux_maps_sum, fluxes

# Define a 2D Gaussian function
def gaussian_2d(xy, amplitude, xo, yo, sigma, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    w = 1/(sigma**2)
    g = offset + amplitude * np.exp(-(w*((x-xo)**2) + w*((y-yo)**2)))
    return g.ravel()
    
def fit_gaussian_on_flux(fluxes, xmod, ymod):
    """
    Fit a 2D Gaussian to the flux data.
    """
    # Interpolate the fluxes onto a grid
    # Create a grid of points for interpolation
    # Use the mean fluxes for the grid
    
    # Prepare data for fitting
    z = fluxes
    x = xmod
    y = ymod
    amplitude_0=np.max(fluxes)-np.min(fluxes)
    x_0= x[fluxes.argmax()]
    y_0= y[fluxes.argmax()]
    sigma_0 = (x.max()-x.min())/4
    offset_0=np.min(fluxes)

    # Initial guess for the parameters
    initial_guess = (amplitude_0,x_0,y_0,sigma_0,offset_0)

    # Fit the Gaussian
    popt, _ = curve_fit(gaussian_2d, (x, y), z, p0=initial_guess)

    return popt
