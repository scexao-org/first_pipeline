#%%
import os
import numpy as np
from astropy.io import fits
import glob
import matplotlib
# if ("VSCODE_PID" in os.environ or os.environ.get('TERM_PROGRAM') == 'vscode'):
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot,hist,clf,figure,legend,imshow,clf
plt.ion()

directory="/Users/slacour/DATA/LANTERNE/darks"

# Find all FITS files in directory
fits_files = glob.glob(os.path.join(directory, "*.fits"))
fits_files.extend(glob.glob(os.path.join(directory, "*.fit")))


directory="/Users/slacour/DATA/LANTERNE/20250514/preproc/"
# Find all FITS files in directory
fits_files = glob.glob(os.path.join(directory, "*BETA*.fits"))

directory="/Users/slacour/DATA/LANTERNE/raw/20251119/firstpl/calib_wavelength/"
# Find all FITS files in directory
fits_files = glob.glob(os.path.join(directory, "*.fits"))

directory="/Users/slacour/DATA/LANTERNE/20251121/firstpl/"
# Find all FITS files in directory
fits_files = glob.glob(os.path.join(directory, "firstpl_1*.fits"))

# Sort files by name for consistent ordering
fits_files.sort()
# Check for EXPTIME keyword consistency
# fits_files.pop(6)
# fits_files.pop(4)

if not fits_files:
        print("No FITS files found in directory")

print(f"Found {len(fits_files)} FITS files")
print("-" * 60)

all_means = []
all_stds = []
all_exptimes = []

for fits_file in fits_files:
        try:
                with fits.open(fits_file) as hdul:
                        # Get primary image data
                        data = hdul[0].data
                        header = hdul[0].header
                        if header.get('NAXIS3', None) > 5:
                                exptime = header.get('EXPTIME', None)
                                
                                all_exptimes.append(exptime)

                                if data is None:
                                        print(f"{os.path.basename(fits_file)}: No image data")
                                        continue
                                
                                # Calculate statistics
                                mean_val = np.mean(data)
                                noise_val = np.std(data)
                                
                                all_means.append(mean_val)
                                all_stds.append(noise_val)
                                
                                print(f"{os.path.basename(fits_file)}:")
                                print(f"  Shape: {data.shape}")
                                print(f"  EXPTIME: {exptime}")
                                print(f"  Mean: {mean_val:.3f}")
                                print(f"  Noise (std): {noise_val:.3f}")
                                print(f"  Min/Max: {np.min(data):.3f} / {np.max(data):.3f}")
                        
        except Exception as e:
                print(f"Error reading {fits_file}: {e}")

#%%

# Overall statistics
if all_means:
        print("-" * 60)
        print("OVERALL STATISTICS:")
        print(f"Mean of means: {np.mean(all_means):.3f}")
        print(f"Mean of noise values: {np.mean(all_stds):.3f}")
        print(f"Range of means: {np.min(all_means):.3f} to {np.max(all_means):.3f}")
        print(f"Range of noise: {np.min(all_stds):.3f} to {np.max(all_stds):.3f}")

# Create plot with means and variance vs exptime
if all_means and all_exptimes:
        # Calculate variance from standard deviations
        all_vars = [std**2 for std in all_stds]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot means
        color = 'tab:blue'
        ax1.set_xlabel('Exposure Time (s)')
        ax1.set_ylabel('Mean Value', color=color)
        ax1.scatter(all_exptimes, np.array(all_means)-200, color=color, alpha=0.7, label='Mean')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='lower right')
        
        # Create second y-axis for variance
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Variance', color=color)
        ax2.scatter(all_exptimes, all_vars, color=color, alpha=0.7, label='Variance')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Dark Frame Statistics vs Exposure Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.legend()
        plt.show()
        # Set y-axis limits to start from 0 for         both subplots
        ax1.set_ylim(0, ax1.get_ylim()[1])
        ax2.set_ylim(0, ax2.get_ylim()[1])

        # plt.savefig("dark_frame_statistics.png")
# %%

n_image= 2
image=fits.getdata(fits_files[n_image]).mean(axis=0)
image_std=fits.getdata(fits_files[n_image]).std(axis=0)
# Plot both images side by side
fig, (ax1, ax2) = plt.subplots( 2, 1, num="Focus", figsize=(14, 8),clear=True)

# Calculate percentiles for proper scaling
vmin_mean, vmax_mean = np.percentile(image, [1, 99])
vmin_std, vmax_std = np.percentile(image_std, [1, 99])
# vmin_mean, vmax_mean= 204.25694444444446,230.10062499999

# Plot mean image
im1 = ax1.imshow(image, cmap='viridis', vmin=vmin_mean, vmax=vmax_mean, aspect='auto', interpolation='none')
ax1.set_title(f'Flux Mean ({all_exptimes[n_image]}s exptime)')
ax1.set_xlabel('X pixel')
ax1.set_ylabel('Y pixel')
plt.colorbar(im1, ax=ax1)

# Plot standard deviation image
im2 = ax2.imshow(image_std, cmap='viridis', vmin=vmin_std, vmax=vmax_std, aspect='auto')
ax2.set_title('Flux Standard Deviation')
ax2.set_xlabel('X pixel')
ax2.set_ylabel('Y pixel')
plt.colorbar(im2, ax=ax2)
plt.tight_layout()
plt.show()
plt.savefig("super_K.png")

# %%

image_5=fits.getdata(fits_files[-2]).mean(axis=0)
image_5_std=fits.getdata(fits_files[-2]).std(axis=0)
# Plot both images side by side
fig, (ax1, ax2) = plt.subplots( 2, 1, num="Dark Frame Statistics File 0.036", figsize=(14, 8),clear=True)

# Calculate percentiles for proper scaling
vmin_mean_5, vmax_mean_5 = np.percentile(image_5, [1, 99])
vmin_std_5, vmax_std_5 = np.percentile(image_5_std, [1, 99])

# Plot mean image
im1 = ax1.imshow(image_5, cmap='viridis', vmin=vmin_mean_5, vmax=vmax_mean_5)
ax1.set_title('Dark Frame Mean - File 5')
ax1.set_xlabel('X pixel')
ax1.set_ylabel('Y pixel')
plt.colorbar(im1, ax=ax1)

# Plot standard deviation image
im2 = ax2.imshow(image_5_std, cmap='viridis', vmin=vmin_std_5, vmax=vmax_std_5)
ax2.set_title('Dark Frame Standard Deviation - File 5')
ax2.set_xlabel('X pixel')
ax2.set_ylabel('Y pixel')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.show()
# %%



R = np.linspace(0,11,100) 
snr = 10**(0.4*(11-R))

astrometric_accuracy_10s = 20/np.sqrt(snr)
astrometric_accuracy_10s[astrometric_accuracy_10s<0.05]=0.05
astrometric_accuracy_30m = 20/np.sqrt(snr)/np.sqrt(30*6)
astrometric_accuracy_30m[astrometric_accuracy_30m<0.05]=0.05

contrast_range_10s = 1/snr
contrast_range_30m = 1/snr/np.sqrt(30*6)
contrast_range_10s[contrast_range_10s<8e-4]=8e-4
contrast_range_30m[contrast_range_30m<8e-4]=8e-4


#%%

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot astrometric accuracy
color = 'tab:blue'
ax1.set_xlabel('R magnitude')
ax1.set_ylabel('Astrometric Accuracy (mas)', color=color)
ax1.plot(R, astrometric_accuracy_10s, "--", color=color, linewidth=2, label='Astrometric Accuracy in 10 sec')
ax1.plot(R, astrometric_accuracy_30m, color=color, linewidth=2, label='Astrometric Accuracy in 30 min')
ax1.plot(2.9,5e-2,'*',color='blue',markersize=12,label='Beta CMI (demonstrated)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')


# Create second y-axis for contrast range
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Contrast Range (@100 mas)', color=color)
ax2.plot(R, contrast_range_10s, "--", color=color, linewidth=2, label='Contrast Range in 10 sec')
ax2.plot(R, contrast_range_30m, color=color, linewidth=2, label='Contrast Range in 30 min')
ax2.plot(6.0,1e-2,'*',color='red',markersize=12,label='HIP81126 AB (demonstrated)')
# ax2.plot(6.0,9e-4,'o',color='red',markersize=6,label='HIP81126 off-axis (120mas)')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_yscale('log')
ax2.legend(loc='lower right')
ax2.set_ylim(7e-4, 1)
ax2.grid(True, alpha=0.3)

plt.title('Astrometric Accuracy and Contrast Range vs R Magnitude')
plt.tight_layout()
plt.show()
plt.savefig("astrometric_contrast_vs_Rmag.png")
# %%

mag_K=np.linspace(0,12,1000)
saturation = 0.5*10**((mag_K-3)/2.5)

DIT = np.logspace(-4,1,1000)
readout_flux_star_equivalent = DIT*10*1500
readout_flux_star_equivalent[DIT>0.15]*=.4
mag_readout=np.log10(readout_flux_star_equivalent)/0.4

mag_recommanded = mag_readout-1.5
for i in range(len(DIT)):
        if mag_recommanded[i]>mag_recommanded[i:].min():
                mag_recommanded[i]=mag_recommanded[i:].min()

mag_recommanded[DIT>4]=mag_readout[DIT>4]

DIT_mag_recommanded = DIT.copy()
DIT_mag_recommanded[DIT_mag_recommanded<0.03]=0.03
# mag_recommanded[DIT<0.03]=0

DIT_mag_recommanded_upper = DIT_mag_recommanded  * 2.5

saturatrion_mag_recommanded =0.5*10**((mag_recommanded-3)/2.5)
DIT_mag_recommanded_upper[DIT_mag_recommanded_upper>saturatrion_mag_recommanded]=saturatrion_mag_recommanded[DIT_mag_recommanded_upper>saturatrion_mag_recommanded]

# DIT_recommanded[DIT_recommanded<0.01]=0.01

plt.clf()
plt.plot(mag_K,saturation,'r',label='Saturation limit',zorder=5)
plt.plot(mag_readout,DIT,'C0-',label='Star shot noise equivalent to readout noise',zorder=10)
plt.plot(mag_recommanded,DIT_mag_recommanded,'C1-')
plt.plot(mag_recommanded,DIT_mag_recommanded_upper,'C1-')
plt.fill_between(mag_recommanded,DIT_mag_recommanded, DIT_mag_recommanded_upper, color='purple', alpha=0.2, label='recommended DIT range')
# plt.fill_between(mag_recommanded2,DIT, DIT_recommanded, color='purple', alpha = 0.1)

plt.plot([0,12],[0.15,0.15],'k--',linewidth=0.8,alpha=0.5)
plt.text(9,0.12*1.5,"Low noise readout",fontsize=10,alpha=0.9)
plt.text(9,0.06*1.5,"Fast readout mode",fontsize=10,alpha=0.9)
plt.yscale('log')
plt.xlabel("R Magnitude of the star")
plt.ylabel("DIT (s)")
plt.title("Recommended DIT vs Star Magnitude")
plt.grid(True,alpha=0.3)
plt.legend(loc='upper left')
plt.axhline(y=0.03, color='gray', linestyle=':', alpha=0.7, linewidth=1)
# plt.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7, linewidth=1)
plt.axhline(y=.3, color='gray', linestyle=':', alpha=0.7, linewidth=1)
plt.axhline(y=3.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)
plt.ylim(1e-2,10)
plt.xlim(0,12)
plt.tight_layout()
plt.savefig("recommended_DIT_vs_Rmag.png")

# %%

import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt

# Define t
t = np.logspace(-4, 1, 1000)

def get_magnitude(t_values,snr):
      
        def equation(m, t):
                y = 10**(0.4*(13 - m))
                if t<0.15:
                        return (y*t)**2 - snr**2*(y*t + 0.4**2 + 0.03*t)
                else:
                        return (y*t)**2 - snr**2*(y*t + 0.25**2 + 0.03*t)

        # Compute m(t)
        m_values = []
        for ti in t_values:
                # Solve for m, using a reasonable bracket (adjust if needed)
                sol = root_scalar(equation, args=(ti,), bracket=[-10, 30], method='brentq')
                m_values.append(sol.root)

        m_values = np.array(m_values)
        return m_values



# Compute m(t)
m_values_5 = get_magnitude(t, snr=5)
m_values_10 = get_magnitude(t, snr=10)
m_values_20 = get_magnitude(t, snr=20)


mag_K=np.linspace(0,12,1000)
saturation = 0.5*10**((mag_K-3)/2.5)
minimum_DIT = 0.03

tm = t > minimum_DIT/100
# Plot
plt.clf()
plt.plot([0,12],[0.15,0.15],'k--',linewidth=0.8,alpha=0.5)
plt.text(2.3,0.12*1.5,"Low noise readout",fontsize=10,alpha=0.9)
plt.text(2.3,0.073*1.5,"Fast readout mode",fontsize=10,alpha=0.9)
plt.plot(mag_K,saturation,'r',label='Saturation limit',zorder=5)
plt.plot(mag_K,minimum_DIT*np.ones_like(mag_K),'C1',label='Piezo speed limit',zorder=-5)
plt.plot(m_values_5,t,'gray', label='SNR of 5')
plt.plot(m_values_10[tm],t[tm],"C2", label='SNR of 10')
# plt.plot([0,5.7],(minimum_DIT+0.0001)*np.ones(2),"C2")

mask_below_minimum = t < minimum_DIT
plt.fill_between(m_values_5[mask_below_minimum], t[mask_below_minimum], minimum_DIT, color='C1', alpha=0.2)

plt.fill_between([0,m_values_5[0]],0,minimum_DIT*np.ones_like(2), color='C1', alpha=0.2)
plt.fill_betweenx(t[tm], m_values_5[tm], m_values_10[tm], color='green', alpha=0.3, label='Recommended DIT',zorder=-15)
plt.fill_between(mag_K, saturation, 10, color='r', alpha=0.2)
plt.fill_between(m_values_5, 0, t, color='gray', alpha=0.2)


plt.axhline(y=0.03, color='gray', linestyle=':', alpha=0.7, linewidth=1)
plt.axhline(y=.3, color='gray', linestyle=':', alpha=0.7, linewidth=1)
plt.axhline(y=3.0, color='gray', linestyle=':', alpha=0.7, linewidth=1)

plt.text(0.7,1.2,"Detector saturated",fontsize=10,alpha=0.9)
plt.text(2.3,0.02,"Piezo modulation not working",fontsize=10,alpha=0.9)
plt.text(8.1,0.055,"Not enough signal",fontsize=10,alpha=0.9)

plt.xlabel("R Magnitude of the star")
plt.ylabel("DIT (s)")
plt.title("Recommended DIT vs Star Magnitude")
plt.grid(True,alpha=0.3)
plt.xlim(0,12)
plt.ylim(0.005,10)
plt.legend()
plt.yscale('log')
plt.savefig("recommended_DIT_vs_Rmag.png")

# %%
