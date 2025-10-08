import torch
import numpy as np
import matplotlib.pyplot as plt

def fft_downsample_3d(volume, out_shape):
    """
    Downsamples a 3D real volume via Fourier cropping. More robust to flipping.
    Args:
        volume: torch.Tensor, shape (D, H, W), real dtype (float32/64)
        out_shape: tuple (D_new, H_new, W_new), desired output shape
    Returns:
        torch.Tensor, shape out_shape, real part only
    """
    D_old = volume.shape[0]
    D_new = out_shape[0]

    vol_f = torch.fft.fftn(volume)
    vol_f = torch.fft.fftshift(vol_f)

    in_shape = volume.shape
    out_f = torch.zeros(out_shape, dtype=vol_f.dtype, device=vol_f.device)
    slices_in = []
    slices_out = []
    for i, (n, m) in enumerate(zip(in_shape, out_shape)):
        c_in = n // 2
        c_out = m // 2
        start_in = c_in - c_out
        end_in = start_in + m
        start_out = 0
        end_out = m
        slices_in.append(slice(start_in, end_in))
        slices_out.append(slice(start_out, end_out))
    out_f[slices_out[0], slices_out[1], slices_out[2]] = vol_f[slices_in[0], slices_in[1], slices_in[2]]

    out_f = torch.fft.ifftshift(out_f)
    out_vol = torch.fft.ifftn(out_f) * (D_new / D_old)**3 # bringing the scale back to normal (that's because different normalizations will be applied due to the change of the total grid size..!)
    return out_vol.real


def fft_upsample_3d(volume: torch.Tensor, out_shape):
    """
    Upsamples a 3-D real volume via Fourier zero-padding (ideal sinc interpolation).

    Parameters
    ----------
    volume : torch.Tensor
        Real-valued tensor of shape (D, H, W) and dtype float32/64.
    out_shape : tuple
        Desired output shape (D_new, H_new, W_new); every entry must be
        >= the corresponding input size.

    Returns
    -------
    torch.Tensor
        Real-valued tensor of shape `out_shape`.
    """
    if any(m < n for m, n in zip(out_shape, volume.shape)):
        raise ValueError("All target sizes must be ≥ the input sizes.")

    # forward FFT → centre the zero–frequency component
    vol_f = torch.fft.fftn(volume)
    vol_f = torch.fft.fftshift(vol_f)

    in_shape = volume.shape
    out_f = torch.zeros(out_shape, dtype=vol_f.dtype, device=vol_f.device)

    # embed the low-frequency block from the input into the larger array
    slices_in, slices_out = [], []
    for n, m in zip(in_shape, out_shape):
        c_in  = n // 2                          # centre index of input
        c_out = m // 2                          # centre index of output
        start_out = c_out - c_in                # where to paste along this axis
        end_out   = start_out + n
        slices_in.append(slice(0, n))           # take the whole input axis
        slices_out.append(slice(start_out, end_out))

    out_f[slices_out[0], slices_out[1], slices_out[2]] = \
        vol_f[slices_in[0], slices_in[1], slices_in[2]]

    # back to spatial domain
    out_f  = torch.fft.ifftshift(out_f)
    out_vol = torch.fft.ifftn(out_f) * (out_shape[0] / in_shape[0])**3  # restore scale

    return out_vol.real

def apply_resolution_cutoff(vol, pixel_size, resolution_cutoff):
    """
    Applies a sharp resolution cutoff to a 3D map in Fourier space.

    vol: torch.Tensor (3D) - The input map.
    pixel_size: float - The voxel spacing in Ångstroms.
    resolution_cutoff: float - The resolution cutoff in Ångstroms (e.g., 3.5).
                               All information beyond this resolution will be removed.
    returns: torch.Tensor (3D) - The resolution-filtered map.
    """
    # 1. Get the frequency grid for the volume
    _, _, s2_full = compute_freqs(vol, pixel_size)
    
    # 2. Calculate the cutoff threshold in squared frequency space
    # The resolution is in Å, frequency s is in 1/Å.
    # We work with s^2 for convenience.
    cutoff_s2 = 1.0 / (resolution_cutoff**2)
    
    # 3. Create the binary mask
    # The mask is 1 inside the resolution sphere and 0 outside.
    mask = (s2_full <= cutoff_s2).to(vol.dtype)
    
    # 4. Apply the mask in Fourier space
    # FFT the volume
    fft_vol = torch.fft.fftshift(torch.fft.fftn(vol))
    
    # Multiply the FFT by the mask to remove high frequencies
    fft_filtered = fft_vol * mask
    
    # 5. Inverse FFT to get the filtered volume back in real space
    filtered_vol = torch.fft.ifftn(torch.fft.ifftshift(fft_filtered)).real
    
    return filtered_vol



def compute_freqs(vol, pixel_size):
    # Perform FFT and calculate amplitude
    fft_data = torch.fft.fftshift(torch.fft.fftn(vol))
    amplitude = fft_data.abs()
    
    # Freqs
    freqs_flat = [torch.fft.fftfreq(shape, d=pixel_size) for shape in vol.shape]
    freqs = torch.meshgrid(*freqs_flat, indexing='ij')
    freqs = [torch.fft.fftshift(f) for f in freqs]

    # Calculate s² values
    s2_full  = (freqs[0]**2 + freqs[1]**2 + freqs[2]**2)

    return freqs, amplitude, s2_full


def apply_bfactor_to_map(vol, pixel_size, B_blur, device):
    """
    vol: torch.Tensor (3D) - input map
    pixel_size: float - voxel spacing in Å
    B_blur: float - positive B-factor to apply (Å²)
    returns: blurred torch.Tensor (3D)
    """
    N = vol.shape
    # FFT
    fft_vol = torch.fft.fftshift(torch.fft.fftn(vol))
    
    # Frequencies grid
    freqs, amplitude, s2_full = compute_freqs(vol, pixel_size)
    
    # Apply B-factor attenuation in Fourier space
    s2_full = s2_full.to(device)
    attenuation = torch.exp(- (B_blur / 4.0) * s2_full).to(device)
    fft_blur = fft_vol * attenuation
    
    # Inverse FFT
    fft_blur = torch.fft.ifftshift(fft_blur)
    blurred = torch.fft.ifftn(fft_blur).real
    return blurred



def plot_power_spectrum(image_data):
    """
    Calculates and plots the power spectrum of an image.

    Args:
        image_data: A 2D NumPy array representing the image.
    """

    # 1. Transform the image to the frequency domain
    f_image = np.fft.fft2(image_data)

    # 2. Calculate the magnitude (power)
    power_spectrum = np.abs(f_image)

    # 3. Shift the spectrum to center it (optional, but recommended)
    power_spectrum = np.fft.fftshift(power_spectrum)

    # 4. Calculate radial average for plotting (optional, but recommended)
    rows, cols = image_data.shape
    cx, cy = cols // 2, rows // 2
    radius = np.arange(np.max((cols, rows)) // 2 + 1)
    radial_power = np.zeros_like(radius)
    for i in range(len(radius)):
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        radial_power[i] = np.mean(power_spectrum[np.where(r == i)])

    # 5. Plot the radial average (power spectrum vs frequency)
    plt.plot(radius, np.log(radial_power + 1e-6))  # Add a small value to avoid log(0)
    plt.xlabel("Frequency (normalized)")
    plt.ylabel("Log Power")
    plt.title("Power Spectrum of Image")
    plt.savefig("signal.png")

def plot_power_spectrum_3d(volume, voxel_size=0.97, resolution_cutoff=4.1, save_path="volume_power_spectrum.png"):
    """
    Calculates and plots the radial power spectrum of a 3D volume with a resolution cutoff indicator.

    Args:
        volume: 3D NumPy array representing the volume.
        voxel_size: Voxel size in Ångströms.
        resolution_cutoff: Reported resolution in Å (default: 4.1 Å).
        save_path: Path to save the plot.
    """
    # 1. FFT of the volume
    fft_vol = np.fft.fftn(volume)
    fft_vol = np.fft.fftshift(fft_vol)

    # 2. Power spectrum
    power_spectrum = np.abs(fft_vol) ** 2

    # 3. Radius grid
    z, y, x = np.indices(volume.shape)
    cz, cy, cx = np.array(volume.shape) // 2
    r = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2).astype(np.int32)

    # 4. Radial average
    max_r = np.max(r)
    radial_power = np.bincount(r.ravel(), weights=power_spectrum.ravel())
    counts = np.bincount(r.ravel())
    radial_power /= (counts + 1e-8)

    # 5. Frequency axis (in voxel⁻¹)
    freqs = np.arange(len(radial_power))

    # 6. Compute cutoff frequency in voxel⁻¹
    freq_ang = 1.0 / resolution_cutoff
    freq_voxel = freq_ang * voxel_size
    cutoff_index = freq_voxel * volume.shape[0]  # assuming cubic volume

    # 7. Plot
    plt.figure()
    plt.plot(freqs, np.log(radial_power + 1e-8), label="Log Power")
    plt.axvline(x=cutoff_index, color='r', linestyle='--', label=f"{resolution_cutoff} Å cutoff")
    plt.xlabel("Spatial Frequency (voxel⁻¹)")
    plt.ylabel("Log Power")
    plt.title("Radial Power Spectrum of Volume")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
# A vizualisation of these apparent differences..! Which are just noise and generally speaking not as nice..!
# fig, ax = plt.subplots(nrows=2, ncols=1); ax[0].imshow(test_lowpass.sum(0), cmap="gray"); ax[1].imshow(self.fo.sum(0).cpu().numpy(), cmap="gray"); plt.savefig("test.png")
# fig, ax = plt.subplots(); ax.imshow(np.abs(self.fo.sum(0).cpu().numpy() - test_lowpass).sum(0), cmap="gray"); plt.savefig("test.png")

def lowpass_filter_volume(volume, voxel_size=0.97, resolution_cutoff=4.1):
    """
    Removes signal beyond a given resolution from a 3D volume via low-pass filtering.

    Args:
        volume: 3D NumPy array (real space).
        voxel_size: Size of one voxel in Ångströms.
        resolution_cutoff: Cutoff in Ångströms (e.g. 4.1 Å).

    Returns:
        filtered_volume: 3D NumPy array after low-pass filtering.
    """
    # 1. FFT of the volume (no shift!)
    fft_vol = np.fft.fftn(volume)

    # 2. Frequency grid in Å⁻¹
    shape = volume.shape
    nz, ny, nx = shape

    z = np.fft.fftfreq(nz, d=voxel_size)
    y = np.fft.fftfreq(ny, d=voxel_size)
    x = np.fft.fftfreq(nx, d=voxel_size)
    zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')

    freq_magnitude = np.sqrt(xx**2 + yy**2 + zz**2)

    # 3. Build low-pass mask
    cutoff_freq = 1.0 / resolution_cutoff
    mask = freq_magnitude <= cutoff_freq

    # 4. Apply mask
    fft_filtered = fft_vol * mask

    # 5. Inverse FFT
    filtered_volume = np.fft.ifftn(fft_filtered).real

    return filtered_volume
