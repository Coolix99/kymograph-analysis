import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from powersmooth.powersmooth import powersmooth_general, upsample_with_mask
from scipy.signal import hilbert
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d


WINDOW_LENGTH = 20
TF=0.0 #movement detection thresold factor
CILIA_COL='Cilia_EndPoint_Y_um'
ACTUATOR_COL='Actuator_ymin_um'
TIME_COL='Time_s'
HILBERT_REMOVE=20




def detect_actuator_activity_segments(df: pd.DataFrame,
                                      window_size: int = 5,
                                      debug: bool = False
                                     ):
    """
    Detect continuous low- and high-activity segments in actuator movement
    based on standard deviation threshold from bimodal std histogram.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ACTUATOR_COL, TIME_COL, and CILIA_COL.
    window_size : int
        Number of frames per sliding window (must be < number of rows).
    debug : bool
        If True, display diagnostic plot.

    Returns
    -------
    segments : List[Dict]
        Each dict has keys:
        - 'type' : 'low' or 'high'
        - 'start_time', 'end_time' : float
        - 'cilia_times', 'cilia_values' : np.ndarray
        - 'actuator_values' : np.ndarray (only for 'high' segments)
    """
    actuator = df[ACTUATOR_COL].to_numpy()
    times = df[TIME_COL].to_numpy()
    cilia = df[CILIA_COL].to_numpy()

    n_t = len(actuator)
    if window_size >= n_t:
        raise ValueError(f"window_size ({window_size}) must be smaller than total frames ({n_t})")

    # Compute sliding std dev
    std_per_window = np.std(sliding_window_view(actuator, window_shape=window_size), axis=1)

    # Estimate threshold via KDE valley detection
    kde = gaussian_kde(std_per_window)
    std_range = np.linspace(std_per_window.min(), std_per_window.max(), 1000)
    density = kde(std_range)

    # Find local minima of KDE
    minima_idx = argrelextrema(density, np.less)[0]
    maxima_idx = argrelextrema(density, np.greater)[0]

    if len(minima_idx) == 0 or len(maxima_idx) < 2:
        raise RuntimeError("Could not detect bimodal distribution in std values.")

    # Use minimum between the two largest maxima as threshold
    peak_vals = density[maxima_idx]
    top_two = np.argsort(peak_vals)[-2:]
    peak_locs = np.sort(std_range[maxima_idx[top_two]])
    thresh_candidates = std_range[minima_idx]
    thresh = thresh_candidates[(thresh_candidates > peak_locs[0]) & (thresh_candidates < peak_locs[1])]

    if len(thresh) == 0:
        raise RuntimeError("No threshold found between KDE peaks.")
    thresh = thresh[0]

    # Classify windows
    is_active = std_per_window > thresh

    # Identify contiguous segments
    segments = []
    curr_type = is_active[0]
    start_idx = 0

    for i in range(1, len(is_active)):
        if is_active[i] != curr_type:
            end_idx = i
            s = {
                'type': 'high' if curr_type else 'low',
                'start_time': times[start_idx],
                'end_time': times[end_idx + window_size - 1] if (end_idx + window_size - 1 < len(times)) else times[-1],
                'cilia_times': times[start_idx + window_size // 2:end_idx + window_size // 2],
                'cilia_values': cilia[start_idx + window_size // 2:end_idx + window_size // 2],
            }
            if curr_type:  # Only add actuator for active segments
                s['actuator_values'] = actuator[start_idx + window_size // 2:end_idx + window_size // 2]
            segments.append(s)
            start_idx = i
            curr_type = is_active[i]

    # Add last segment
    end_idx = len(is_active)
    s = {
        'type': 'high' if curr_type else 'low',
        'start_time': times[start_idx],
        'end_time': times[-1],
        'cilia_times': times[start_idx + window_size // 2:],
        'cilia_values': cilia[start_idx + window_size // 2:],
    }
    if curr_type:
        s['actuator_values'] = actuator[start_idx + window_size // 2:]
    segments.append(s)

    # --- Debug plot ---
    if debug:
        plt.figure(figsize=(10, 4))
        plt.plot(times[:len(std_per_window)], std_per_window, label='Sliding STD', color='gray')
        plt.axhline(thresh, color='red', linestyle='--', label=f'Threshold = {thresh:.3g}')
        for seg in segments:
            t0, t1 = seg['start_time'], seg['end_time']
            color = 'green' if seg['type'] == 'low' else 'orange'
            plt.axvspan(t0, t1, color=color, alpha=0.3)
        plt.xlabel('Time (s)')
        plt.ylabel('STD (windowed)')
        plt.title('Detected Activity Segments')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return segments


def plot_minima_spacing_histogram_kde(t_min, bins=30, bandwidth='scott', debug=False):
    """
    Plot histogram and KDE of spacing between local minima, and determine the peak spacing.

    Parameters
    ----------
    t_min : array-like
        Times of local minima (must be sorted).
    bins : int
        Number of histogram bins.
    bandwidth : str or float
        Bandwidth method for KDE ('scott', 'silverman', or float).

    Returns
    -------
    spacings : np.ndarray
        Array of spacing values.
    peak_spacing : float
        Location of the peak of the KDE (most common spacing).
    """
    t_min = np.sort(np.asarray(t_min))
    spacings = np.diff(t_min)

    if len(spacings) < 2:
        print("Not enough minima to estimate spacing.")
        return spacings, np.nan

    # Fit KDE
    kde = gaussian_kde(spacings, bw_method=bandwidth)
    spacing_range = np.linspace(spacings.min(), spacings.max(), 1000)
    density = kde(spacing_range)

    # Find peak of KDE
    peak_index = np.argmax(density)
    peak_spacing = spacing_range[peak_index]
    if debug:
        # Plot
        plt.figure(figsize=(7, 4))
        plt.hist(spacings, bins=bins, density=True, alpha=0.5, edgecolor='black', label='Histogram')
        plt.plot(spacing_range, density, label='KDE', linewidth=2)
        plt.axvline(peak_spacing, color='r', linestyle='--', label=f'Peak = {peak_spacing:.3f}s')
        plt.xlabel("Time between Minima (s)")
        plt.ylabel("Density")
        plt.title("Spacing Between Local Minima")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return spacings, peak_spacing

def process_signal(t,v,flat_tol=0.001,res_spacing_tol=0.35,debug=False,smooth_weight=1e-16):
    t = np.asarray(t)
    v = np.asarray(v)

    minima_t = []
    minima_v = []

    i = 1
    while i < len(v) - 1:
        # Case 1: standard local minimum
        if v[i - 1] > v[i] < v[i + 1]:
            minima_t.append(t[i])
            minima_v.append(v[i])
            i += 1
        # Case 2: flat local minimum
        elif v[i - 1] > v[i] and np.isclose(v[i], v[i + 1], rtol=flat_tol):
            start = i
            while i + 1 < len(v) and np.isclose(v[i], v[i + 1], rtol=flat_tol):
                i += 1
            end = i
            if end + 1 < len(v) and v[end + 1] > v[end]:
                flat_t = t[start:end + 1].mean()
                flat_v = v[start:end + 1].mean()
                minima_t.append(flat_t)
                minima_v.append(flat_v)
            i += 1
        else:
            i += 1


    _,peak_spacing=plot_minima_spacing_histogram_kde(minima_t)

    spacing_tol = res_spacing_tol * peak_spacing
    window_size = 7

    minima_t = np.array(minima_t)
    minima_v = np.array(minima_v)

    filtered_t = []
    filtered_v = []

    n_minima = len(minima_t)

    for i in range(n_minima):
        # Define the window: centered at i (or as close as possible)
        half = window_size // 2
        start = max(0, i - half)
        end = min(n_minima, start + window_size)
        start = max(0, end - window_size)  # adjust in case of early boundary

        # Skip if the window is too small
        if end - start < window_size:
    
            continue

        # Compute local offset and phase indices
        window_t = np.concatenate((minima_t[start:i], minima_t[i+1:end]))
        test_offsets = np.linspace(0, peak_spacing, 10, endpoint=False)
        best_offset = None
        min_residual = np.inf

        for test_offset in test_offsets:
            window_n = np.round((window_t - test_offset) / peak_spacing)
            t_theory = window_n * peak_spacing + test_offset
            residuals = np.abs(window_t - t_theory)
            mean_res = np.mean(residuals)

            if mean_res < min_residual:
                min_residual = mean_res
                best_offset = test_offset

        # Use best offset
        window_offset = best_offset
        

        # plt.figure(figsize=(8, 3))
        # plt.scatter(window_t, np.concatenate((minima_v[start:i], minima_v[i+1:end])), color='blue', label='Local Minima')

        # # Plot vertical lines at theoretical times
        # for j in range(window_n.shape[0]):
        #     n_i = np.round((window_t[j] - window_offset) / peak_spacing)
        #     t_theo = n_i * peak_spacing + window_offset
        #     plt.axvline(t_theo, color='black', linestyle='--', linewidth=0.7)

        # plt.title(f"Sliding Window {start}–{end}")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Signal")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # Check middle point only
        t_i = minima_t[i]
        v_i = minima_v[i]
        n_i = np.round((t_i - window_offset) / peak_spacing)
        t_theory = n_i * peak_spacing + window_offset

        if abs(t_i - t_theory) <= spacing_tol:
            # Only append if not already added (avoids duplicates at window overlaps)
            if len(filtered_t) == 0 or t_i != filtered_t[-1]:
                filtered_t.append(t_i)
                filtered_v.append(v_i)


    filtered_t = np.array(filtered_t)
    filtered_v = np.array(filtered_v)

    if debug:
        # --- Plot signal and detected minima ---
        plt.figure(figsize=(10, 3))
        plt.plot(t, v, label='Signal')
        plt.scatter(minima_t, minima_v, color='red', label=' Minima')
        plt.scatter(filtered_t, filtered_v, color='green', label='Filtered Minima')
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.title("Signal with Local Minima (Filtered)")
        plt.legend()
        plt.tight_layout()
        plt.show()


    final_t = []
    final_v = []
    final_phase = []

    phase_val = 0.0  # start at 0

    for i in range(len(filtered_t) - 1):
        t_curr = filtered_t[i]
        t_next = filtered_t[i + 1]
        v_curr = filtered_v[i]
        v_next = filtered_v[i + 1]

        dt = t_next - t_curr
        n_steps = int(np.round(dt / peak_spacing))

        if n_steps <= 0:
            UserWarning("Non-positive phase step detected. Check input order or peak_spacing.")
            continue

        # Add current point
        final_t.append(t_curr)
        final_v.append(v_curr)
        final_phase.append(phase_val)

        # Interpolate if gap > 1 cycle
        for j in range(1, n_steps):
            alpha = j / n_steps
            t_interp = t_curr + alpha * dt
            v_interp = v_curr + alpha * (v_next - v_curr)
            phase_interp = phase_val + 2 * np.pi * j
            final_t.append(t_interp)
            final_v.append(v_interp)
            final_phase.append(phase_interp)

        # Step to next phase value
        phase_val += 2 * np.pi * n_steps

    # Add final point
    final_t.append(filtered_t[-1])
    final_v.append(filtered_v[-1])
    final_phase.append(phase_val)

    final_t = np.array(final_t)
    final_v = np.array(final_v)
    phase_raw = np.array(final_phase)

    if debug:
        # --- Plot signal with original filtered and filled minima ---
        plt.figure(figsize=(10, 3))
        plt.plot(t, v, label='Signal')
        plt.scatter(filtered_t, filtered_v, color='green', label='Filtered Minima')
        plt.scatter(final_t, final_v, color='orange', marker='x', label='Filled (Interpolated)')
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.title("Signal with Filtered and Interpolated Minima")
        plt.legend()
        plt.tight_layout()
        plt.show()

    n_phase = np.arange(len(final_t))
    phase_raw = 2 * np.pi * n_phase  # Arbitrary phase assignment (e.g. 2π steps)

    # --- Upsample phase signal for smoothing ---
    target_dx = 0.1 * peak_spacing  # desired resolution
    t_dense, phase_sparse, mask = upsample_with_mask(final_t, phase_raw, dx=target_dx)

    # --- Apply powersmooth ---
    phase_smooth = powersmooth_general(t_dense, phase_sparse, weights={2: smooth_weight, 3: 0e-9}, mask=mask)

    if debug:
        # --- Plot raw vs smoothed phase ---
        plt.figure(figsize=(10, 3))
        plt.plot(final_t, phase_raw, 'o', label='Raw Phase (2π steps)')
        plt.plot(t_dense, phase_smooth, '-', label='Smoothed Phase')
        plt.xlabel("Time (s)")
        plt.ylabel("Phase (rad)")
        plt.title("Phase vs. Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Remove linear trend for clarity ---
    t0 = t_dense[0]
    phi0 = phase_smooth[0]
    phi1 = phase_smooth[-1]
    trend = phi0 + (t_dense - t0) * (phi1 - phi0) / (t_dense[-1] - t0)
    detrended_phase = phase_smooth - trend
    if debug:
        plt.figure(figsize=(10, 3))
        plt.plot(t_dense, detrended_phase, label='Detrended Phase')
        plt.xlabel("Time (s)")
        plt.ylabel("Detrended Phase (rad)")
        plt.title("Phase(t) − Linear Trend")
        plt.axhline(0, color='gray', linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'time':t_dense,
        'phase': phase_smooth,
        'final_t':final_t,
        'final_v':final_v,
        't':t,
        'v':v
    }

def fourier_denoise(signal, dt, f_low=4.0, f_high=1000.0, f_rel_low=0.3, f_rel_high=5.0, debug=False):
    """
    Denoise a 1D signal using FFT bandpass filtering around the peak frequency.

    Parameters
    ----------
    signal : ndarray, shape (n_t,)
        1D signal to be denoised.
    dt : float
        Time step between samples.
    f_low : float
        Lower absolute frequency bound (Hz).
    f_high : float
        Upper absolute frequency bound (Hz).
    f_rel_low : float
        Lower bound relative to peak frequency.
    f_rel_high : float
        Upper bound relative to peak frequency.
    power_cut : float
        Relative threshold to mask frequency components.
    debug : bool
        If True, plot PSD and signal before/after denoising.

    Returns
    -------
    denoised : ndarray
        Full reconstructed signal.
    denoised_removed : ndarray
        Signal with stripes (low-power components) removed.
    peak_freq : float
        Detected peak frequency.
    """
    # FFT along time axis
    fft_sig = np.fft.fft(signal, axis=0)
    n_t = signal.shape[0]

    # Frequencies and amplitudes
    freq = np.fft.fftfreq(n_t, d=dt)
    amps = np.abs(fft_sig)

    # PSD
    psd = amps**2

    # Initial coarse band mask
    coarse_band = (np.abs(freq) >= f_low) & (np.abs(freq) <= f_high)

    # Find peak frequency in positive range
    pos = (freq >= 0) & coarse_band
    peak_idx = np.argmax(psd[pos])
    peak_freq = freq[pos][peak_idx]

    # Narrow band around peak
    band_lower = peak_freq * f_rel_low
    band_upper = peak_freq * f_rel_high
    narrow_band = (np.abs(freq) >= band_lower) & (np.abs(freq) <= band_upper)

    # Final mask
    final_mask = (coarse_band & narrow_band)

    # Apply mask
    fft_masked = fft_sig * final_mask

    # Inverse FFT → denoised signal
    denoised = np.fft.ifft(fft_masked).real

    # --- Debug plots ---
    if debug:
        masked_psd = np.abs(fft_masked)**2

        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axs[0].plot(freq[pos], psd[pos], label='Original PSD')
        axs[0].plot(freq[pos], masked_psd[pos], label='Filtered PSD')
        axs[0].axvline(peak_freq, color='r', linestyle='--', label=f'Peak: {peak_freq:.2f} Hz')
        axs[0].set_ylabel('Power')
        axs[0].legend()
        axs[0].set_title('Power Spectral Density')

        time = np.arange(n_t) * dt
        axs[1].plot(time, signal, label='Original')
        axs[1].plot(time, denoised, label='Reconstructed')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()
        axs[1].set_title('Signal: Original vs. Denoised')

        plt.tight_layout()
        plt.show()

    return denoised, peak_freq

def compute_phase_from_protophi(protophi: np.ndarray, nharm: int = 10) -> np.ndarray:
    """
    Compute a proper phase from a given protophase using harmonic correction.

    Parameters:
    - protophi (np.ndarray): Real-valued 1D array representing protophase angles (in radians).
    - nharm (int): Number of harmonics to use in correction (default: 10).

    Returns:
    - np.ndarray: Unwrapped corrected phase of the same shape as protophi.
    """
    if not isinstance(protophi, np.ndarray) or protophi.ndim != 1:
        raise ValueError("protophi must be a 1D NumPy array")
    if not np.issubdtype(protophi.dtype, np.floating):
        raise ValueError("protophi must be a real-valued array")

    k = np.arange(1, nharm + 1)
    exp_kphi = np.exp(-1j * np.outer(k, protophi))  # shape: (nharm, len(protophi))
    Sn_pos = np.mean(exp_kphi, axis=1)  # shape: (nharm,)

    phi = protophi.astype(np.complex128)
    correction_terms = Sn_pos[:, None] * (np.exp(1j * np.outer(k, protophi)) - 1) / (1j * k[:, None])
    phi += 2 * np.sum(correction_terms, axis=0)

    return np.unwrap(np.real(phi))



def process_signal_actuator(t,v):
    denoised, peak_freq = fourier_denoise(v,t[1]-t[0],f_rel_low=0.3,f_rel_high=1.7, debug=False)
    
    analytic = hilbert(denoised)
    analytic = analytic[HILBERT_REMOVE:-HILBERT_REMOVE] # Remove edge artifacts from Hilbert transform
    phase = np.unwrap(np.angle(analytic))
    phase=compute_phase_from_protophi(phase)

    return {
        'time':t,
        'phase': phase,
        'denoised': denoised,
    }

def plot_analysis( c_data, b_data):

    c_data["freq_phase"]=(c_data['phase'][-1]-c_data['phase'][0])/(c_data['time'][-1]-c_data['time'][0])/2/np.pi
    

    # time = time[HILBERT_REMOVE:-HILBERT_REMOVE]  # Adjust time to match analytic data length
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(c_data['t'], (c_data['v']-np.mean(c_data['v']))/np.std(c_data['v']), color='purple', label='Cilia')
    plt.scatter(c_data['final_t'], (c_data['final_v']-np.mean(c_data['v']))/np.std(c_data['v']), color='purple', label='Cilia')
    if b_data is not None:
        b_data["freq_phase"]=(b_data['phase'][-1]-b_data['phase'][0])/(b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE][-1]-b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE][0])/2/np.pi
        plt.plot(b_data['time'], b_data['denoised']/np.std(b_data['denoised']), color='orange', label='Beam')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(c_data['time'], c_data['phase'], color='purple', label=f'C frequency (phase): {c_data["freq_phase"]:.2f} Hz')
    if b_data is not None:
        plt.plot(b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE], b_data['phase'], color='orange', label=f'B frequency (phase): {b_data["freq_phase"]:.2f} Hz')
    plt.ylabel('Phase')
    plt.legend()

    plt.subplot(3, 1, 3)
    # Phase deviation from linear fit
    c_dev = c_data['phase'] - 2 * np.pi * c_data["freq_phase"] * (c_data['time'] - c_data['time'][0]) - c_data['phase'][0]
    if b_data is not None:
        b_dev = b_data['phase'] - 2 * np.pi * b_data["freq_phase"] * (b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE] - b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE][0]) - b_data['phase'][0]
        #diff = c_data['phase'] - b_data['phase']

    plt.plot(c_data['time'], c_dev, color='purple', label='C phase change')
    if b_data is not None:
        # Step 1: Determine overlapping time interval
        t_c = c_data['time']
        t_b = b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE]
        phase_b = b_data['phase']

        t_start = max(t_c[0], t_b[0])
        t_end = min(t_c[-1], t_b[-1])
        if t_end <= t_start:
            raise ValueError("No overlap between cilia and beam time arrays.")

        # Step 2: Create common time base (dense enough to resolve both signals)
        t_common = np.linspace(t_start, t_end, num=1000)

        # Step 3: Interpolate both phases to t_common
        from scipy.interpolate import interp1d
        interp_c = interp1d(t_c, c_data['phase'], kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_b = interp1d(t_b, phase_b, kind='linear', bounds_error=False, fill_value="extrapolate")

        phase_c_common = interp_c(t_common)
        phase_b_common = interp_b(t_common)

        # Step 4: Compute deviations from linear trend
        dev_c = phase_c_common - 2 * np.pi * c_data["freq_phase"] * (t_common - t_common[0]) - phase_c_common[0]
        dev_b = phase_b_common - 2 * np.pi * b_data["freq_phase"] * (t_common - t_common[0]) - phase_b_common[0]
        diff = dev_c - dev_b

        # Plot
        plt.plot(t_common, dev_b, color='orange', label='B phase change')
        plt.plot(t_common, diff, color='blue', label='C phase - B phase')

        # Annotate with multiples of π
        y_min = np.min(diff)
        y_max = np.max(diff)
        phase_min = np.floor(y_min / (2 * np.pi))
        phase_max = np.ceil(y_max / (2 * np.pi))
        if np.abs(phase_min - phase_max) < 10:
            for n in np.arange(phase_min, phase_max + 0.5, 0.5):
                pos = n * np.pi
                if n % 1 == 0:
                    plt.axhline(pos, color='black', linestyle='--', linewidth=0.5)  # integer multiples
                else:
                    plt.axhline(pos, color='black', linestyle=':', linewidth=0.5)   # half-integer multiples

    plt.xlabel('Time (s)')
    plt.ylabel('Phase difference')
    plt.legend()
    plt.tight_layout()
    plt.show()
    if b_data is not None:
        from scipy.signal import argrelextrema
        from scipy.interpolate import interp1d

        # Ensure phase and time arrays match
        b_time = b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE]
        b_phase = b_data['phase']
        denoised = b_data['denoised'][HILBERT_REMOVE:-HILBERT_REMOVE]

        # Find local minima in actuator denoised signal
        min_indices = argrelextrema(denoised, np.less, order=2)[0]
        min_times = b_time[min_indices]

        # Interpolate actuator phase at its own minima
        interp_phase_at_min = interp1d(b_time, b_phase, kind='linear', bounds_error=False, fill_value=np.nan)
        actuator_min_phases = interp_phase_at_min(min_times)
        actuator_min_phases = np.mod(actuator_min_phases, 2 * np.pi)

        # Compute circular mean of phases
        shift = np.angle(np.mean(np.exp(1j * actuator_min_phases)))
        if shift < 0:
            shift += 2 * np.pi

        # Interpolate actuator phase at cilia minima
        cilia_times = c_data['final_t']
        interp_b_phase = interp1d(b_time, b_phase, kind='linear', bounds_error=False, fill_value=np.nan)
        raw_angles = interp_b_phase(cilia_times)
        angles = np.mod(raw_angles - shift, 2 * np.pi)

        # Drop NaNs (e.g., outside interpolation range)
        valid_mask = ~np.isnan(angles)
        cilia_times = cilia_times[valid_mask]
        angles = angles[valid_mask]

        radii = cilia_times - cilia_times[0]

        # --- Polar scatter plot ---
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, polar=True)
        sc = ax1.scatter(angles, radii, c=radii, cmap='viridis', s=20)
        ax1.set_title("Polar Scatter: Actuator Phase @ Cilia Minima")
        plt.colorbar(sc, ax=ax1, label="Time since first minimum (s)")

        # --- Polar histogram ---
        ax2 = fig.add_subplot(1, 2, 2, polar=True)
        bins = 30
        hist_vals, bin_edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.bar(bin_centers, hist_vals, width=2*np.pi/bins, color='orange', edgecolor='black')
        ax2.set_title("Polar Histogram: Actuator Phase")

        plt.tight_layout()
        plt.show()



def analyze_before_after_beam_oscillation(
    df,
    filename,
    show_plots=True,
    std_window_size=5,
):
    
    segments = detect_actuator_activity_segments(
        df,
        window_size=std_window_size,
        debug=True
    )
    print(f"{filename}: detected {len(segments)} segments")
 
    # --- Split data and analyze ---
    split_index = np.searchsorted(df[TIME_COL].values, split_time)
    # --- Split data and analyze ---
    before_time = df[TIME_COL].values[:split_index]
    after_time = df[TIME_COL].values[split_index:]
    before_cilia = df[CILIA_COL].values[:split_index]
    after_cilia = df[CILIA_COL].values[split_index:]
    after_actuator = df[ACTUATOR_COL].values[split_index:]
    # Process signals with time and value
    before_c = process_signal(before_time, before_cilia) #change maybe because of better tracking
    after_c = process_signal(after_time, after_cilia)
    after_b = process_signal_actuator(after_time, after_actuator)

    
    # --- Plot results ---
    if show_plots:
        if before_c :
            plot_analysis(
                before_c, None
            )
        if after_c and after_b:
            plot_analysis(
                after_c, after_b
            )



def plot_full_tracked_before_after(filepath_pattern, show_plots=True):
    filepaths = glob.glob(filepath_pattern)
    for fpath in filepaths:
        df = pd.read_csv(fpath)
        filename = Path(fpath).name

        analyze_before_after_beam_oscillation(
            df,
            filename,
            show_plots=show_plots,
            std_window_size=WINDOW_LENGTH,
        )
        

if __name__ == "__main__":
    plot_full_tracked_before_after("/home/max/Documents/02_Data/Cilia_project/farzin/csv_to_analyze/*.csv", show_plots=True)