import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view
from powersmooth.powersmooth import powersmooth_general, upsample_with_mask,powersmooth_upsample
from scipy.signal import hilbert
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider
from scipy.signal import argrelextrema, savgol_filter

WINDOW_LENGTH = 30
CILIA_COL='Cilia_EndPoint_Y_um'
ACTUATOR_COL='Actuator_ymin_um'
TIME_COL='Time_s'
HILBERT_REMOVE=30


def detect_actuator_activity_segments(df, window_size: int = 5):
    """
    Detect continuous low- and high-activity segments in actuator movement,
    then pick the largest high-activity segment and the surrounding low-activity
    segments.  Allows fine-tuning of all four boundaries via sliders:
      • end_low1: end of the first low‐activity phase
      • start_high: start of the high‐activity phase
      • end_high: end of the high‐activity phase
      • start_low2: start of the final low‐activity phase

    Returns
    -------
    segments : List[Dict]
        Ordered list of dicts for each adjusted segment with keys:
        - 'type': 'low' or 'high'
        - 'start_time', 'end_time'
        - 'cilia_times', 'cilia_values'
        - 'actuator_values' (for 'high' only)
    """
    times = df[TIME_COL].to_numpy()
    actuator = df[ACTUATOR_COL].to_numpy()
    cilia = df[CILIA_COL].to_numpy()

    # sliding‐window std
    n = len(actuator)
    if window_size >= n:
        raise ValueError(f"window_size ({window_size}) must be < total frames ({n})")
    stds = np.std(sliding_window_view(actuator, window_size), axis=1)

    # threshold via KDE valley
    kde = gaussian_kde(stds)
    grid = np.linspace(stds.min(), stds.max(), 1000)
    dens = kde(grid)
    mins = argrelextrema(dens, np.less)[0]
    maxs = argrelextrema(dens, np.greater)[0]
    if len(mins) < 1 or len(maxs) < 2:
        raise RuntimeError("Could not detect bimodal std distribution.")
    peak2 = np.argsort(dens[maxs])[-2:]
    bounds = np.sort(grid[maxs[peak2]])
    thresh = grid[mins][(grid[mins] > bounds[0]) & (grid[mins] < bounds[1])][0]

    # label active/inactive
    active = stds > thresh

    # find all segments
    segs = []
    curr, start = active[0], 0
    for i in range(1, len(active)):
        if active[i] != curr:
            end = i - window_size
            if end > start:
                seg = {
                    'type': 'high' if curr else 'low',
                    'start_time': times[start],
                    'end_time':   times[end],
                    'cilia_times': times[start:end],
                    'cilia_values': cilia[start:end]
                }
                if curr:
                    seg['actuator_values'] = actuator[start:end]
                segs.append(seg)
            start, curr = i, active[i]
    # last
    if times[-1] > times[start]:
        seg = {
            'type': 'high' if curr else 'low',
            'start_time': times[start],
            'end_time':   times[-1],
            'cilia_times': times[start:],
            'cilia_values': cilia[start:]
        }
        if curr:
            seg['actuator_values'] = actuator[start:]
        segs.append(seg)

    # pick main high + flanking lows
    highs = [s for s in segs if s['type']=='high']
    if not highs:
        raise RuntimeError("No high-activity found")
    main_high = max(highs, key=lambda s: s['end_time']-s['start_time'])
    lows = [s for s in segs if s['type']=='low']
    low1 = max([s for s in lows if s['end_time'] <= main_high['start_time']] or [None],
               key=lambda s: (s['end_time']-s['start_time']) if s else -1)
    low2 = max([s for s in lows if s['start_time'] >= main_high['end_time']] or [None],
               key=lambda s: (s['end_time']-s['start_time']) if s else -1)

    # initial boundaries
    t0, t_end = times[0], times[-1]
    end_low1   = low1['end_time']    if low1 else t0
    start_high = main_high['start_time']
    end_high   = main_high['end_time']
    start_low2 = low2['start_time']  if low2 else t_end

    # 6) interactive plot
    fig, ax = plt.subplots(figsize=(12,6))
    t_plot = times[:len(stds)]

    def redraw(end_low1, start_high, end_high, start_low2):
        ax.clear()
        ax.plot(t_plot, stds, color='gray', label='Sliding STD')
        ax.axhline(thresh, color='red', linestyle='--', label=f'Thresh={thresh:.3g}')
        # shade regions
        ax.axvspan(t0,        end_low1,   color='green',  alpha=0.3)
        ax.axvspan(end_low1,  start_high, color='yellow', alpha=0.3)
        ax.axvspan(start_high,end_high,  color='orange', alpha=0.3)
        ax.axvspan(end_high,  start_low2, color='yellow', alpha=0.3)
        ax.axvspan(start_low2,t_end,      color='green',  alpha=0.3)
        ax.set_xlim(t0, t_end)
        ax.set_ylim(stds.min(), stds.max()*1.1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('STD')
        ax.set_title('Adjust activity boundaries')
        ax.legend(loc='upper right')

    redraw(end_low1, start_high, end_high, start_low2)

    axcol = 'lightgoldenrodyellow'
    s_low2 = Slider(plt.axes([0.15, 0.02, 0.7, 0.02], facecolor=axcol),
                    'Start 3', end_high, t_end, valinit=start_low2)
    s_eh   = Slider(plt.axes([0.15, 0.06, 0.7, 0.02], facecolor=axcol),
                    'End 2',   start_high, start_low2, valinit=end_high)
    s_sh   = Slider(plt.axes([0.15, 0.10, 0.7, 0.02], facecolor=axcol),
                    'Start 2', end_low1, end_high, valinit=start_high)
    s_low1 = Slider(plt.axes([0.15, 0.14, 0.7, 0.02], facecolor=axcol),
                    'End 1',   t0, start_high, valinit=end_low1)


    def update(val):
        e1 = s_low1.val
        sh = s_sh.val
        eh = s_eh.val
        l2 = s_low2.val
        # enforce ordering
        if not (t0 <= e1 <= sh <= eh <= l2 <= t_end):
            return
        # adjust slider bounds dynamically
        s_sh.valmin = e1;   s_sh.valmax = eh
        s_eh.valmin = sh;   s_eh.valmax = l2
        s_low2.valmin = eh
        redraw(e1, sh, eh, l2)
        fig.canvas.draw_idle()

    for s in (s_low1, s_sh, s_eh, s_low2):
        s.on_changed(update)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.267)
    plt.show()

    # 7) build final segments
    out = []
    if low1:
        mask = (times >= t0) & (times <= s_low1.val)
        out.append({
            'type':'low','start_time':t0,'end_time':s_low1.val,
            'cilia_times':times[mask],'cilia_values':cilia[mask]
        })
    mask = (times >= s_sh.val) & (times <= s_eh.val)
    out.append({
        'type':'high','start_time':s_sh.val,'end_time':s_eh.val,
        'cilia_times':times[mask],'cilia_values':cilia[mask],
        'actuator_values':actuator[mask]
    })
    if low2:
        mask = (times >= s_low2.val) & (times <= t_end)
        out.append({
            'type':'low','start_time':s_low2.val,'end_time':t_end,
            'cilia_times':times[mask],'cilia_values':cilia[mask]
        })
    return out

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

def process_signal_cilia(t,v,flat_tol=0.001,debug=False,smooth_weight=1e-16):
    t = np.asarray(t)
    v_orig = np.asarray(v)

    # v = (np.asarray(v_orig) - np.mean(v_orig)) / np.std(v_orig)
    # t, v,_ = powersmooth_upsample(t_orig, v_orig, weights = {2: 1e-10, 3: 1e-14}, dx=(t[1]-t[0])/5)
    
    v, peak_freq = fourier_denoise(v_orig,t[1]-t[0],f_rel_low=0.3,f_rel_high=1.7, debug=False)
    if debug:
        plt.plot(t,v_orig)
        #plt.plot(t,v)
        plt.plot(t,v)
        plt.show()
    minima_t = []
    minima_v = []

    mean_v = np.mean(v)
    i = 1
    while i < len(v) - 1:
        # Case 1: standard local minimum
        if v[i - 1] > v[i] < v[i + 1] and v[i] < mean_v:
            minima_t.append(t[i])
            minima_v.append(v[i])
            i += 1
        # Case 2: flat local minimum
        elif v[i - 1] > v[i] and np.isclose(v[i], v[i + 1], rtol=flat_tol):
            start = i
            while i + 1 < len(v) and np.isclose(v[i], v[i + 1], rtol=flat_tol):
                i += 1
            end = i
            flat_v = v[start:end + 1].mean()
            if end + 1 < len(v) and v[end + 1] > v[end] and flat_v < mean_v:
                flat_t = t[start:end + 1].mean()
                minima_t.append(flat_t)
                minima_v.append(flat_v)
            i += 1
        else:
            i += 1


    _,peak_spacing=plot_minima_spacing_histogram_kde(minima_t)

    minima_t = np.array(minima_t)
    minima_v = np.array(minima_v)


    if debug:
        # --- Plot signal and detected minima ---
        plt.figure(figsize=(10, 3))
        plt.plot(t, v, label='Signal')
        plt.scatter(minima_t, minima_v, color='red', label=' Minima')
        # plt.scatter(filtered_t, filtered_v, color='green', label='Filtered Minima')
        plt.xlabel("Time (s)")
        plt.ylabel("Signal")
        plt.title("Signal with Local Minima (Filtered)")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    if len(minima_t)<1:
        print('no min')
        return None


    final_phase = []
    phase_val = 0.0  # start at 0
    for i in range(len(minima_t) - 1):
        final_phase.append(phase_val)
        phase_val += 2 * np.pi 


    final_phase.append(phase_val)

    phase_raw = np.array(final_phase)


    n_phase = np.arange(len(minima_t))
    phase_raw = 2 * np.pi * n_phase  # Arbitrary phase assignment (e.g. 2π steps)

    # --- Upsample phase signal for smoothing ---
    target_dx = 0.1 * peak_spacing  # desired resolution
    t_dense, phase_sparse, mask = upsample_with_mask(minima_t, phase_raw, dx=target_dx)

    # --- Apply powersmooth ---
    phase_smooth = powersmooth_general(t_dense, phase_sparse, weights={2: smooth_weight, 3: 0e-9}, mask=mask)

    if debug:
        # --- Plot raw vs smoothed phase ---
        plt.figure(figsize=(10, 3))
        plt.plot(minima_t, phase_raw, 'o', label='Raw Phase (2π steps)')
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
        'final_t':minima_t,
        'final_v':minima_v,
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

def plot_analysis(c_data, b_data):
    c_data["freq_phase"] = (c_data['phase'][-1] - c_data['phase'][0]) / (c_data['time'][-1] - c_data['time'][0]) / (2 * np.pi)

    if b_data is not None:
        b_data["freq_phase"] = (b_data['phase'][-1] - b_data['phase'][0]) / (
            b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE][-1] - b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE][0]) / (2 * np.pi)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    # 1. Plot signal with minima
    axs[0].plot(c_data['t'], (c_data['v'] - np.mean(c_data['v'])) / np.std(c_data['v']), color='purple', label='Cilia')
    axs[0].scatter(c_data['final_t'], (c_data['final_v'] - np.mean(c_data['v'])) / np.std(c_data['v']), color='purple')
    if b_data is not None:
        axs[0].plot(b_data['time'], b_data['denoised'] / np.std(b_data['denoised']), color='orange', label='Beam')
    axs[0].set_ylabel('Intensity')
    axs[0].legend()

    # 2. Plot phase - linear trend
    t_c = c_data['time']
    phi_c = c_data['phase']
    linear_trend_c = phi_c[0] + 2 * np.pi * c_data["freq_phase"] * (t_c - t_c[0])
    phase_dev_c = phi_c - linear_trend_c
    axs[1].plot(t_c, phase_dev_c, color='purple', label=f'Cilia Phase Deviation (f = {c_data["freq_phase"]:.2f} Hz)')
    axs[1].set_ylabel('Phase − Linear Trend (rad)')
    axs[1].legend()
    if b_data is not None:
        t_b = b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE]
        phi_b = b_data['phase']
        freq_b = b_data["freq_phase"]
        linear_trend_b = phi_b[0] + 2 * np.pi * freq_b * (t_b - t_b[0])
        phase_dev_b = phi_b - linear_trend_b
        axs[1].plot(t_b, phase_dev_b, color='orange', label=f'Actuator Phase Deviation (f = {freq_b:.2f} Hz)')
        axs[1].legend()


    # 3. Phase difference (only if beam data is available)
    if b_data is not None:
        t_b = b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE]
        phi_b = b_data['phase']

        # Determine overlapping interval
        t_start = max(t_c[0], t_b[0])
        t_end = min(t_c[-1], t_b[-1])
        if t_end <= t_start:
            raise ValueError("No overlap between cilia and beam time arrays.")

        t_common = np.linspace(t_start, t_end, num=1000)
        interp_c = interp1d(t_c, phi_c, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp_b = interp1d(t_b, phi_b, kind='linear', bounds_error=False, fill_value="extrapolate")

        phi_c_common = interp_c(t_common)
        phi_b_common = interp_b(t_common)

        phase_diff = phi_c_common - phi_b_common
        phase_diff -= np.mean(phase_diff)

        axs[2].plot(t_common, phase_diff, color='blue', label='Cilia − Beam Phase')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Phase Diff (rad)')
        axs[2].legend()

        # Optional guiding lines if range is small
        y_min, y_max = np.min(phase_diff), np.max(phase_diff)
        if np.abs(y_max - y_min) < 10 * np.pi:
            phase_min = np.floor(y_min / (np.pi))
            phase_max = np.ceil(y_max / (np.pi))
            for n in np.arange(phase_min, phase_max + 0.5, 0.5):
                pos = n * np.pi
                style = '--' if n % 1 == 0 else ':'
                axs[2].axhline(pos, color='black', linestyle=style, linewidth=0.5)

    plt.tight_layout()
    plt.show()


    if b_data is not None:
        # --- Setup ---
        connect_points = True  # ← Set to True to connect the scatter points with lines

        b_time = b_data['time'][HILBERT_REMOVE:-HILBERT_REMOVE]
        b_phase = b_data['phase']
        denoised = b_data['denoised'][HILBERT_REMOVE:-HILBERT_REMOVE]

        # --- Find minima in actuator signal ---
        min_indices = argrelextrema(denoised, np.less, order=2)[0]
        min_times = b_time[min_indices]

        interp_phase_at_min = interp1d(b_time, b_phase, kind='linear', bounds_error=False, fill_value=np.nan)
        actuator_min_phases = interp_phase_at_min(min_times)
        actuator_min_phases = np.mod(actuator_min_phases, 2 * np.pi)

        shift = np.angle(np.mean(np.exp(1j * actuator_min_phases)))
        if shift < 0:
            shift += 2 * np.pi

        # --- Actuator phase at cilia minima ---
        cilia_times = c_data['final_t']
        interp_b_phase = interp1d(b_time, b_phase, kind='linear', bounds_error=False, fill_value=np.nan)
        raw_angles = interp_b_phase(cilia_times)
        angles = np.mod(raw_angles - shift, 2 * np.pi)

        # --- Clean ---
        valid_mask = ~np.isnan(angles)
        cilia_times = cilia_times[valid_mask]
        angles = angles[valid_mask]
        radii = cilia_times - cilia_times[0]+cilia_times[-1]*0.3

        # --- Polar plots ---
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1, polar=True)

        if connect_points:
            ax1.plot(angles, radii, color='black', linewidth=1.0, marker='o', markersize=5, label='Minima')
        else:
            ax1.scatter(angles, radii, color='black', s=20, label='Minima')

        ax1.set_title("Actuator Phase @ Cilia Minima: radius = 'time'")

        # --- Polar histogram ---
        ax2 = fig.add_subplot(1, 2, 2, polar=True)
        bins = 30
        hist_vals, bin_edges = np.histogram(angles, bins=bins, range=(0, 2*np.pi))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.bar(bin_centers, hist_vals, width=2*np.pi/bins, color='orange', edgecolor='black')
        ax2.set_title("Actuator Phase @ Cilia Minima: Histogram")

        plt.tight_layout()
        plt.show()

def plot_psd_histogram_by_segment(segments, dt):
    """
    Plot normalized PSD curves for cilia and actuator across multiple segments.
    Low-activity (before/after) segments are shown in green,
    high-activity subsegments are shown in gray (brightness ∝ mean distance).
    Actuator PSDs are shown in red (overlaid).

    Parameters
    ----------
    segments : list of dict
        Output of split_distancebased_segments(), including 'low' and 'high' segments.
        Each high segment must have 'actuator_values' and 'mean_distance'.
    dt : float
        Sampling interval (s).
    """
    from scipy.fft import fft, fftfreq
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Separate low/high segments
    lows = [s for s in segments if s['type'] == 'low']
    highs = [s for s in segments if s['type'] == 'high']

    # --- Compute PSD helper ---
    def compute_norm_psd(signal, dt):
        signal = np.asarray(signal) - np.mean(signal)
        n = len(signal)
        freqs = fftfreq(n, d=dt)[:n//2]
        psd = np.abs(fft(signal))[:n//2] ** 2
        f_peak = freqs[np.argmax(psd)]
        fmin, fmax = 0.3 * f_peak, 3.5 * f_peak
        mask = (freqs >= fmin) & (freqs <= fmax)
        freqs_m = freqs[mask]
        psd_m = psd[mask]
        area = np.trapz(psd_m, freqs_m)
        psd_norm = psd_m / area if area > 0 else psd_m
        return freqs_m, psd_norm, f_peak

    # --- Create figure ---
    fig, (ax_c, ax_a) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Cilia PSDs ---
    cmap_gray = cm.get_cmap('Greys')
    if highs:
        dist_vals = np.array([h['mean_distance'] for h in highs])
        # Normalize distances to [0.3, 0.9] gray levels
        if np.ptp(dist_vals) == 0:
            normed = np.full_like(dist_vals, 0.6)
        else:
            normed = 0.3 + 0.6 * (dist_vals - dist_vals.min()) / (dist_vals.max() - dist_vals.min())

    # Before (low1)
    if len(lows) >= 1:
        freqs_m, psd_norm, f_peak = compute_norm_psd(lows[0]['cilia_values'], dt)
        ax_c.plot(freqs_m, psd_norm, color='green', lw=2, label=f'Before: {f_peak:.2f} Hz')

    # Middle (highs)
    for i, h in enumerate(highs):
        freqs_m, psd_norm, f_peak = compute_norm_psd(h['cilia_values'], dt)
        color = cmap_gray(normed[i])
        ax_c.plot(freqs_m, psd_norm, color=color, lw=2, label=f'High {i+1}: {f_peak:.2f} Hz, Δ={h["mean_distance"]:.1f}µm')

    # After (low2)
    if len(lows) == 2:
        freqs_m, psd_norm, f_peak = compute_norm_psd(lows[-1]['cilia_values'], dt)
        ax_c.plot(freqs_m, psd_norm, color='green', lw=2, ls='--', label=f'After: {f_peak:.2f} Hz')

    ax_c.set_title('Cilia PSD (normalized)')
    ax_c.set_ylabel('Normalized Power')
    ax_c.legend(fontsize=8)

    # --- Actuator PSDs (all high segments) ---
    for i, h in enumerate(highs):
        freqs_m, psd_norm, f_peak = compute_norm_psd(h['actuator_values'], dt)
        ax_a.plot(freqs_m, psd_norm, color='red', lw=1.5, alpha=0.8,
                  label=f'Actuator {i+1}: {f_peak:.2f} Hz')

    ax_a.set_title('Actuator PSDs (normalized)')
    ax_a.set_xlabel('Frequency (Hz)')
    ax_a.set_ylabel('Normalized Power')
    ax_a.legend(fontsize=8)

    plt.tight_layout()
    plt.show()

def split_distancebased_segments(
    segments,
    envelope_timescale=0.5,
    poly_order=3,
    distance_cutoff=3.5,
    max_segments=5,
    debug=True
):
    """
    Split the middle segment into subsegments based on drift of the distance
    between cilia and actuator lower envelopes.

    Parameters
    ----------
    segments : list
        Output of detect_actuator_activity_segments().
    envelope_timescale : float
        Characteristic smoothing time (seconds) for envelope extraction.
    poly_order : int
        Polynomial order for Savitzky–Golay smoothing.
    distance_cutoff : float
        Maximum allowed variation (µm) in cilia–actuator distance per subsegment.
    max_segments : int
        Maximum number of subsegments to create.
    debug : bool
        If True, print diagnostic info and show plots.

    Returns
    -------
    subsegments : list of dict
        Each dict has keys: 'start_time', 'end_time', 'mean_distance'.
    """
    cilia_values = np.asarray(segments[1]['cilia_values'])
    cilia_times = np.asarray(segments[1]['cilia_times'])
    actuator_values = np.asarray(segments[1]['actuator_values'])

    dt = np.median(np.diff(cilia_times))
    window_len = int(np.clip(envelope_timescale / dt, 5, len(cilia_times)//3))
    if window_len % 2 == 0:
        window_len += 1  # must be odd for savgol_filter

    def compute_lower_envelope(t, v, label):
        min_idx = argrelextrema(v, np.less, order=5)[0]
        if len(min_idx) < 3:
            print(f"⚠️ Not enough minima for {label}.")
            return np.full_like(v, np.nan)
        t_min = t[min_idx]
        v_min = v[min_idx]
        f_env = interp1d(t_min, v_min, kind='linear', bounds_error=False, fill_value='extrapolate')
        env_raw = f_env(t)
        env_smooth = savgol_filter(env_raw, window_length=window_len, polyorder=poly_order)
        return env_smooth

    env_cilia = compute_lower_envelope(cilia_times, cilia_values, "Cilia")
    env_act = compute_lower_envelope(cilia_times, actuator_values, "Actuator")

    distance = env_cilia - env_act

    # --- Determine segment boundaries based on distance variation ---
    n_segments = 1
    cut_indices = []
    while n_segments <= max_segments:
        split_idx = np.linspace(0, len(distance) - 1, n_segments + 1, dtype=int)
        ok = True
        for i in range(n_segments):
            seg = distance[split_idx[i]:split_idx[i+1]]
            if np.nanmax(seg) - np.nanmin(seg) > distance_cutoff:
                ok = False
                break
        if ok:
            break
        n_segments += 1

    # Final indices
    split_idx = np.linspace(0, len(distance) - 1, n_segments + 1, dtype=int)

    # --- Compute mean distance per subsegment ---
    subsegments = []
    for i in range(n_segments):
        start_i, end_i = split_idx[i], split_idx[i+1]
        seg_dist = distance[start_i:end_i]
        mean_dist = np.nanmean(seg_dist)
        subsegments.append({
            'type': 'high',  # subsegments come from high-activity region
            'start_time': cilia_times[start_i],
            'end_time': cilia_times[end_i-1],
            'cilia_times': cilia_times[start_i:end_i],
            'cilia_values': cilia_values[start_i:end_i],
            'actuator_values': actuator_values[start_i:end_i],
            'mean_distance': mean_dist,
        })

    if debug:
        print(f"\nSplit into {n_segments} subsegments (cutoff = {distance_cutoff} µm):")
        for i, s in enumerate(subsegments):
            print(f"  Segment {i+1}: {s['start_time']:.2f}–{s['end_time']:.2f}s, "
                  f"mean Δ = {s['mean_distance']:.3f} µm")
        # --- Plot as before ---
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(cilia_times, cilia_values, color='purple', alpha=0.6, label='Cilia')
        ax1.plot(cilia_times, actuator_values, color='orange', alpha=0.6, label='Actuator')
        ax1.plot(cilia_times, env_cilia, color='purple', lw=2, label='Cilia Lower Envelope')
        ax1.plot(cilia_times, env_act, color='orange', lw=2, label='Actuator Lower Envelope')
        for s in subsegments[:-1]:
            ax1.axvline(s['end_time'], color='black', ls='--', lw=1)
        ax2 = ax1.twinx()
        ax2.plot(cilia_times, distance, color='gray', lw=1, label='Envelope Distance (Cilia−Actuator)')
        ax2.set_ylabel("Envelope Distance (µm)", color='gray')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude (µm)")
        ax1.set_title("Segment splitting based on envelope distance drift")
        ax1.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

    # --- Reassemble full segment list (low1 + subsegments + low2) ---
    new_segments = []
    if len(segments) >= 1 and segments[0]['type'] == 'low':
        new_segments.append(segments[0])
    new_segments.extend(subsegments)
    if len(segments) >= 3 and segments[-1]['type'] == 'low':
        new_segments.append(segments[-1])

    return new_segments

def analyze(
    df,
    filename,
    show_seg_plots=True,
    show_PSD=True,
    std_window_size=5,
):
    segments = detect_actuator_activity_segments(df, window_size=std_window_size)
    print(f"{filename}: detected {len(segments)} segments")
    if len(segments)!=3:
        raise
    
    #split intermediate segments according to distance
    segments = split_distancebased_segments(segments)
    

    # Lists of processed data dicts
    cilia_procs   = []
    actuator_procs = []
    for i, seg in enumerate(segments):
        print(i,seg)
        
        print(f"Segment {i+1}/{len(segments)}: {seg['type']} activity, "
              f"t = {seg['start_time']:.2f}–{seg['end_time']:.2f}s")

        # process cilia
        c_data = process_signal_cilia(seg['cilia_times'], seg['cilia_values'], debug=False)
        if c_data is None:
            continue

        # process actuator for high‐activity
        if seg['type'] == 'high':
            b_data = process_signal_actuator(seg['cilia_times'], seg['actuator_values'])
        else:
            b_data = None

        # optionally plot each segment
        if show_seg_plots:
            plot_analysis(c_data, b_data)

        cilia_procs.append(c_data)
        actuator_procs.append(b_data)

    # PSD block 
    if show_PSD:
        dt = np.median(np.diff(df[TIME_COL]))
        plot_psd_histogram_by_segment(segments, dt)


  
def full_analysis(filepath_pattern, show_seg_plots=True, show_PSD=True):
    filepaths = glob.glob(filepath_pattern)
    all_res=[]
    for fpath in filepaths:
        df = pd.read_csv(fpath)
        filename = Path(fpath).name

        res=analyze(
            df,
            filename,
            show_seg_plots=show_seg_plots,
            show_PSD=show_PSD,
            std_window_size=WINDOW_LENGTH,
        )
        all_res.append(res)
    
    #convert to csv or df or np object

if __name__ == "__main__":
    # full_analysis("/home/max/Documents/02_Data/Cilia_project/farzin/csv_to_analyze/*.csv", show_seg_plots=True, show_PSD=True)
    full_analysis("/home/max/Downloads/download/*.csv", show_seg_plots=False, show_PSD=True)