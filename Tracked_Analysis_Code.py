import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from scipy.signal import argrelextrema
def find_local_minima(signal, tolerance=0.001):
    minima_indices = []
    i = 1
    n = len(signal)

    while i < n - 1:
        prev = signal[i - 1]
        curr = signal[i]
        next = signal[i + 1]

        # Standard local minimum
        if curr < prev and curr < next:
            minima_indices.append(i)
            i += 1
        # Flat or nearly flat minimum
        elif (abs(curr - next) / max(abs(curr), 1e-12) < tolerance) and curr < prev:
            # Start of flat region
            start = i
            while i + 1 < n and abs(signal[i] - signal[i + 1]) / max(abs(signal[i]), 1e-12) < tolerance:
                i += 1
            end = i
            middle = (start + end) // 2
            if middle > 0 and middle < n - 1 and signal[middle] < signal[middle - 1] and signal[middle] < signal[middle + 1]:
                minima_indices.append(middle)
            i += 1
        else:
            i += 1

    return np.array(minima_indices)

def detect_and_plot_minima(df):
    t = df["Time_s"].values
    y = df["Cilia_EndPoint_Y_um"].values
    y_act = df["Actuator_ymin_um"].values

    # Use the custom local minima detection
    y_min_idx = find_local_minima(y)
    y_act_min_idx = find_local_minima(y_act)

    # Plot y(t) and y_act(t) with minima
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label='Y(t)', color='tab:orange')
    plt.plot(t, y_act, label='Actuator Y(t)', color='tab:green')
    plt.scatter(t[y_min_idx], y[y_min_idx], color='tab:orange', label='Y minima', marker='o')
    plt.scatter(t[y_act_min_idx], y_act[y_act_min_idx], color='tab:green', label='Actuator Y minima', marker='x')
    plt.xlabel("Time [s]")
    plt.ylabel("Displacement [μm]")
    plt.title("Y and Actuator Y with Detected Minima")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Time difference between successive minima
    dt_y = np.diff(t[y_min_idx])
    dt_y_act = np.diff(t[y_act_min_idx])

    # Plot histograms with finer bins
    plt.figure(figsize=(10, 5))
    plt.hist(dt_y, bins=50, alpha=0.7, label='Y minima Δt', color='tab:orange')
    plt.hist(dt_y_act, bins=50, alpha=0.7, label='Actuator Y minima Δt', color='tab:green')
    plt.xlabel("Time between minima [s]")
    plt.ylabel("Count")
    plt.title("Histogram of Time Intervals Between Minima")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    refined_y_min_idx, spacing_y = refine_minima_with_phase(t, y, y_min_idx)
    print(f"Typical spacing Y: {spacing_y:.4f} s")

    # Plot original and improved minima
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label='Y(t)', color='tab:orange')
    plt.scatter(t[y_min_idx], y[y_min_idx], color='tab:orange', marker='o', label='Original minima')
    plt.scatter(t[refined_y_min_idx], y[refined_y_min_idx], color='black', marker='x', label='Refined minima')
    plt.xlabel("Time [s]")
    plt.ylabel("Y [μm]")
    plt.title("Original and Refined Y Minima")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from scipy.stats import gaussian_kde

def refine_minima_with_phase(t, y, minima_indices, tolerance_ratio=0.2):
    times = t[minima_indices]
    dt = np.diff(times)

    # Estimate typical spacing using KDE
    kde = gaussian_kde(dt)
    x_vals = np.linspace(min(dt), max(dt), 1000)
    pdf = kde(x_vals)
    typical_spacing = x_vals[np.argmax(pdf)]

    # Accept intervals close to the typical spacing
    refined_times = [times[0]]
    for i in range(1, len(times)):
        gap = times[i] - refined_times[-1]
        expected_count = int(round(gap / typical_spacing))
        if abs(gap - expected_count * typical_spacing) <= tolerance_ratio * typical_spacing:
            # Insert missing if any
            if expected_count > 1:
                for k in range(1, expected_count):
                    refined_times.append(refined_times[-1] + k * typical_spacing)
            refined_times.append(times[i])
        else:
            # Gap too small: probably overdetection → skip this minimum
            continue

    # Map refined_times back to indices (closest actual index in t)
    refined_indices = [np.argmin(np.abs(t - ti)) for ti in refined_times]

    return np.array(refined_indices), typical_spacing

def plot_psd(df):
    # Extract time and signals
    t = df["Time_s"].values
    x = df["Cilia_EndPoint_X_um"].values
    y = df["Cilia_EndPoint_Y_um"].values
    y_act = df["Actuator_ymin_um"].values

    dt = t[1] - t[0]  # assume constant sampling rate
    fs = 1.0 / dt     # sampling frequency
    n = len(t)

    def compute_psd(signal):
        # Remove mean to avoid DC peak
        signal_detrended = signal - np.mean(signal)
        fft = np.fft.rfft(signal_detrended)
        psd = np.abs(fft)**2 / (n * fs)
        freqs = np.fft.rfftfreq(n, d=dt)
        return freqs, psd

    # Compute PSDs
    freqs_x, psd_x = compute_psd(x)
    freqs_y, psd_y = compute_psd(y)
    freqs_act, psd_act = compute_psd(y_act)

    # Plot
    plt.figure(figsize=(10, 8))

    plt.subplot(3, 1, 1)
    plt.semilogy(freqs_x, psd_x, label='X PSD')
    plt.title("Power Spectral Density: X(t)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.semilogy(freqs_y, psd_y, label='Y PSD', color='tab:orange')
    plt.title("Power Spectral Density: Y(t)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.semilogy(freqs_act, psd_act, label='Actuator Y PSD', color='tab:green')
    plt.title("Power Spectral Density: Actuator Y(t)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()




def main():
    # Load the CSV file
    filepath = r"C:\Users\kotzm\Downloads\calibrated_cilia_actuator_geodesic_coords.csv"
    df = pd.read_csv(filepath)

    # Create the figure and subplots
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 1. Trajectory plot (Cilia X vs Y)
    axs[0].plot(df["Cilia_EndPoint_X_um"], df["Cilia_EndPoint_Y_um"], label='Cilia Trajectory')
    axs[0].set_xlabel("X [μm]")
    axs[0].set_ylabel("Y [μm]")
    axs[0].set_title("Cilia Trajectory (X vs Y)")
    axs[0].grid(True)
    axs[0].legend()

    # 2. X(t)
    axs[1].plot(df["Time_s"], df["Cilia_EndPoint_X_um"], label='X(t)', color='tab:blue')
    axs[1].set_ylabel("X [μm]")
    axs[1].set_title("Cilia X over Time")
    axs[1].grid(True)
    axs[1].legend()

    # 3. Y(t)
    axs[2].plot(df["Time_s"], df["Cilia_EndPoint_Y_um"], label='Y(t)', color='tab:orange')
    axs[2].set_ylabel("Y [μm]")
    axs[2].set_title("Cilia Y over Time")
    axs[2].grid(True)
    axs[2].legend()

    # 4. Actuator y(t)
    axs[3].plot(df["Time_s"], df["Actuator_ymin_um"], label='Actuator y(t)', color='tab:green')
    axs[3].set_xlabel("Time [s]")
    axs[3].set_ylabel("Actuator Y [μm]")
    axs[3].set_title("Actuator Minimum Y over Time")
    axs[3].grid(True)
    axs[3].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


    #plot_psd(df)

    detect_and_plot_minima(df)

if __name__ == "__main__":
    main()
