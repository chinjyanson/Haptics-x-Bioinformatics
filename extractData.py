"""
Data Extraction and Visualization for Muse 2 EEG and Polar H10 HR Data
Includes Morlet wavelet analysis, ICA visualization, and HRV metrics analysis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_array_morlet
from typing import Optional, Dict, List, Tuple, Any


class DataExtractor:
    """Extract and analyze data from BCI experiment sessions."""

    def __init__(self, data_path: str):
        """
        Initialize the data extractor.

        Args:
            data_path: Path to the session data (without extension, or with .json)
        """
        self.data_path = Path(data_path)
        self.base_path = self.data_path.with_suffix('') if self.data_path.suffix else self.data_path

        # Data storage
        self.eeg_data: Optional[np.ndarray] = None
        self.eeg_times: Optional[np.ndarray] = None
        self.hr_data: Optional[pd.DataFrame] = None
        self.metadata: Optional[Dict] = None
        # Muse 2 has 4 EEG electrodes via Bluetooth: TP9, AF7, AF8, TP10
        # Note: A 5th channel (AUX) requires physical micro-USB connection
        self.channels = ['TP9', 'AF7', 'AF8', 'TP10']
        self.sampling_rate = 256  # Default Muse 2 sampling rate

    def load_json(self) -> Dict[str, Any]:
        """Load data from JSON file."""
        json_path = self.base_path.with_suffix('.json')
        if not json_path.exists():
            raise FileNotFoundError(f"JSON file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        self.metadata = data.get('metadata', {})
        self.sampling_rate = self.metadata.get('muse_sampling_rate', 256)
        self.channels = self.metadata.get('muse_channels', self.channels)

        # Extract EEG data
        eeg_samples = data.get('eeg_data', [])
        if eeg_samples:
            self.eeg_times = np.array([s['time'] for s in eeg_samples])
            self.eeg_data = np.array([
                [s['channels'].get(ch, 0) for ch in self.channels]
                for s in eeg_samples
            ]).T  # Shape: (n_channels, n_samples)

        # Extract HR data
        hr_samples = data.get('hr_data', [])
        if hr_samples:
            self.hr_data = pd.DataFrame(hr_samples)
            # Expand rr_intervals list into separate analysis

        print(f"Loaded data from {json_path}")
        print(f"  EEG samples: {self.eeg_data.shape[1] if self.eeg_data is not None else 0}")
        print(f"  HR samples: {len(self.hr_data) if self.hr_data is not None else 0}")

        return data

    def load_csv(self) -> None:
        """Load data from CSV files."""
        eeg_path = Path(f"{self.base_path}_eeg.csv")
        hr_path = Path(f"{self.base_path}_hr.csv")

        if eeg_path.exists():
            eeg_df = pd.read_csv(eeg_path)
            self.eeg_times = eeg_df['time'].values
            self.eeg_data = eeg_df[self.channels].values.T
            print(f"Loaded EEG data: {self.eeg_data.shape}")

        if hr_path.exists():
            self.hr_data = pd.read_csv(hr_path)
            print(f"Loaded HR data: {len(self.hr_data)} samples")

    def load_data(self) -> None:
        """Load data from available files (prefers JSON)."""
        json_path = self.base_path.with_suffix('.json')
        if json_path.exists():
            self.load_json()
        else:
            self.load_csv()

    # ==================== MORLET WAVELET ANALYSIS ====================

    def compute_morlet_wavelet(self, channel_idx: int = 0,
                               freqs: Optional[np.ndarray] = None,
                               n_cycles: float = 7.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Morlet wavelet transform for a specific channel using MNE.

        Args:
            channel_idx: Index of the EEG channel to analyze
            freqs: Frequencies to analyze (default: 1-40 Hz)
            n_cycles: Number of cycles in wavelet (default: 7)

        Returns:
            Tuple of (power, freqs, times)
        """
        if self.eeg_data is None:
            raise ValueError("No EEG data loaded. Call load_data() first.")

        if freqs is None:
            freqs = np.arange(1, 41, 0.5)  # 1-40 Hz in 0.5 Hz steps

        # Prepare data for MNE: shape (n_epochs, n_channels, n_times)
        data = self.eeg_data[channel_idx:channel_idx+1, :][np.newaxis, :, :]

        # Compute Morlet wavelet transform using MNE
        # n_cycles can be a float or array of same length as freqs
        power = tfr_array_morlet(
            data,
            sfreq=self.sampling_rate,
            freqs=freqs,
            n_cycles=n_cycles,
            output='power',
            verbose=False
        )

        # power shape: (n_epochs, n_channels, n_freqs, n_times)
        # Extract the single epoch and channel
        power = power[0, 0, :, :]  # Shape: (n_freqs, n_times)

        times = self.eeg_times if self.eeg_times is not None else np.arange(self.eeg_data.shape[1]) / self.sampling_rate

        return power, freqs, times

    def plot_morlet_wavelet(self, channel_idx: int = 0,
                           freqs: Optional[np.ndarray] = None,
                           figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """
        Plot Morlet wavelet time-frequency representation.

        Args:
            channel_idx: Index of the EEG channel to analyze
            freqs: Frequencies to analyze
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        power, freqs, times = self.compute_morlet_wavelet(channel_idx, freqs)

        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 2])

        # Plot raw signal
        axes[0].plot(times, self.eeg_data[channel_idx, :], 'b-', linewidth=0.5)
        axes[0].set_ylabel('Amplitude (uV)')
        axes[0].set_title(f'Raw EEG Signal - {self.channels[channel_idx]}')
        axes[0].set_xlim(times[0], times[-1])
        axes[0].grid(True, alpha=0.3)

        # Plot spectrogram
        im = axes[1].pcolormesh(times, freqs, 10 * np.log10(power + 1e-10),
                                shading='gouraud', cmap='jet')
        axes[1].set_ylabel('Frequency (Hz)')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_title('Morlet Wavelet Time-Frequency Representation')

        # Add frequency band annotations
        band_limits = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
                      'Beta': (13, 30), 'Gamma': (30, 45)}
        for band, (low, high) in band_limits.items():
            if low >= freqs[0] and high <= freqs[-1]:
                axes[1].axhline(y=low, color='white', linestyle='--', alpha=0.5, linewidth=0.5)
                axes[1].text(times[0] + 0.5, (low + high) / 2, band,
                           color='white', fontsize=8, va='center')

        plt.colorbar(im, ax=axes[1], label='Power (dB)')
        plt.tight_layout()

        return fig

    def plot_all_channels_wavelet(self, freqs: Optional[np.ndarray] = None,
                                  figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
        """
        Plot Morlet wavelet for all EEG channels.

        Args:
            freqs: Frequencies to analyze
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if freqs is None:
            freqs = np.arange(1, 41, 0.5)

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for idx, (ax, ch) in enumerate(zip(axes, self.channels)):
            power, freqs_out, times = self.compute_morlet_wavelet(idx, freqs)

            im = ax.pcolormesh(times, freqs_out, 10 * np.log10(power + 1e-10),
                              shading='gouraud', cmap='jet')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            ax.set_title(f'{ch} - Morlet Wavelet Spectrogram')
            plt.colorbar(im, ax=ax, label='Power (dB)')

        plt.suptitle('Morlet Wavelet Analysis - All Channels', fontsize=14)
        plt.tight_layout()

        return fig

    # ==================== ICA ANALYSIS ====================

    def compute_ica(self, n_components: Optional[int] = None,
                   method: str = 'fastica') -> Tuple[ICA, mne.io.RawArray]:
        """
        Compute ICA decomposition on EEG data.

        Args:
            n_components: Number of ICA components (default: n_channels)
            method: ICA method ('fastica', 'infomax', 'picard')

        Returns:
            Tuple of (ICA object, MNE Raw object)
        """
        if self.eeg_data is None:
            raise ValueError("No EEG data loaded. Call load_data() first.")

        if n_components is None:
            n_components = len(self.channels)

        # Create MNE Info and RawArray
        info = mne.create_info(
            ch_names=self.channels,
            sfreq=self.sampling_rate,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(self.eeg_data, info, verbose=False)

        # Set up montage with Muse 2 electrode positions
        # Muse 2 electrodes: TP9 (left ear), AF7 (left forehead), AF8 (right forehead), TP10 (right ear)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='warn', verbose=False)

        # Apply bandpass filter for ICA
        raw_filtered = raw.copy()
        raw_filtered.filter(l_freq=1.0, h_freq=40.0, verbose=False)

        # Fit ICA
        ica = ICA(n_components=n_components, method=method,
                  random_state=42, max_iter=500, verbose=False)
        ica.fit(raw_filtered)

        return ica, raw

    def plot_ica_topomaps(self, n_components: Optional[int] = None,
                         figsize: Tuple[int, int] = (12, 3)) -> plt.Figure:
        """
        Plot ICA component topographic maps (like the scalp maps shown in the reference image).

        Args:
            n_components: Number of ICA components
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        ica, raw = self.compute_ica(n_components)

        # Use MNE's built-in ICA plotting for topomaps
        fig = ica.plot_components(picks=range(ica.n_components_),
                                  show=False, title='ICA components')

        return fig

    def plot_ica_sources(self, n_components: Optional[int] = None,
                        start: float = 0, stop: float = 10,
                        figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """
        Plot ICA source time courses using MNE's visualization.

        Args:
            n_components: Number of ICA components
            start: Start time in seconds
            stop: Stop time in seconds
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        ica, raw = self.compute_ica(n_components)

        # Plot ICA sources
        fig = ica.plot_sources(raw, start=start, stop=stop, show=False)

        return fig

    def plot_ica_components(self, n_components: Optional[int] = None,
                           figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot ICA component time courses and topographies.

        Args:
            n_components: Number of ICA components
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        ica, raw = self.compute_ica(n_components)

        # Get ICA sources
        sources = ica.get_sources(raw).get_data()
        times = self.eeg_times if self.eeg_times is not None else np.arange(sources.shape[1]) / self.sampling_rate

        n_comps = sources.shape[0]
        fig, axes = plt.subplots(n_comps, 2, figsize=figsize,
                                gridspec_kw={'width_ratios': [3, 1]})

        for i in range(n_comps):
            # Time course
            axes[i, 0].plot(times, sources[i, :], 'b-', linewidth=0.5)
            axes[i, 0].set_ylabel(f'IC{i+1}')
            axes[i, 0].set_xlim(times[0], times[-1])
            axes[i, 0].grid(True, alpha=0.3)

            if i == 0:
                axes[i, 0].set_title('ICA Component Time Courses')
            if i == n_comps - 1:
                axes[i, 0].set_xlabel('Time (s)')

            # Component weights (topography proxy)
            weights = ica.get_components()[:, i]
            axes[i, 1].bar(self.channels, weights, color='steelblue')
            axes[i, 1].set_ylabel('Weight')
            axes[i, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            if i == 0:
                axes[i, 1].set_title('Channel Weights')

        plt.suptitle('Independent Component Analysis (ICA)', fontsize=14)
        plt.tight_layout()

        return fig

    def plot_ica_sources_spectrum(self, n_components: Optional[int] = None,
                                  figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot power spectrum of ICA components.

        Args:
            n_components: Number of ICA components
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        ica, raw = self.compute_ica(n_components)
        sources = ica.get_sources(raw).get_data()

        n_comps = sources.shape[0]
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for i in range(min(n_comps, 4)):
            # Compute PSD
            freqs, psd = signal.welch(sources[i, :], fs=self.sampling_rate,
                                      nperseg=min(256, len(sources[i, :]) // 4))

            axes[i].semilogy(freqs, psd, 'b-', linewidth=1)
            axes[i].set_xlabel('Frequency (Hz)')
            axes[i].set_ylabel('Power Spectral Density')
            axes[i].set_title(f'IC{i+1} Power Spectrum')
            axes[i].set_xlim(0, 45)
            axes[i].grid(True, alpha=0.3)

            # Mark frequency bands
            for band, (low, high, color) in {
                'Delta': (0.5, 4, 'red'),
                'Theta': (4, 8, 'orange'),
                'Alpha': (8, 13, 'green'),
                'Beta': (13, 30, 'blue'),
                'Gamma': (30, 45, 'purple')
            }.items():
                axes[i].axvspan(low, high, alpha=0.1, color=color)

        plt.suptitle('ICA Components - Power Spectral Density', fontsize=14)
        plt.tight_layout()

        return fig

    # ==================== HRV ANALYSIS ====================

    def compute_hrv_metrics(self) -> pd.DataFrame:
        """
        Compute HRV metrics from HR data.

        Returns:
            DataFrame with HRV metrics
        """
        if self.hr_data is None or len(self.hr_data) == 0:
            raise ValueError("No HR data loaded. Call load_data() first.")

        # Extract all RR intervals
        all_rr = []
        for rr_list in self.hr_data['rr_intervals']:
            if isinstance(rr_list, str):
                # Parse string representation
                rr_list = [float(x) for x in rr_list.split(';') if x.strip()]
            if isinstance(rr_list, list) and len(rr_list) > 0:
                all_rr.extend(rr_list)

        if len(all_rr) < 2:
            print("Warning: Not enough RR intervals for HRV analysis")
            return pd.DataFrame()

        rr_array = np.array(all_rr)

        # Time domain metrics
        metrics = {
            'Mean RR (ms)': np.mean(rr_array),
            'Mean HR (bpm)': 60000 / np.mean(rr_array),
            'SDNN (ms)': np.std(rr_array, ddof=1),
            'RMSSD (ms)': np.sqrt(np.mean(np.diff(rr_array) ** 2)),
            'NN50': np.sum(np.abs(np.diff(rr_array)) > 50),
            'pNN50 (%)': (np.sum(np.abs(np.diff(rr_array)) > 50) / len(np.diff(rr_array))) * 100,
            'CV (%)': (np.std(rr_array, ddof=1) / np.mean(rr_array)) * 100,
            'Total RR intervals': len(rr_array)
        }

        return pd.DataFrame([metrics])

    def compute_sliding_hrv(self, window_size: float = 30.0,
                           step_size: float = 5.0) -> pd.DataFrame:
        """
        Compute HRV metrics using sliding windows.

        Args:
            window_size: Window size in seconds
            step_size: Step size in seconds

        Returns:
            DataFrame with time-varying HRV metrics
        """
        if self.hr_data is None or len(self.hr_data) == 0:
            raise ValueError("No HR data loaded. Call load_data() first.")

        # Build RR interval time series with timestamps
        rr_times = []
        rr_values = []

        for _, row in self.hr_data.iterrows():
            time = row['time']
            rr_list = row['rr_intervals']

            if isinstance(rr_list, str):
                rr_list = [float(x) for x in rr_list.split(';') if x.strip()]

            if isinstance(rr_list, list):
                for rr in rr_list:
                    rr_times.append(time)
                    rr_values.append(rr)
                    time += rr / 1000  # Advance time by RR interval

        if len(rr_values) < 10:
            print("Warning: Not enough RR intervals for sliding window analysis")
            return pd.DataFrame()

        rr_times = np.array(rr_times)
        rr_values = np.array(rr_values)

        # Sliding window analysis
        results = []
        start_time = rr_times[0]
        end_time = rr_times[-1]

        current_time = start_time
        while current_time + window_size <= end_time:
            mask = (rr_times >= current_time) & (rr_times < current_time + window_size)
            window_rr = rr_values[mask]

            if len(window_rr) >= 5:
                diff_rr = np.diff(window_rr)

                metrics = {
                    'time': current_time + window_size / 2,  # Center of window
                    'mean_rr': np.mean(window_rr),
                    'mean_hr': 60000 / np.mean(window_rr),
                    'sdnn': np.std(window_rr, ddof=1) if len(window_rr) > 1 else 0,
                    'rmssd': np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else 0,
                    'pnn50': (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0,
                    'n_intervals': len(window_rr)
                }
                results.append(metrics)

            current_time += step_size

        return pd.DataFrame(results)

    def plot_hr_analysis(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Plot comprehensive HR and HRV analysis.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if self.hr_data is None or len(self.hr_data) == 0:
            raise ValueError("No HR data loaded. Call load_data() first.")

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # 1. Heart Rate over time
        times = self.hr_data['time'].values
        hr = self.hr_data['heart_rate'].values

        axes[0, 0].plot(times, hr, 'r-', linewidth=1, marker='o', markersize=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Heart Rate (bpm)')
        axes[0, 0].set_title('Heart Rate Over Time')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Heart Rate Distribution
        axes[0, 1].hist(hr, bins=20, color='coral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(hr), color='red', linestyle='--',
                          label=f'Mean: {np.mean(hr):.1f} bpm')
        axes[0, 1].set_xlabel('Heart Rate (bpm)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Heart Rate Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Extract RR intervals
        all_rr = []
        for rr_list in self.hr_data['rr_intervals']:
            if isinstance(rr_list, str):
                rr_list = [float(x) for x in rr_list.split(';') if x.strip()]
            if isinstance(rr_list, list) and len(rr_list) > 0:
                all_rr.extend(rr_list)

        if len(all_rr) > 1:
            rr_array = np.array(all_rr)

            # 3. RR Interval Distribution
            axes[1, 0].hist(rr_array, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            axes[1, 0].axvline(np.mean(rr_array), color='blue', linestyle='--',
                              label=f'Mean: {np.mean(rr_array):.1f} ms')
            axes[1, 0].set_xlabel('RR Interval (ms)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('RR Interval Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Poincare Plot
            if len(rr_array) > 1:
                axes[1, 1].scatter(rr_array[:-1], rr_array[1:],
                                  alpha=0.5, c='green', s=10)
                axes[1, 1].plot([min(rr_array), max(rr_array)],
                               [min(rr_array), max(rr_array)],
                               'r--', alpha=0.5, label='Identity line')
                axes[1, 1].set_xlabel('RR(n) (ms)')
                axes[1, 1].set_ylabel('RR(n+1) (ms)')
                axes[1, 1].set_title('Poincare Plot')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_aspect('equal')

        # 5 & 6. Sliding window HRV metrics
        try:
            sliding_hrv = self.compute_sliding_hrv(window_size=30, step_size=5)

            if len(sliding_hrv) > 0:
                # RMSSD over time
                axes[2, 0].plot(sliding_hrv['time'], sliding_hrv['rmssd'],
                               'b-', linewidth=1.5, marker='o', markersize=3)
                axes[2, 0].fill_between(sliding_hrv['time'], 0, sliding_hrv['rmssd'],
                                        alpha=0.3, color='blue')
                axes[2, 0].set_xlabel('Time (s)')
                axes[2, 0].set_ylabel('RMSSD (ms)')
                axes[2, 0].set_title('RMSSD Over Time (30s sliding window)')
                axes[2, 0].grid(True, alpha=0.3)

                # SDNN over time
                axes[2, 1].plot(sliding_hrv['time'], sliding_hrv['sdnn'],
                               'g-', linewidth=1.5, marker='o', markersize=3)
                axes[2, 1].fill_between(sliding_hrv['time'], 0, sliding_hrv['sdnn'],
                                        alpha=0.3, color='green')
                axes[2, 1].set_xlabel('Time (s)')
                axes[2, 1].set_ylabel('SDNN (ms)')
                axes[2, 1].set_title('SDNN Over Time (30s sliding window)')
                axes[2, 1].grid(True, alpha=0.3)
        except Exception as e:
            print(f"Could not compute sliding HRV: {e}")

        plt.suptitle('Heart Rate Variability Analysis', fontsize=14)
        plt.tight_layout()

        return fig

    def plot_hrv_metrics_over_time(self, window_size: float = 30.0,
                                   step_size: float = 5.0,
                                   figsize: Tuple[int, int] = (14, 12)) -> plt.Figure:
        """
        Plot detailed HRV metrics over time.

        Args:
            window_size: Window size in seconds
            step_size: Step size in seconds
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        sliding_hrv = self.compute_sliding_hrv(window_size, step_size)

        if len(sliding_hrv) == 0:
            raise ValueError("Not enough data for sliding window HRV analysis")

        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

        times = sliding_hrv['time']

        # 1. Mean HR
        axes[0].plot(times, sliding_hrv['mean_hr'], 'r-', linewidth=1.5)
        axes[0].fill_between(times, sliding_hrv['mean_hr'].min(), sliding_hrv['mean_hr'],
                            alpha=0.3, color='red')
        axes[0].set_ylabel('Mean HR (bpm)')
        axes[0].set_title(f'Heart Rate Variability Metrics Over Time ({window_size}s window, {step_size}s step)')
        axes[0].grid(True, alpha=0.3)

        # 2. RMSSD
        axes[1].plot(times, sliding_hrv['rmssd'], 'b-', linewidth=1.5)
        axes[1].fill_between(times, 0, sliding_hrv['rmssd'], alpha=0.3, color='blue')
        axes[1].set_ylabel('RMSSD (ms)')
        axes[1].grid(True, alpha=0.3)

        # Add interpretation zone
        axes[1].axhspan(20, 50, alpha=0.1, color='green', label='Normal range')
        axes[1].legend(loc='upper right')

        # 3. SDNN
        axes[2].plot(times, sliding_hrv['sdnn'], 'g-', linewidth=1.5)
        axes[2].fill_between(times, 0, sliding_hrv['sdnn'], alpha=0.3, color='green')
        axes[2].set_ylabel('SDNN (ms)')
        axes[2].grid(True, alpha=0.3)

        # 4. pNN50
        axes[3].plot(times, sliding_hrv['pnn50'], 'm-', linewidth=1.5)
        axes[3].fill_between(times, 0, sliding_hrv['pnn50'], alpha=0.3, color='magenta')
        axes[3].set_ylabel('pNN50 (%)')
        axes[3].set_xlabel('Time (s)')
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        return fig

    # ==================== COMBINED ANALYSIS ====================

    def generate_full_report(self, output_dir: Optional[str] = None,
                            show_plots: bool = True) -> None:
        """
        Generate a full analysis report with all visualizations.

        Args:
            output_dir: Directory to save figures (optional)
            show_plots: Whether to display plots
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("BCI DATA ANALYSIS REPORT")
        print("=" * 70)

        # Print summary statistics
        if self.eeg_data is not None:
            print(f"\nEEG Data Summary:")
            print(f"  Channels: {self.channels}")
            print(f"  Samples: {self.eeg_data.shape[1]}")
            print(f"  Duration: {self.eeg_data.shape[1] / self.sampling_rate:.1f} seconds")
            print(f"  Sampling Rate: {self.sampling_rate} Hz")

        if self.hr_data is not None and len(self.hr_data) > 0:
            print(f"\nHR Data Summary:")
            print(f"  Samples: {len(self.hr_data)}")
            print(f"  Mean HR: {self.hr_data['heart_rate'].mean():.1f} bpm")
            print(f"  HR Range: {self.hr_data['heart_rate'].min()}-{self.hr_data['heart_rate'].max()} bpm")

            try:
                hrv_metrics = self.compute_hrv_metrics()
                if len(hrv_metrics) > 0:
                    print("\nHRV Metrics:")
                    for col in hrv_metrics.columns:
                        print(f"  {col}: {hrv_metrics[col].values[0]:.2f}")
            except Exception as e:
                print(f"  Could not compute HRV metrics: {e}")

        print("\n" + "=" * 70)
        print("Generating visualizations...")
        print("=" * 70)

        # Generate EEG visualizations
        if self.eeg_data is not None:
            try:
                print("\n1. Morlet Wavelet Analysis (all channels)...")
                fig1 = self.plot_all_channels_wavelet()
                if output_dir:
                    fig1.savefig(output_dir / 'morlet_wavelet_all_channels.png', dpi=150)

                print("2. ICA Topomaps (scalp maps)...")
                fig2 = self.plot_ica_topomaps()
                if output_dir:
                    fig2.savefig(output_dir / 'ica_topomaps.png', dpi=150)

                print("3. ICA Components (time courses)...")
                fig3 = self.plot_ica_components()
                if output_dir:
                    fig3.savefig(output_dir / 'ica_components.png', dpi=150)

                print("4. ICA Power Spectra...")
                fig4 = self.plot_ica_sources_spectrum()
                if output_dir:
                    fig4.savefig(output_dir / 'ica_power_spectra.png', dpi=150)

            except Exception as e:
                print(f"Error in EEG analysis: {e}")

        # Generate HR visualizations
        if self.hr_data is not None and len(self.hr_data) > 0:
            try:
                print("5. HR/HRV Analysis...")
                fig5 = self.plot_hr_analysis()
                if output_dir:
                    fig5.savefig(output_dir / 'hr_analysis.png', dpi=150)

                print("6. HRV Metrics Over Time...")
                fig6 = self.plot_hrv_metrics_over_time()
                if output_dir:
                    fig6.savefig(output_dir / 'hrv_over_time.png', dpi=150)

            except Exception as e:
                print(f"Error in HR analysis: {e}")

        if output_dir:
            print(f"\nFigures saved to: {output_dir}")

        if show_plots:
            plt.show()


def main():
    """Main function to demonstrate data extraction and visualization."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract and visualize BCI experiment data')
    parser.add_argument('data_path', type=str, nargs='?',
                       help='Path to session data (JSON or base path)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for figures')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (only save)')

    args = parser.parse_args()

    if args.data_path is None:
        # Look for data in the data directory
        data_dir = Path('data')
        if data_dir.exists():
            # Find the most recent session
            json_files = list(data_dir.rglob('*.json'))
            if json_files:
                args.data_path = str(max(json_files, key=lambda x: x.stat().st_mtime))
                print(f"Using most recent data file: {args.data_path}")
            else:
                print("No data files found in 'data' directory.")
                print("Usage: python extractData.py <path_to_session_data>")
                print("Example: python extractData.py data/participant1/session_20240101_120000")
                return
        else:
            print("No data directory found.")
            print("Usage: python extractData.py <path_to_session_data>")
            return

    # Create extractor and load data
    extractor = DataExtractor(args.data_path)
    extractor.load_data()

    # Generate report
    extractor.generate_full_report(
        output_dir=args.output,
        show_plots=not args.no_show
    )


if __name__ == "__main__":
    main()
