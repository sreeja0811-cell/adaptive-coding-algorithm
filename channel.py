import numpy as np

class RayleighChannel:
    """
    Simulates a Rayleigh Fading Channel with AWGN.
    """
    def __init__(self):
        pass

    def apply_channel(self, signal, snr_db):
        """
        Applies Rayleigh fading and AWGN to the signal.
        signal: numpy array of transmitted symbols (e.g., +/- 1).
        snr_db: Signal-to-Noise Ratio in dB (Eb/N0).
        returns: received_signal, channel_gains
        """
        num_symbols = len(signal)
        
        # Power of signal
        # For BPSK +/- 1, signal power is 1. Eb = 1.
        signal_power = np.mean(np.abs(signal)**2)
        
        # Calculate noise power based on SNR
        # SNR_linear = Pb / Pn -> Pn = Pb / SNR_linear
        # SNR_db = 10 * log10(SNR_linear)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Generate complex noise
        # Noise power N0 is split between Real and Imaginary parts
        # variance per dimension = N0 / 2
        noise_std = np.sqrt(noise_power / 2.0)
        noise = noise_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        
        # Generate Rayleigh fading coefficients
        # Expected power E[|h|^2] = 1
        # Real and Imaginary parts are independent Gaussian with variance 0.5
        h_std = np.sqrt(0.5)
        h = h_std * (np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols))
        
        # Received signal y = h*x + n
        received_signal = h * signal + noise
        
        return received_signal, h
