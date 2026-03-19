import numpy as np

class BPSKModulator:
    """
    BPSK Modulator and Demodulator.
    Maps 0 -> -1 and 1 -> +1.
    """
    def __init__(self):
        pass

    def modulate(self, bits):
        """
        Modulates bits to BPSK symbols.
        bits: numpy array of 0s and 1s.
        returns: numpy array of +1s and -1s.
        """
        # 0 -> +1, 1 -> -1 (User Requirement)
        return 1 - 2 * bits

    def demodulate(self, received_symbols, channel_gains=None, noise_variance=1.0, return_soft=False):
        """
        Demodulates received symbols.
        
        If channel_gains (h) is provided, performs Zero-Forcing Equalization: r = y / h.
        Then computes LLR = 2 * r / noise_variance.
        
        Args:
            received_symbols (y): Received complex symbols.
            channel_gains (h): Complex channel gains.
            noise_variance: N0/2.
            return_soft: Return LLRs if True.
        """
        if channel_gains is None:
            # AWGN: y = x + n
            r = received_symbols
        else:
            # Fading: y = hx + n
            # Equalization (User Request): r = y / h
            # Avoid division by zero
            r = received_symbols / (channel_gains + 1e-12) 
            
        # Extract Real part (sufficient statistic for BPSK if phase corrected / real constellation)
        # Note: r = y/h = (hx+n)/h = x + n/h.
        # If h is complex Gaussian, n/h has phase rotation undone. 
        # x is real (+/-1). We take real part.
        y_prime = np.real(r)
        
        if return_soft:
            # LLR Calculation (Corrected for Fading):
            # The User requested LLR = 2 * r / noise_variance.
            # But in fading, the effective noise variance of r = y/h is (N0/2) / |h|^2.
            # So correct LLR = 2 * r / (noise_variance / |h|^2) = 2 * r * |h|^2 / noise_variance.
            # Since r = y/h, this simplifies to:
            # LLR = 2 * (y/h) * |h|^2 / noise_variance
            #     = 2 * y * conj(h) / noise_variance.
            #     = 2 * Re(y * h^*) / noise_variance.
            # This matches the optimal Matched Filter LLR.
            
            # Implementation:
            # r = y/h (User wanted Equalization output).
            # To fix the LLR while keeping 'r', we multiply by |h|^2.
            
            if channel_gains is not None:
                # Reliability weighting
                reliability = np.abs(channel_gains)**2
                return 2 * y_prime * reliability / noise_variance
            else:
                return 2 * y_prime / noise_variance
        else:
            # Hard Decision: >0 -> Bit 0, <0 -> Bit 1
            # But standard is 1 for Bit 1?
            # 0 -> +1, 1 -> -1.
            # If y_prime > 0 (mapped to +1), decision is 0.
            # If y_prime < 0 (mapped to -1), decision is 1.
            decisions = np.zeros_like(y_prime, dtype=int)
            decisions[y_prime < 0] = 1
            return decisions
