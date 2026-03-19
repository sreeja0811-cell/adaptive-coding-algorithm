import numpy as np

def calculate_ber(transmitted_bits, received_bits):
    """
    Calculates Bit Error Rate.
    """
    errors = np.sum(transmitted_bits != received_bits)
    ber = errors / len(transmitted_bits)
    return ber

def db_to_linear(snr_db):
    return 10**(snr_db / 10.0)

def linear_to_db(snr_linear):
    return 10 * np.log10(snr_linear)
