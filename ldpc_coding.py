import numpy as np
import pyldpc

def get_ldpc_params(n, rate):
    """
    Returns (d_v, d_c) for a regular LDPC code with given rate using pyldpc.
    Rate = 1 - d_v/d_c
    """
    # Target Rates: 1/3, 1/2, 3/4
    # 1/3 = 1 - 3/4.5 (No) -> 1 - 4/6 = 1/3. (d_v=4, d_c=6)
    if abs(rate - 1/3) < 0.05:
        return 4, 6
    # 1/2 = 1 - 3/6. (d_v=3, d_c=6)
    elif abs(rate - 1/2) < 0.05:
        return 3, 6
    # 3/4 = 1 - 3/12. (d_v=3, d_c=12)
    elif abs(rate - 3/4) < 0.05:
        return 3, 12
    # Fallback to rate 1/2
    return 3, 6

def create_ldpc_code(n, d_v=None, d_c=None, rate=None):
    """
    Creates an LDPC code (H and G matrices) using pyldpc.
    
    Args:
        n: Codeword length.
        d_v: Number of variable node connections (column weight). If None, rate must be provided.
        d_c: Number of check node connections (row weight). If None, rate must be provided.
        rate: Target code rate. Used to determine d_v and d_c if they are not provided.
        
    Returns:
        H: Parity check matrix.
        G: Generator matrix.
    """
    if d_v is None or d_c is None:
        if rate is None:
            raise ValueError("Either d_v and d_c must be provided, or rate must be provided.")
        d_v, d_c = get_ldpc_params(n, rate)

    # Use pyldpc to create regular LDPC code
    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    return H, G

def encode(G, message):
    """
    Encodes the message bits using generator matrix G.
    
    Args:
        G: Generator matrix.
        message: Message bits.
        
    Returns:
        coded_bits: Encoded bits.
    """
    # pyldpc.encode source:
    # d = utils.binaryproduct(tG, v)
    # x = (-1) ** d
    # sigma = 10 ** (- snr / 20)
    # e = np.random.randn(*x.shape) * sigma
    # y = x + e
    # return y
    
    # We only want 'd' (the bits).
    
    # pyldpc.make_ldpc returns G but binaryproduct expects G.T?
    # No, make_ldpc returns G (k, n).
    # binaryproduct expects (n, k).
    
    # pyldpc.make_ldpc returns G (n, k) which is the Transposed Generator Matrix tG.
    # binaryproduct(tG, message) computes (tG @ message) % 2.
    # So we should pass G directly.
    coded_bits = pyldpc.utils.binaryproduct(G, message)
    return coded_bits

def decode(H, received_llrs, snr_db):
    """
    Decodes using Belief Propagation.
    
    Args:
        H: Parity check matrix.
        received_llrs: Log-Likelihood Ratios of received bits.
        snr_db: SNR in dB (unused for decoding if LLRs are passed correctly, 
                but kept for interface consistency).
        
    Returns:
        estimated_codeword: Decoded codeword bits (length n).
    """
    # pyldpc.decode expcts received signal 'y' and computes LLR = 2 * y / sigma^2.
    # We already have LLRs computed from the channel.
    # If 0->+1: y~+1, LLR>0 (matches pyldpc expectation).
    # If 0->-1: y~-1, LLR<0 (inverted).
    #
    # Current Modulation (User Requested): 0 -> +1.
    # So LLRs are correct. No negation needed.
    # Why 0.5? pyldpc.decode(y, snr) -> internally computes 2*y/sigma^2.
    # If we pass LLRs as 'y' and set sigma^2=2? 
    # pyldpc.decode(y, snr): sigma = 10**(-snr/20). var = sigma^2. LLR_int = 2*y/var.
    # We want LLR_int = LLR_input.
    # 2 * LLR_input / var = LLR_input => var = 2.
    # sigma = sqrt(2). 
    # We can pass snr such that sigma=sqrt(2).
    # Or simplified: if we pass LLRs directly, we need to bypass internal calculation.
    # But pyldpc doesn't support 'direct LLRs'.
    # It takes 'y'.
    # If we pass y = LLR/2. Then 2*(LLR/2)/1 = LLR. (If var=1).
    # Let's set SNR such that sigma=1 (SNR=0dB? No, 10**0=1).
    # snr_db argument is ignored by our logic? No.
    # 
    # Actually, the previous code was: `pyldpc.decode(H, -0.5 * received_llrs, 0, ...)`
    # snr=0 -> sigma=1 -> var=1.
    # LLR_calc = 2 * Input / 1 = 2 * Input.
    # Input = -0.5 * LLR.
    # LLR_calc = 2 * (-0.5 * LLR) = -LLR.
    #
    # Now we want LLR_calc = +LLR.
    # Input = 0.5 * LLR.
    # LLR_calc = 2 * (0.5 * LLR) = LLR.
    #
    # So pass 0.5 * LLRs.
    
    return pyldpc.decode(H, 0.5 * received_llrs, 0, maxiter=20)

def get_message(G, codeword):
    """
    Extracts the message bits from the decoded codeword.
    
    Args:
        G: Generator matrix (k x n).
        codeword: Decoded codeword (n).
        
    Returns:
        message: Message bits (k).
    """
    # Validated via inspect_ldpc.py: Systematic part is at the beginning.
    k = G.shape[1]
    return codeword[:k]

