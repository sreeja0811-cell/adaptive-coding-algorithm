
import numpy as np

def construct_polar_code(N, K, design_snr_db=0.0):
    """
    Constructs a Polar Code of length N with K information bits.
    Uses Bhattacharyya Bounds for channel reliability ordering (BEC approximation).
    
    Args:
        N: Block length (power of 2).
        K: Number of information bits.
        design_snr_db: Design SNR in dB (for initial Bhattacharyya parameter).
        
    Returns:
        frozen_mask: Boolean array of length N. True if bit is frozen (0), False if info.
    """
    n = int(np.log2(N))
    assert 2**n == N, "N must be a power of 2"
    
    # Bhattacharyya Parameter Initialization
    # For BEC, Z(W) is erasure probability.
    # Initial Z0 roughly corresponds to channel quality. 
    # Let's start with Z0 = 0.5 as a standard initialization.
    z = np.zeros(N)
    
    # Iterative construction of Z vector
    # This computes Z parameters for all synthetic channels W_N^{(i)}
    # Base channel
    z = np.array([0.5])
    
    for _ in range(n):
        z_new = np.zeros(2 * len(z))
        # Upper branch (W-): Z(W-) <= 2*Z(W) - Z(W)^2
        z_new[0::2] = 2 * z - z**2
        # Lower branch (W+): Z(W+) = Z(W)^2
        z_new[1::2] = z**2
        z = z_new

    # Find indices of K best channels (smallest Z)
    # Smaller Z means more reliable channel (better for Info bits)
    sorted_indices = np.argsort(z)
    
    # First K indices are the Best channels (Info)
    info_indices = sorted_indices[:K] 
    
    frozen_mask = np.ones(N, dtype=bool)
    frozen_mask[info_indices] = False # Info bits correspond to best channels
    
    return frozen_mask

def polar_encode_recursive(u):
    """
    Helper for recursive encoding x = u * G_N.
    """
    N = len(u)
    if N == 1:
        return u
    
    # Split into odd/even indices (or upper/lower halves depending on definition)
    # Arikan's transform:
    # x = [u1+u2, u2] for N=2
    # General N: 
    # u_odd = u[0::2]
    # u_even = u[1::2]
    # x = [encode(u_odd+u_even), encode(u_even)] ?
    # Standard:
    # u1 (upper), u2 (lower)
    # x1 = u1 + u2
    # x2 = u2
    
    n_half = N // 2
    u_upper = u[:n_half] # First half
    u_lower = u[n_half:] # Second half
    
    # Combining step first? No, recursion first?
    # G2 = [1 0; 1 1]
    # x = u * G
    # [x1 x2] = [u1 u2] * [1 0; 1 1] = [u1+u2, u2]
    # So we combine THEN recurse?
    # No, usually recurse then combine?
    # Actually simpler:
    # u_upper, u_lower are vectors.
    # x_upper = encode(u_upper + u_lower)
    # x_lower = encode(u_lower)
    # Result is [x_upper, x_lower] ?? No.
    
    # Correct Arikan Standard:
    # 1. Apply Polar Transform recursively
    # x = B_N u (Bit Reversal) * G_N ? 
    # Or just G_N u?
    # Usually we work with bit-reversed indices or natural order.
    # Let's stick to the definition:
    # x = [x', x''] where x' = encode(u' + u'') and x'' = encode(u'')
    # where u = [u', u'']
    
    u_sum = (u_upper + u_lower) % 2
    
    x_first = polar_encode_recursive(u_sum)
    x_second = polar_encode_recursive(u_lower)
    
    return np.concatenate((x_first, x_second))

def polar_encode(info_bits, frozen_mask):
    """
    Encodes K information bits into N code bits using frozen_mask.
    
    Args:
        info_bits: Array of K bits (0 or 1).
        frozen_mask: Boolean array of length N.
        
    Returns:
        codeword: Array of N bits.
    """
    N = len(frozen_mask)
    K = len(info_bits)
    assert np.sum(~frozen_mask) == K
    
    # 1. Place info bits and frozen bits into u
    u = np.zeros(N, dtype=int)
    u[~frozen_mask] = info_bits
    
    # 2. Encode
    # Using the recursive function structure defined above
    return polar_encode_recursive(u)

def polar_decode(llrs, frozen_mask):
    """
    Successive Cancellation (SC) Decoder.
    
    Args:
        llrs: Received Log-Likelihood Ratios (length N).
        frozen_mask: Boolean array of length N.
        
    Returns:
        decoded_info_bits: Array of K bits.
    """
    N = len(llrs)
    n = int(np.log2(N))
    
    # We need to store the decision 'u' vector globally during recursion
    u_hat = np.zeros(N, dtype=int)
    
    def sc_decode_recursive(in_llrs, index_offset):
        """
        Recursive SC decoder.
        in_llrs: LLRs for this stage.
        index_offset: Global index offset for the bits we are decoding.
        
        Returns:
            Decoded partial sums (u bits) for this block.
        """
        curr_N = len(in_llrs)
        
        if curr_N == 1:
            # Leaf: Make a decision
            if frozen_mask[index_offset]:
                bit = 0 # Frozen bit is known 0
            else:
                # Decision: 0 if LLR >= 0, else 1
                if in_llrs[0] >= 0:
                    bit = 0
                else:
                    bit = 1
            
            u_hat[index_offset] = bit
            return np.array([bit])
        
        half_N = curr_N // 2
        
        # Split LLRs
        # Standard Arikan: y1 (upper/odd?), y2 (lower/even?)
        # Indices are usually sequential in the vector.
        # Top half is first branch, Bottom half is second branch.
        l_upper = in_llrs[:half_N]
        l_lower = in_llrs[half_N:]
        
        # 1. Left Child (Function f)
        # f(a,b) = sign(a)*sign(b)*min(|a|, |b|)
        l_left = np.sign(l_upper) * np.sign(l_lower) * np.minimum(np.abs(l_upper), np.abs(l_lower))
        
        # Recurse Left
        u_left = sc_decode_recursive(l_left, index_offset)
        
        # 2. Right Child (Function g)
        # g(a, b, u) = b + (1-2u)a
        # u here is the u_left decisions
        l_right = l_lower + (1 - 2 * u_left) * l_upper
        
        # Recurse Right
        u_right = sc_decode_recursive(l_right, index_offset + half_N)
        
        # Combine partial sums to return up
        # Relationship: x = [u' + u'', u'']
        # Return [u_left + u_right, u_right] % 2
        u_combined = np.concatenate(((u_left + u_right) % 2, u_right))
        
        return u_combined

    # Start Decoding
    sc_decode_recursive(llrs, 0)
    
    # Extract Information Bits
    return u_hat[~frozen_mask]

if __name__ == "__main__":
    # --- Example Usage ---
    print("--- Polar Code Verification ---")
    
    # Parameters
    N = 256
    K = 128
    DesignSNR_dB = 0.0 # Standard approximation point
    
    # 1. Construction
    frozen = construct_polar_code(N, K, DesignSNR_dB)
    print(f"Polar Code Constructed: N={N}, K={K}, Rate={K/N}")
    
    # 2. Random Data
    np.random.seed(42)
    info_bits = np.random.randint(0, 2, K)
    print(f"Original Info Bits (first 16): {info_bits[:16]}")
    
    # 3. Encode
    codeword = polar_encode(info_bits, frozen)
    print(f"Encoded Codeword (first 16):   {codeword[:16]}")
    
    # 4. Channel (BPSK + AWGN)
    # 0 -> +1, 1 -> -1 (Standard Polar)
    tx_signal = 1 - 2 * codeword
    
    # Noise Setup for simulation
    # Let's test at 5 dB (should be clean)
    Test_SNR_dB = 5.0
    snr_lin = 10**(Test_SNR_dB/10.0)
    # Rate adjusted sigma
    sigma = np.sqrt(1 / (2 * (K/N) * snr_lin))
    
    noise = sigma * np.random.randn(N)
    rx_signal = tx_signal + noise
    
    # 5. LLR Calculation
    # LLR = 2y/sigma^2
    llrs = 2 * rx_signal / sigma**2
    
    # 6. Decode
    decoded_bits = polar_decode(llrs, frozen)
    print(f"Decoded Info Bits (first 16):  {decoded_bits[:16]}")
    
    # 7. Check Errors
    errors = np.sum(info_bits != decoded_bits)
    print(f"Bit Errors: {errors}")
    print(f"BER: {errors/K}")
