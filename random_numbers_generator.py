import random

def is_prime(n, k=40):
    """Miller-Rabin primality test."""
    #Based on Fermat's Little Thoerem: a^(n-1) = 1 mod n

    
    if n == 2 or n == 3: return True
    if n < 2 or n % 2 == 0: return False
    
    # Write n-1 as d * 2^r
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
        
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def generate_prime(bits):
    """Generates a random prime of the exact specified bit length."""
    while True:
        # Generate random bits and ensure the highest and lowest bits are 1
        p = random.getrandbits(bits)
        p |= (1 << (bits - 1)) | 1
        if is_prime(p):
            return p

# def generate_dlp_parameters(q_bits, p_bits):
#     """
#     Generates cryptographic parameters for DLP: g^x = h (mod p).
#     - q: Subgroup order (q_bits)
#     - p: Prime modulus (p_bits) where q divides (p-1)
#     - g: Generator of the subgroup of order q
#     - x: The secret exponent
#     - h: The target value
#     """
#     # 1. Generate the prime subgroup order q
#     q = generate_prime(q_bits)
    
#     # 2. Generate prime modulus p such that p = k*q + 1
#     k_bits = p_bits - q_bits
#     while True:
#         k = random.getrandbits(k_bits)
#         # Force k to be even so that k*q + 1 is odd
#         k = (k | (1 << (k_bits - 1))) & ~1 
#         if k == 0: continue
        
#         p = k * q + 1
#         if p.bit_length() == p_bits and is_prime(p):
#             break
            
#     # 3. Find a generator g for the subgroup of order q
#     while True:
#         # Pick a random element from the multiplicative group Z_p^*
#         h_temp = random.randrange(2, p - 1)
#         # Project it into the subgroup of order q using Lagrange's theorem
#         g = pow(h_temp, (p - 1) // q, p)
#         if g != 1:
#             break
            
#     # 4. Generate the secret x and the target h
#     x = random.randrange(1, q)
#     h = pow(g, x, p)
    
#     return p, q, g, x, h

def generate_dlp_parameters(q_bits, p_bits):
    """
    Generates cryptographic parameters for DLP: g^x = h (mod p).
    """
    k_bits = p_bits - q_bits
    
    # NEW LOOP STRUCTURE: If we can't find a p, we draw a fresh q.
    while True:
        # 1. Generate the prime subgroup order q
        q = generate_prime(q_bits)
        
        # 2. Try to generate prime modulus p such that p = k*q + 1
        p_found = False
        
        # We try 100 times to find a valid k. 
        # If we fail, the outer loop gives us a completely new q to try.
        for _ in range(100):
            k = random.getrandbits(k_bits)
            # Force k to be even so that k*q + 1 is odd, and ensure bit length
            k = (k | (1 << (k_bits - 1))) & ~1 
            if k == 0: 
                continue
            
            p = k * q + 1
            if p.bit_length() == p_bits and is_prime(p):
                p_found = True
                break
                
        if p_found:
            break  # Successfully found both q and p!
            
    # 3. Find a generator g for the subgroup of order q
    while True:
        # Pick a random element from the multiplicative group Z_p^*
        h_temp = random.randrange(2, p - 1)
        # Project it into the subgroup of order q using Lagrange's theorem
        g = pow(h_temp, (p - 1) // q, p)
        if g != 1:
            break
            
    # 4. Generate the secret x and the target h
    x = random.randrange(1, q)
    h = pow(g, x, p)
    
    return p, q, g, x, h

# --- Example Usage for the UI / Testing ---
if __name__ == "__main__":
    # Test on a highly feasible 32-bit subgroup
    p, q, g, secret_x, h = generate_dlp_parameters(q_bits=32, p_bits=64)
    print(f"--- 32-bit Subgroup Test ---")
    print(f"Modulus p: {p}")
    print(f"Subgroup q: {q}")
    print(f"Generator g: {g}")
    print(f"Target h: {h}")
    print(f"Secret x (Keep Hidden from Solver!): {secret_x}\n")