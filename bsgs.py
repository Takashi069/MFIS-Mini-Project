import math
import sys
import tracemalloc

from random_numbers_generator import generate_dlp_parameters


def extended_euclidean(a, b):
    """Extended Euclidean Algorithm to compute gcd(a, b) and Bezout coefficient."""
    t1 = 1
    t2 = 0

    while b != 0:
        q = a // b
        r = a % b

        t3 = t1 - q * t2

        a = b
        b = r
        t1 = t2
        t2 = t3

    return (a, t1)


def mod_inverse(a, m):
    """Find the modular multiplicative inverse a^-1 mod m."""
    a = a % m
    gcd, t1 = extended_euclidean(a, m)

    if gcd != 1:
        raise ValueError(f"{a} does not have an inverse for mod {m}")
    else:
        return t1 % m


def _finish_memory(track_memory):
    """Stop tracing and return the peak, or None if tracing was never started."""
    if not track_memory:
        return None
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def bsgs_dlp(p, q, g, h, track_memory=True):
    """
    Solve g^x = h (mod p) using Baby-Step Giant-Step (BSGS),
    where g generates a prime-order subgroup of order q.

    Returns:
        (x, iterations, memory_bytes)
        - x: The recovered secret exponent (or None if not found)
        - iterations: Total baby + giant steps executed
        - memory_bytes: Peak memory allocated during the algorithm (in bytes),
          or None when track_memory=False

    track_memory: True (default) keeps the original behavior, so this file's own
    test runner works unchanged. The Streamlit app passes False because it times
    and measures memory itself, for both algorithms the same way: tracemalloc
    slows the code down and would distort the timing.
    """
    # Start tracking memory usage for Task (d) comparison
    if track_memory:
        tracemalloc.start()

    # Step 1: Compute step size m = ceil(sqrt(q))
    m = math.isqrt(q)
    if m * m < q:
        m += 1

    iterations = 0

    # ---------------------------------------------------------
    # Step 2: Baby Steps (Store g^j mod p for j = 0, 1, ..., m-1)
    # ---------------------------------------------------------
    baby_steps = {}
    current_baby = 1  # g^0 mod p = 1

    for j in range(m):
        iterations += 1
        # Store only the first occurrence (smallest non-negative exponent)
        if current_baby not in baby_steps:
            baby_steps[current_baby] = j
        current_baby = (current_baby * g) % p

    # ---------------------------------------------------------
    # Step 3: Giant Steps (Compute h * (g^-m)^i mod p)
    # ---------------------------------------------------------
    # Compute g^m mod p, then take its modular inverse to get g^-m mod p
    g_pow_m = pow(g, m, p)
    giant_stride = mod_inverse(g_pow_m, p)

    current_giant = h  # Start at i = 0: h * (g^-m)^0 = h

    for i in range(m):
        iterations += 1
        # Check if current giant step matches any stored baby step
        if current_giant in baby_steps:
            j = baby_steps[current_giant]
            x = i * m + j

            return x, iterations, _finish_memory(track_memory)

        # Take one giant step forward: multiply by g^-m mod p
        current_giant = (current_giant * giant_stride) % p

    return None, iterations, _finish_memory(track_memory)


# ---------------------------------------------------------
# Standalone Test Runner (Same style as pollards_rho.py)
# ---------------------------------------------------------
def test_runner():
    print("--- Baby-Step Giant-Step (BSGS) DLP Test ---")

    p_test, q_test, g_test, x_test, h_test = generate_dlp_parameters(q_bits=24, p_bits=48)

    print(f"Finding x such that {g_test}^x ≡ {h_test} (mod {p_test})")
    print(f"Subgroup Order q = {q_test}")

    x_result, steps, mem_bytes = bsgs_dlp(p_test, q_test, g_test, h_test)

    if x_result is not None:
        print(f"Success in {steps} total steps!")
        print(f"Peak Memory Used: {mem_bytes / 1024:.2f} KB ({mem_bytes} bytes)")
        print(f"Calculated x = {x_result}, Actual secret x = {x_test}")

        # Verify the math using modular exponentiation
        verification = pow(g_test, x_result, p_test)
        print(f"Verification: {g_test}^{x_result} mod {p_test} = {verification}")

        if verification == h_test and x_result == x_test:
            print("Status: PASSED")
        else:
            print("Status: MATH ERROR")
    else:
        print("Status: FAILED (Solution not found)")


if __name__ == "__main__":
    for i in range(1, 6):
        print(f"Test {i}:")
        test_runner()
        print("\n")
