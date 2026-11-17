import math
import random

from random_numbers_generator import generate_dlp_parameters

def extended_euclidean(a, b):
    #creating the table
    #initial values q,a,b,r,t1,t2,t1-q*t2

    # a<b
    t1 = 1
    t2 = 0

    # s1 = 1
    # s2 = 0

    while b!=0:
        q = a//b
        r = a%b

        t3 = t1 - q*t2
        # s3 = s1 - q*s2

        a = b
        b = r
        t1 = t2
        t2 = t3
        # s1 = s2
        # s2 = s3

    # return (a,s1,t1)
    return (a, t1)


def mod_inverse(a,m):
    #find the multiplicative inverse of a in mod m i.e a^-1 mod m
    #This ensures a<m always
    a = a%m
    
    gcd, t1 = extended_euclidean(a,m)

    if (gcd != 1):
        raise ValueError(f"{a} does not have an inverse for mod {m}")
    else:
        # t1%m ensure that the number is always positive
        return t1 % m

def update_state(y, a, b, p, q, g, x):
    # This evaluates which of the 3 paths to take based on y mod 3
    partition = y % 3
    
    if partition == 0:
        y_next = (y * y) % p
        a_next = (2 * a) % (q)
        b_next = (2 * b) % (q)
        
    elif partition == 1:
        y_next = (x * y) % p
        a_next = a
        b_next = (b + 1) % (q)
        
    elif partition == 2:
        y_next = (g * y) % p
        a_next = (a + 1) % (q)
        b_next = b
        
    return (y_next, a_next, b_next)

def pollards_rho_dlp(p, q, g, h, randomize=False):
    # Solve g^k = h (mod p) where g has prime order q.
    # Step 1: Initialization
    # Starting at 0 is standard, meaning y = g^0 * x^0 = 1
    # If randomize is True (used for retries), pick random exponents.
    # Otherwise, start at 0, meaning y = g^0 * x^0 = 1
    if randomize:
        a_0 = random.randint(1, q-1) #exclusive of q as if a_0 is q then q mod q = 0
        b_0 = random.randint(1, q-1) #exlcusive of q as if a_0 is q then q mod q = 0
        y_0 = (pow(g, a_0, p) * pow(h, b_0, p)) % p

    else:
        a_0 = 0
        b_0 = 0
        y_0 = 1

    y_t, a_t, b_t = y_0, a_0, b_0
    y_h, a_h, b_h = y_0, a_0, b_0

    # Step 2: Floyd's Cycle Detection: Tortoise and Hare 
    iterations = 0
    while True:
        # Tortoise moves 1 step
        y_t, a_t, b_t = update_state(y_t, a_t, b_t, p, q, g, h)
        
        # Hare moves 2 steps
        y_h, a_h, b_h = update_state(y_h, a_h, b_h, p, q, g, h)
        y_h, a_h, b_h = update_state(y_h, a_h, b_h, p, q, g, h)

        iterations+=1
        # Check for collision
        if y_t == y_h:
            break
            
    # Step 3: Extract u and v from the collision state
    # u = (a_hare - a_tortoise) mod (q)
    # v = (b_tortoise - b_hare) mod (q)
    u = (a_h - a_t) % q
    v = (b_t - b_h) % q
    
    # Step 4: Division by zero check
    if v == 0:
        # Returning None so that we to know it should restart with another random a, b
        return None, iterations
        
    # Step 5: Solve v * k = u mod (q)
    d, _ = extended_euclidean(v, q) # There is no need for this as the gcd will always be 1 since q is a prime number
    
    # if d == 1:
    # ~Path A: Unique solution~ 

    # since q is prime, there will always be a inverse as gcd is always 1

    v_inv = mod_inverse(v, q)
    k = (v_inv * u) % (q)
    return k, iterations
        
    # else:
    #     # Path B: GCD > 1 (Multiple candidates)
    #     if u % d != 0:
    #         return None  # No valid solution exists for this collision
            
    #     # Reduce the coefficients by the GCD
    #     v_reduced = v // d
    #     u_reduced = u // d
    #     p_reduced = (p - 1) // d
        
    #     # Find the base solution using your helpers
    #     v_inv_reduced = mod_inverse(v_reduced, p_reduced)
    #     k_0 = (v_inv_reduced * u_reduced) % p_reduced
        
    #     # Test all 'd' possible candidates in the original modulo space
    #     for i in range(d):
    #         k_test = (k_0 + i * p_reduced) % (p - 1)
            
    #         # Using Python's built-in modular exponentiation pow(base, exp, mod)
    #         if pow(g, k_test, p) == x:
    #             return k_test
                
    #     return None

    #--------------------------------------

def test_runner():
    print("--- Pollard's Rho DLP Test ---")
    
    # Known working example: 2^77 mod 227 = 200
    # p_test = 227
    # g_test = 2
    # x_test = 200

    p_test, q_test, g_test, x_test, h_test = generate_dlp_parameters(q_bits=24, p_bits=48)
    
    print(f"Finding k such that {g_test}^k ≡ {h_test} (mod {p_test})")
    
    max_retries = 5
    attempt = 0
    k_result = None
    
    while attempt < max_retries:
        # Pass randomize=True on subsequent attempts so it doesn't repeat the same dud path
        randomize_start = (attempt > 0)
        
        k_result,_ = pollards_rho_dlp(p_test, q_test, g_test, h_test, randomize=randomize_start)
        
        if k_result is not None:
            print(f"Success on attempt {attempt + 1}!")
            print(f"Calculated k = {k_result}")
            
            # Verify the math
            verification = pow(g_test, k_result, p_test)
            print(f"Verification: {g_test}^{k_result} mod {p_test} = {verification}")

            print(f"k_result = {k_result}, x_test = {x_test}")
            
            if verification == h_test:
                print("Status: PASSED")
            else:
                print("Status: MATH ERROR")
            break
            
        else:
            print(f"Attempt {attempt + 1} hit a dud collision. Restarting with random values...")
            attempt += 1

    if k_result is None:
        print("Status: FAILED (Exceeded max retries)")

if __name__ == "__main__":
    for i in range(1,10): 
        print(f"Test {i}: ")
        test_runner()
        print("\n\n")