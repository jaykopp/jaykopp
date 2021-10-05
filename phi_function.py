import sys
import numpy as np

def isPrime(n):
    if n < 2:
        return False
    for i in range(2,int(np.sqrt(n))+1):
        if (n/i) == int(n/i):
            return False
    return True

def factorise(n):
    if n < 1:
        return []
    if isPrime(n):
        return np.array([n])
    factors = []
    for i in range(2,n+1):
        if (n/i) == int(n/i) and isPrime(i):
            factors.append(i)
    return np.array(factors)



def phi(n):
    if n < 0:
        return 0
    product = n
    factors = factorise(n)
    for factor in factors:
        product *= 1-1/factor
    return int(product)

def main():
    print(phi(int(sys.argv[1])))

if __name__ == "__main__":
    main()
