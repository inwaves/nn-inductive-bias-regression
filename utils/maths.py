def chebyshev_polynomial(x, n):
    """Returns the nth Chebyshev polynomial of x."""
    if n == 0:
        return 1
    elif n == 1:
        return 2*x
    else:
        return 2 * x * chebyshev_polynomial(x, n - 1) - chebyshev_polynomial(x, n - 2)