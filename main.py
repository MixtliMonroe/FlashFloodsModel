import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

def generate_wetted_perimeter(n, ld):
  if n == 1: # Triangular wedge
    def l(h):
      return 2*np.sqrt(h*(h + 4/ld**2))
  
  elif n == 2: # Parabolic riverbed
    def l(h):
      return (1/2)*(np.sqrt(h*(16*h*ld**2 + 1)) + (1/(4*ld))*np.asinh(4*ld*np.sqrt(h)))
  
  elif (n >= 1e5) or (n == np.inf): # Square riverbed
    def l(h):
      return 2*ld*h + 1
  
  else:
    def l(h, fulloutput = False):
      dLdh = lambda h: (1/n)*h**((1-n)/n)
      L = quad(lambda h: np.sqrt(dLdh(h)**2 + 4*ld**2), 0, h)

      if fulloutput:
        return L
      return L[0]

  return l

def generate_v(l, n, ld):
  def v(h):
    A = (n/(n+1))*h**((n+1)/n)
    return (1/2)*np.sqrt(A/l(h))*(3 - (n/(n+1))*(h/l(h))*np.sqrt((1/n**2)*h*(2*(1-n)/n) + 4*ld**2))
  return v


if __name__ == "__main__":
  n = 1e100
  ld = 1

  l = generate_wetted_perimeter(n=n, ld=ld)
  v = generate_v(l=l, n=n, ld=ld)