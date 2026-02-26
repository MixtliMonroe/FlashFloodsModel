import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

def generate_wetted_perimeter(n, ld):
  def _ensure_array(x):
    scalar = np.isscalar(x)
    arr = np.atleast_1d(x)
    return arr, scalar

  if n == 1: # Triangular wedge
    def l(h):
      return 2*np.sqrt(h*(h + 4/ld**2))
  
  elif n == 2: # Parabolic riverbed
    def l(h):
      return (1/2)*(np.sqrt(h*(16*h*ld**2 + 1)) + (1/(4*ld))*np.asinh(4*ld*np.sqrt(h)))
  
  elif (n >= 1e5) or (n == np.inf): # Square riverbed
    def l(h):
      return 2*ld*h + 1
  
  else: # general n vectorised
    def scalar_l(h_scalar, fulloutput=False):
      dLdh = lambda hh: (1/n)*hh**((1-n)/n)
      res = quad(lambda hh: np.sqrt(dLdh(hh)**2 + 4*ld**2), 0, h_scalar)
      return res if fulloutput else res[0]

    vec_l = np.vectorize(lambda hh: scalar_l(hh, fulloutput=False), otypes=[float])

    def l(h, fulloutput=False):
      h_arr, scalar = _ensure_array(h)
      if fulloutput:
        if scalar:
          return scalar_l(h, fulloutput=True)
        return [scalar_l(float(hh), fulloutput=True) for hh in h_arr]
      L = vec_l(h_arr)
      return L[0] if scalar else L

  return l

def generate_v(l, n, ld):
  def v(h):
    A = (n/(n+1))*h**((n+1)/n)
    return (1/2)*np.sqrt(A/l(h))*(3 - (n/(n+1))*(h/l(h))*np.sqrt((1/n**2)*h*(2*(1-n)/n) + 4*ld**2))
  return v


if __name__ == "__main__":
  n = 1
  ld = 1

  l = generate_wetted_perimeter(n=n, ld=ld)
  v = generate_v(l=l, n=n, ld=ld)

  def h_init(x):
    return np.exp(-x**2)
  
  x0 = np.linspace(-5,5,500)
  h0 = h_init(x0)
  t = np.linspace(0,5,2)

  for x, h in zip(x0, h0):
    plt.plot(v(h)*t+x, t, color="black", lw=.1)
  
  plt.show()