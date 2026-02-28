import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad

def generate_wetted_perimeter(n, ld):
  '''
  Generates a function for the wetted perimeter of a given riverbed shape parametrised by n and aspect ratio lambda.
  '''
  def _ensure_array(x):
    scalar = np.isscalar(x)
    arr = np.atleast_1d(x)
    return arr, scalar

  if n == 1: # Triangular wedge
    def l(h):
      return h*np.sqrt(1 + 4*ld**2)
  
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

def generate_characteristic_gradient(l, n, ld):
  '''
  Generates a function for the gradient of a characteristic for a given riverbed shape parametrised by n, wetted perimeter l, and aspect ratio lambda.
  '''
  def v(h):
    A = (n/(n+1))*h**((n+1)/n)
    return (1/2)*np.sqrt(A/l(h))*(3 - (n/(n+1))*(h/l(h))*np.sqrt((1/n**2)*h**(2*(1-n)/n) + 4*ld**2))
  return v

def plot_characteristic_curves(A_init=lambda x: np.exp(-x**2), n_arr=[1, 2, 5, 1e100], ld_arr=[.1, .5, 1, 2, 3, 4, 5]):
  '''
  Plots the characteristic curves for various n and lambda and for a given initial condition
  '''
  fig, axes = plt.subplots(len(n_arr), len(ld_arr), sharey=True)

  for i, n in enumerate(n_arr):
    for j, ld in enumerate(ld_arr):
      l = generate_wetted_perimeter(n=n, ld=ld)
      v = generate_characteristic_gradient(l=l, n=n, ld=ld)

      def h_init(x):
        return (((n+1)/n) * A_init(x))**(n/(n+1))

      x0 = np.linspace(-5, 5, 200)
      h0 = h_init(x0)
      t = np.linspace(0, 7, 2)

      ax = axes[i, j]
      # plot characteristic lines for each initial height
      for x, h in zip(x0, h0):
        ax.plot(v(h) * t + x, t, color="black", lw=0.5)

      # formatting
      ax.set_xlim(-5, 5)
      ax.set_ylim(0, 7)
      if i == len(n_arr) - 1:
        ax.set_xlabel("x position")
      if j == 0:
        ax.set_ylabel("time t")
      ax.set_title(f"n={n}, λ={ld}")
      ax.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle("Characteristic curves for different shapes and λ")
    fig.set_size_inches(20, 12)
    fig.tight_layout(rect=[0, 0, 1, 1])
    #plt.savefig("img/characteristics.png", dpi=500)


if __name__ == "__main__":
  n  = 1
  ld = 1

  l = generate_wetted_perimeter(n, ld)
  v = generate_characteristic_gradient(l, n, ld)
  
  def v_deriv(h, dh=1e-10):
    return (v(h+dh)-v(h-dh))/(2*dh)


  def A_init(x):
    return np.exp(-x**2)

  def h_init(x):
    return (((n+1)/n) * A_init(x))**(n/(n+1))
  
  def h_init_deriv(h, dh=1e-10):
    return (h_init(h+dh)-h_init(h-dh))/(2*dh)
  
  x = np.linspace(0, 10)