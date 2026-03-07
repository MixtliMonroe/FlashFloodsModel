import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import fsolve, fmin

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
      dLdh = lambda hh: (1/n)*np.power(hh,((1-n)/n))
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
    A = (n/(n+1))*np.power(h,((n+1)/n))
    return (1/2)*np.sqrt(A/l(h))*(3 - (n/(n+1))*(h/l(h))*np.sqrt((1/n**2)*h**(2*(1-n)/n) + 4*ld**2))
  return v

def generate_ode_func(v, dvdP, h0, A, l):
  '''
  Generates the function f(P1, P2, P1_dot, P2_dot)=0 which is satisfied along the flash flood path
  '''

  def f(P1, P2, X):
    P1_dot, P2_dot = X

    v1 = v(h0(P1))
    v2 = v(h0(P2))
    # Speed of the flash flood in terms of characteristic 1
    S_dot1 = (1/(v1 - v2)**2)*(
      -(v1*P2_dot*dvdP(P2) - v2*P1_dot*dvdP(P1))*(P1 - P2)
      + P1_dot*(v1 - v2)**2
      + v1*(P2_dot - P1_dot)*(v1 - v2)
    )
    # Speed of the flash flood in terms of characteristic 2
    S_dot2 = (1/(v1 - v2)**2)*(
      -(v1*P2_dot*dvdP(P2) - v2*P1_dot*dvdP(P1))*(P1 - P2)
      + P2_dot*(v1 - v2)**2
      + v2*(P2_dot - P1_dot)*(v1 - v2)
    )
    # Speed of the flash flood in terms of the continuity equation
    S_dot_cont = (np.sqrt(A(h0(P2))**3/l(h0(P2))) - np.sqrt(A(h0(P1))**3/l(h0(P1))))/(A(h0(P2)) - A(h0(P1)))
    return np.array([S_dot1 - S_dot_cont, S_dot2 - S_dot_cont])
  
  return f

def generate_dqdt(f, X0):
  '''
  Uses f(P1, P2, P1_dot, P2_dot)=0 to find d/dt[(P1, P2)]
  '''

  def dqdt(t, q):
    P1, P2 = q
    q_dot = fsolve(lambda X: f(P1, P2, X), x0=X0)
    if not all(np.isclose(f(P1, P2, q_dot), [0.0, 0.0])):
      print("Bad solve")
    return q_dot
  
  return dqdt

def flashflood_start(dvdP):
  '''
  Finds the start time and start P of the flash flood
  '''
  minimum = fmin(lambda x: -1/dvdP(x), x0=5)
  P_min = minimum[0]
  T_min = -1/dvdP(P_min )
  return T_min, P_min


if __name__ == "__main__":
  # Define initial parameters and functions for late
  n  = 1e100
  ld = 1

  l = generate_wetted_perimeter(n, ld)
  v = generate_characteristic_gradient(l, n, ld)

  def A(h):
    return (n/(n+1))*np.power(h,(n+1)/n)

  def A_init(x):
    return (1/2)*(1-np.tanh(x))

  def h_init(x):
    return (((n+1)/n) * A_init(x))**(n/(n+1))

  def dvdP(x, dx=1e-10):
    return (v(h_init(x+dx)) - v(h_init(x-dx)))/(2*dx)
  
  # Generate functions for the ODE
  f = generate_ode_func(v=v, dvdP=dvdP, h0=h_init, A=A, l=l)
  dqdt = generate_dqdt(f=f, X0=np.array([-1e-10,1e-10]))

  T0, P0 = flashflood_start(dvdP)

  # Solve the ODE
  t_span = (T0, T0 + 5)
  dP = 1e-10
  q0 = np.array([P0-dP, P0+dP])

  sol = solve_ivp(dqdt, t_span, q0, t_eval=np.linspace(t_span[0], t_span[1], 500), method="Radau")

  # Resulting values
  t_values = sol.t
  x_values = sol.y

  plt.plot(x_values[0], sol.t, label="P1")
  plt.plot(x_values[1], sol.t, label="P2")
  plt.legend()
  plt.show()