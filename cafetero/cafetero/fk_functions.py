import numpy as np
from copy import copy

cos=np.cos; sin=np.sin; pi=np.pi

# =========================
# DENAVIT HATTENBERG
# =========================

def dh(d, theta, a, alpha):
  """
  Calcular la matriz de transformacion homogenea asociada con los parametros
  de Denavit-Hartenberg.
  Los valores d, theta, a, alpha son escalares.
  """
  # Escriba aqui la matriz de transformacion homogenea en funcion de los valores de d, theta, a, alpha
  sth = np.sin(theta)
  cth = np.cos(theta)
  sa  = np.sin(alpha)
  ca  = np.cos(alpha)
  T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                [sth,  ca*cth, -sa*cth, a*sth],
                [0.0,      sa,      ca,     d],
                [0.0,     0.0,     0.0,   1.0]])
  return T

# =========================
# CINEMÁTICA DIRECTA UR5
# =========================

def fkine_ur5(q):

    '''
    q: numpy array de 6 elementos
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint
    '''


    T1 = dh(0.08916, q[0], 0, np.pi/2)
    T2 = dh(0, q[1], -0.425, 0)
    T3 = dh(0, q[2], -0.39225, 0)
    T4 = dh(0.10915, q[3], 0, np.pi/2)
    T5 = dh(0.09465, q[4], 0, -np.pi/2)
    T6 = dh(0.0823, q[5], 0, 0)

    T = T1 @ T2 @ T3 @ T4 @ T5 @ T6

    return T

# =========================
# MATRIZ DE ROTACIÓN A CUATERNIONES
# =========================

def rot2quat(R):
    q = np.zeros(4)
    trace = np.trace(R)

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2,1] - R[1,2]) / s
        q[2] = (R[0,2] - R[2,0]) / s
        q[3] = (R[1,0] - R[0,1]) / s

    else:
        # buscar mayor diagonal
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            q[0] = (R[2,1] - R[1,2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0,1] + R[1,0]) / s
            q[3] = (R[0,2] + R[2,0]) / s

        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
            q[0] = (R[0,2] - R[2,0]) / s
            q[1] = (R[0,1] + R[1,0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1,2] + R[2,1]) / s

        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
            q[0] = (R[1,0] - R[0,1]) / s
            q[1] = (R[0,2] + R[2,0]) / s
            q[2] = (R[1,2] + R[2,1]) / s
            q[3] = 0.25 * s

    return q / (np.linalg.norm(q) + 1e-8)

# =========================
# TF A CUATERNIONES
# =========================

def TF2xyzquat(T):
  """
  Convierte una matriz de transformación en cuaternoines

  Input:
   T -- A Transformación homogénea
  Output:
   X -- El vector de posición y orientación representado en cuaterniones
  """
  quat = rot2quat(T[0:3,0:3])
  res = [T[0,3], T[1,3], T[2,3], quat[0], quat[1], quat[2], quat[3]]
  return np.array(res)
