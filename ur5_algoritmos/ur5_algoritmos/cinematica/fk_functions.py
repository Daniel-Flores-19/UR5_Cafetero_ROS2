import os

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

# Parámetros DH nominales (d, a, alpha; theta = q) de cada modelo. El modelo se
# elige con la variable de entorno PINTOR_UR_MODELO ("ur5" o "ur5e"), que
# pintor.launch.py fija para todos los nodos con el argumento modelo_robot.
# Por defecto "ur5" (los valores con que se escribió y probó este código).
DH_MODELOS = {
    "ur5": {   # UR5 CB3
        "d":     (0.08916, 0.0, 0.0, 0.10915, 0.09465, 0.0823),
        "a":     (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0),
        "alpha": (np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0),
    },
    "ur5e": {  # UR5e (e-Series)
        "d":     (0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996),
        "a":     (0.0, -0.425, -0.3922, 0.0, 0.0, 0.0),
        "alpha": (np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0),
    },
}

MODELO = os.environ.get("PINTOR_UR_MODELO", "ur5").strip().lower()
if MODELO not in DH_MODELOS:
    raise ValueError(f"PINTOR_UR_MODELO={MODELO!r} no válido; usa {sorted(DH_MODELOS)}")
DH = DH_MODELOS[MODELO]


def fkine_ur5(q):

    '''
    q: numpy array de 6 elementos
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint

    Usa los DH del modelo elegido (DH_MODELOS[MODELO]); el nombre se conserva
    para no tocar a quienes la llaman.
    '''
    T = np.eye(4)
    for i in range(6):
        T = T @ dh(DH["d"][i], q[i], DH["a"][i], DH["alpha"][i])
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
