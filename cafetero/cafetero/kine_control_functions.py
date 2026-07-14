from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *

def circular_trajectory(t, center, radius, omega):

    x = center[0] + radius * np.cos(omega * t)
    y = center[1] + radius * np.sin(omega * t)
    z = center[2] 

    q = np.array([0, 0, 1, 0])

    xd = np.hstack((x, y, z, q))

    return xd

# =========================
# CONTROL CINEMÁTICO (DLS)
# =========================
def compute_dq(q, xd, K):

    # ===== FK =====
    T = fkine_ur5(q)
    x = TF2xyzquat(T)

    # ===== ERROR =====
    e = pose_error(xd, x)

    # ===== JACOBIANO =====
    J = numerical_jacobian(fkine_ur5, q, TF2xyzquat)
    # ===== DLS =====
    JT = J.T
    JJ = J @ JT
    lamb = 0.05

    J_dls = JT @ np.linalg.inv(JJ + (lamb**2)*np.eye(6))

    # ===== CONTROL =====
    dq = K * (J_dls @ e)

    return dq
