from ur5_algoritmos.fk_functions import *

# =========================
# CUATERNIONES
# =========================

def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_multiply(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# =========================
# ERROR DE ORIENTACIÓN
# =========================

def orientation_error(qd, q):
    q_inv = quat_conjugate(q)
    qe = quat_multiply(qd, q_inv)
    return qe[1:]  # parte vectorial

# =========================
# ERROR TOTAL (POSE)
# =========================

def pose_error(xd, x):
    ep = xd[0:3] - x[0:3]
    eo = orientation_error(xd[3:], x[3:])
    return np.hstack((ep, eo))

# =========================
# JACOBIANO NUMÉRICO
# =========================

def numerical_jacobian(fkine, q, TF2xyzquat, delta=1e-6):
    n = len(q)
    J = np.zeros((6, n))

    x = TF2xyzquat(fkine(q))

    for i in range(n):
        dq = np.zeros(n)
        dq[i] = delta

        x_d = TF2xyzquat(fkine(q + dq))

        J[:, i] = (pose_error(x_d, x)) / delta

    return J

# ===================
# CINEMÁTICA INVERSA 
# ===================

def ik_pseudo_step(fkine, TF2xyzquat, q, xd, alpha=0.3):

    # ===== FK =====
    x = TF2xyzquat(fkine(q))

    # ===== ERROR =====
    e = pose_error(xd, x)

    # ===== JACOBIANO =====
    J = numerical_jacobian(fkine, q, TF2xyzquat)

    # ===== PSEUDOINVERSA =====
    J_pinv = np.linalg.pinv(J)

    # ===== UPDATE =====
    dq = alpha * (J_pinv @ e)

    return q + dq

# =================================================
# CINEMÁTICA INVERSA CON DAMPED LEAST SQUARE
# =================================================

def ik_dls_step(fkine, TF2xyzquat, q, xd, lamb=0.05):

    # ===== FK =====
    x = TF2xyzquat(fkine(q))

    # ===== ERROR =====
    e = pose_error(xd, x)

    # ===== JACOBIANO =====
    J = numerical_jacobian(fkine, q, TF2xyzquat)

    # ===== DLS =====
    JT = J.T
    JJ = J @ JT

    dq = JT @ np.linalg.inv(JJ + (lamb**2)*np.eye(6)) @ e

    return q + dq

