import numpy as np
import osqp
import scipy.sparse as sp
from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import *

def compute_dq_qp(fkine, jacobian_func, TF2xyzquat, q, xd,
                  K=1.5, lamb=0.01, dt=0.02):

    """
    Control cinemático usando QP (OSQP)

    Retorna:
        dq (6,)
    """

    # =========================
    # 1. FK
    # =========================
    x = TF2xyzquat(fkine(q))

    # =========================
    # 2. Error (posición + orientación)
    # =========================
    e = pose_error(xd, x)

    # =========================
    # 3. Jacobiano
    # =========================
    J = jacobian_func(fkine, q, TF2xyzquat)

    # =========================
    # 4. Velocidad deseada
    # =========================
    vd = K * e

    # =========================
    # 5. QP COST
    # =========================
    n = len(q)

    H = 2 * (J.T @ J + lamb * np.eye(n))
    f = -2 * J.T @ vd

    # convertir a sparse
    P = sp.csc_matrix(H)
    q_osqp = f

    # =========================
    # 6. CONSTRAINTS (límites dq)
    # =========================
    
    # =========================
    # LIMITES VELOCIDAD
    # =========================
    dq_max = np.array([np.pi, np.pi, np.pi, np.pi, np.pi, np.pi])
    dq_min = -dq_max

    # =========================
    # LIMITES POSICION
    # =========================
    q_min = -2*np.pi * np.ones(n)
    q_max =  2*np.pi * np.ones(n)

    dq_min_pos = (q_min - q) / dt
    dq_max_pos = (q_max - q) / dt

    # =========================
    # COMBINACION
    # =========================
    dq_min_final = np.maximum(dq_min, dq_min_pos)
    dq_max_final = np.minimum(dq_max, dq_max_pos)

    # =========================
    # OSQP
    # =========================
    A = sp.eye(n)
    l = dq_min_final
    u = dq_max_final

    # =========================
    # 7. SOLVER
    # =========================
    prob = osqp.OSQP()
    prob.setup(P=P, q=q_osqp, A=A, l=l, u=u, verbose=False)

    res = prob.solve()

    if res.info.status != 'solved':
        print("QP no convergió")
        return np.zeros(n)

    dq = res.x

    return dq