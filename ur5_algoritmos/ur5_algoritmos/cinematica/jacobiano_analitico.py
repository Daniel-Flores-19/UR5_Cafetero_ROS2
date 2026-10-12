"""
Jacobiano analítico del UR5/UR5e (mismos DH que fkine_ur5).

numerical_jacobian necesita 7 evaluaciones de la FK por llamada (~1.9 ms en
Python); este calcula la FK una vez y arma el Jacobiano geométrico con los
ejes z y orígenes de cada eslabón (~0.1 ms).

Convención: devuelve el MISMO Jacobiano que numerical_jacobian, es decir, el
que relaciona dq con pose_error = [dp, parte vectorial de qd ⊗ q⁻¹]. Para un
giro pequeño ω·dt esa parte vectorial vale ω·dt/2, así que la fila de
orientación es la mitad del Jacobiano angular geométrico.
"""
import numpy as np

from ur5_algoritmos.cinematica.fk_functions import DH, TF2xyzquat, dh

# Mismos DH que fkine_ur5 (los del modelo elegido con PINTOR_UR_MODELO).
_D, _A, _ALPHA = DH["d"], DH["a"], DH["alpha"]


def fkine_y_jacobiano(q):
    """(T efector, J 6x6) del UR5 en la configuración q."""
    T = np.eye(4)
    z = [np.array([0.0, 0.0, 1.0])]
    p = [np.zeros(3)]
    for i in range(6):
        T = T @ dh(_D[i], q[i], _A[i], _ALPHA[i])
        z.append(T[0:3, 2].copy())
        p.append(T[0:3, 3].copy())
    pe = p[-1]
    J = np.zeros((6, 6))
    for i in range(6):
        J[0:3, i] = np.cross(z[i], pe - p[i])
        J[3:6, i] = 0.5 * z[i]
    return T, J


def jacobiano_ur5(fkine=None, q=None, TF2xyzquat_fn=None):
    """Firma compatible con numerical_jacobian(fkine, q, TF2xyzquat)."""
    return fkine_y_jacobiano(q)[1]


def dq_seguimiento(q, xd, xd_dot, K, lamb=0.05):
    """
    Control cinemático con avance: dq = J⁺_dls (xd_dot + K e).
    Devuelve (dq, x actual, error de pose).
    """
    from ur5_algoritmos.cinematica.ik_functions import pose_error
    T, J = fkine_y_jacobiano(q)
    x = TF2xyzquat(T)
    e = pose_error(xd, x)
    v = xd_dot + K * e
    JT = J.T
    dq = JT @ np.linalg.solve(J @ JT + (lamb ** 2) * np.eye(6), v)
    return dq, x, e


__all__ = ["fkine_y_jacobiano", "jacobiano_ur5", "dq_seguimiento"]
