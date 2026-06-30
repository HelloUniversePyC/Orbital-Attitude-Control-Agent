import numpy as np

def quaternion_conjugate(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
def random_unit_quaternion() -> np.ndarray:
    q = np.random.randn(4)
    return q/np.linalg.norm(q)

def omega_matrix(omega: np.ndarray)-> np.ndarray:
    wx, wy, wz = omega
    Omega = np.array([
        [ 0.0, -wx,  -wy,  -wz],
        [  wx,  0.0,   wz,  -wy],
        [  wy, -wz,   0.0,   wx],
        [  wz,  wy,  -wx,   0.0]
    ])
    return Omega

def quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    q_target_conj: np.ndarray = quaternion_conjugate(q_target)
    q_error: np.ndarray = quaternion_multiply(q_target_conj, q_current)
    return q_error

def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_conj: np.ndarray = quaternion_conjugate(q)
    rotation: np.ndarray = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return rotation[1:]

