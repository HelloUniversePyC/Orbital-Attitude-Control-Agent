import numpy as np

def quaternion_conjugate(q) -> np.ndarray:
    """Calculates the conjugate of a quaternion"""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2) -> np.ndarray:
    """Applies standard formula to multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])
def random_unit_quaternion() -> np.ndarray:
    """Generates a random 4 dimensional unit vector"""
    q = np.random.randn(4)
    return q/np.linalg.norm(q)

def random_unit_vector() -> np.ndarray:
    """Generates a random 3 dimensional unit vector"""
    q = np.random.randn(3)
    return q/np.linalg.norm(q)


def omega_matrix(omega: np.ndarray)-> np.ndarray:
    """Turns 3 dimensional angular velocity (omega) into 4x4 skew symmetric matrix"""
    wx, wy, wz = omega
    omega_arr = np.array([
        [ 0.0, -wx,  -wy,  -wz],
        [  wx,  0.0,   wz,  -wy],
        [  wy, -wz,   0.0,   wx],
        [  wz,  wy,  -wx,   0.0]
    ])
    return omega_arr

def quaternion_error(q_current: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Error between target and projected quaternion used in Reward function"""
    q_target_conj: np.ndarray = quaternion_conjugate(q_target)
    q_error: np.ndarray = quaternion_multiply(q_target_conj, q_current)
    return q_error

def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Function to take a vector and rotate it to the desired quaternion angle and trajectory"""
    q_v = np.array([0.0, v[0], v[1], v[2]])
    q_conj: np.ndarray = quaternion_conjugate(q)
    rotation: np.ndarray = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return rotation[1:]
def rotate_around_axis(sun_vector: np.ndarray, rot_angle: float) -> np.ndarray:
    """Rotates sun vector by one time steps orbital rotation"""
    orbit_normal = np.array([0,0,1])
    rot_angle_cos = np.cos(rot_angle)
    term_1 = sun_vector* rot_angle_cos

    orthog_comp = np.cross(orbit_normal, sun_vector)
    term_2 = orthog_comp*np.sin(rot_angle)

    sun_projection = np.dot(orbit_normal, sun_vector)
    rotation_scale = 1.0 - np.cos(rot_angle)
    total_scale = sun_projection*rotation_scale
    term_3 = orbit_normal* total_scale

    return term_1 + term_2 + term_3




