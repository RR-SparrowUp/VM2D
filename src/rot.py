#rot.py
import tensorflow as tf
def sixd_to_rotmat(x):
    # Convert a 6D representation to a 3x3 rotation matrix.
    # x shape: (batch, 6) where first 3 values = a1 and next 3 = a2.
    a1 = x[..., :3]
    a2 = x[..., 3:]
    b1 = tf.math.l2_normalize(a1, axis=-1)
    dot = tf.reduce_sum(b1 * a2, axis=-1, keepdims=True)
    a2_ortho = a2 - dot * b1
    b2 = tf.math.l2_normalize(a2_ortho, axis=-1)
    b3 = tf.linalg.cross(b1, b2)
    rotmat = tf.stack([b1, b2, b3], axis=-1)
    return rotmat