import math
import numpy as np

def cos_sim(a, b):
    sims = a @ b.T
    if len(b.shape) == 1:
        sims /= np.linalg.norm(a) * np.linalg.norm(b)
    else:
        sims /= np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return sims

class Node(object):
    def __init__(self, content, vector):
        self.content = content
        self.vector = vector

class Group(object):
    def __init__(self, fragments, vectors):
        self.fragments = fragments
        self.vectors = vectors
        self.mean = np.vstack(vectors).mean(axis=0)

    def induce_order(self, left, right):
        angle_delta = min(abs(right.angle - left.angle), 360 - left.angle + right.angle)
        def get_angle(pair):
            vector = pair[1]
            left_sim = cos_sim(left.vector, vector)
            right_sim = cos_sim(vector, right.vector)
            left_angle = abs(math.acos(left_sim))
            right_angle = abs(math.acos(right_sim))
            return left.angle + left_angle / (left_angle + right_angle) * angle_delta

        pairs = zip(self.fragments, self.vectors)
        self.fragments, self.vectors = zip(*sorted(pairs, key=get_angle))
        self.angles = [get_angle(pair) for pair in zip(self.fragments, self.vectors)]
