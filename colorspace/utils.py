import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


def cos_sim(a, b) -> np.ndarray:
    sims = a @ b.T
    if len(b.shape) == 1:
        sims /= np.linalg.norm(a) * np.linalg.norm(b)
    else:
        sims /= np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return sims

class Group:
    def __init__(self, fragments: list[str], vectors: list[np.ndarray]):
        self.fragments = fragments
        self.vectors = vectors
        self.mean = np.vstack(vectors).mean(axis=0)

    def induce_order(self, left: 'Node', right: 'Node'):
        assert left.angle is not None and right.angle is not None
        angle_delta = min(abs(right.angle - left.angle), 360 - left.angle + right.angle)
        def get_angle(pair: tuple, left: 'Node' = left) -> float:
            vector = pair[1]
            left_sim = cos_sim(left.vector, vector)
            right_sim = cos_sim(vector, right.vector)
            left_angle = abs(math.acos(left_sim))
            right_angle = abs(math.acos(right_sim))
            return left.angle + left_angle / (left_angle + right_angle) * angle_delta # type: ignore

        pairs = zip(self.fragments, self.vectors)
        self.fragments, self.vectors = zip(*sorted(pairs, key=get_angle))
        self.angles: list[float] = [get_angle(pair) for pair in zip(self.fragments, self.vectors)]

@dataclass
class Node:
    content: Group
    vector: np.ndarray
    angle: Optional[float] = None

    @property
    def angles_str(self) -> str:
        if self.angle is None:
            return '----- unassigned'
        else:
            out = '----- {:0.2f} centroid'.format(self.angle)
            for angle, fragment in zip(self.content.angles, self.content.fragments):
                out += '\n({:0.2f}) {}'.format(angle, fragment)
            return out
