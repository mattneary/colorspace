import math
import json
import scipy
import itertools
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import spectral_clustering
from utils import Node, Group, cos_sim

model = SentenceTransformer('all-mpnet-base-v2')

def get_angle_sims(angles):
    forward = (angles[:, None] - angles[None, :]).abs()
    backward = 360. - angles[:, None] + angles[None, :]
    diff = torch.min(forward, backward)
    return diff.cos()

def get_color_sims(ordering, full_range=None):
    full_range = full_range if full_range is not None else len(ordering)
    angles = torch.tensor(ordering) / full_range * 2 * math.pi
    return get_angle_sims(angles)

def color(nodes):
    if len(nodes) > 5:
        raise ValueError('cannot color more than five nodes')

    vectors = torch.tensor([node.vector for node in nodes])
    affinity = cos_sim(vectors, vectors)

    empties = 2
    orders = [x[:-empties] for x in itertools.permutations(list(range(0, len(nodes)+empties)))]

    rs = []
    for order in orders:
      ord_sims = get_color_sims(order, full_range=len(nodes)+empties)
      r = scipy.stats.pearsonr(torch.flatten(affinity), torch.flatten(ord_sims)).statistic
      rs.append((r, order))

    max_corr, max_order = max(rs, key=lambda p: p[0])
    # print('Cluster rho: {}'.format(max_corr))

    for idx, node in zip(max_order, nodes):
        node.angle = idx / (len(nodes) + empties) * 360

    return {node: node.angle for node in nodes}

def cluster(fragments):
    vectors = model.encode(fragments)
    affinity = cos_sim(vectors, vectors)

    num_clusters = min(4, len(fragments))
    clustering = spectral_clustering(affinity, n_clusters=num_clusters)

    groups = []
    for cluster in np.unique(clustering):
        g_fragments = []
        g_vectors = []
        for idx, in_cluster in enumerate(clustering == cluster):
            if in_cluster:
                g_fragments.append(fragments[idx])
                g_vectors.append(vectors[idx])
        groups.append(Group(g_fragments, g_vectors))

    nodes = [Node(group, group.mean) for group in groups]
    coloring = color(nodes)

    # Arrange the fragments w/in groups
    # sorted_nodes = sorted(nodes, key=lambda node: node.angle)
    # lefts = sorted_nodes[-1:] + sorted_nodes[:-1]
    # rights = sorted_nodes[1:] + sorted_nodes[:1]
    # for node, left, right in zip(sorted_nodes, lefts, rights):
    #     node.content.induce_order(left, right)

    return coloring
