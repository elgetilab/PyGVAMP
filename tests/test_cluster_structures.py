"""
Unit tests for per-discovery-cluster representative structure selection.

Covers the logic behind the interactive report's full-atom cluster structures:

1. StateDiscovery.get_cluster_representative_frames picks the medoid of each
   cluster (in the winning embedding space) and maps it back to a frame index
   via the clustering subsample map.
2. The frame -> (trajectory file, in-file frame) arithmetic used by
   generate_cluster_structures to reload that frame full-atom.

Run with: pytest tests/test_cluster_structures.py -v
"""

import numpy as np
import pytest

from pygv.clustering.state_discovery import StateDiscovery


def _make_discovery():
    """StateDiscovery with hand-set internals (bypasses the heavy fit())."""
    sd = StateDiscovery()
    sd.best_k = 2
    sd.chosen_source = 'umap_2'
    # 6 clustered points drawn (subsampled) from a full set of 10 frames.
    sd._cluster_sample_indices = np.array([9, 3, 7, 1, 5, 0])
    # clusters: points 0,1,2 -> 0 ; points 3,4,5 -> 1
    sd.cluster_labels = {2: np.array([0, 0, 0, 1, 1, 1])}
    # 2D coords: cluster 0 near origin, cluster 1 near (10, 10)
    data = np.array([[0.0, 0.0], [0.2, 0.0], [5.0, 5.0],
                     [10.0, 10.0], [10.1, 10.0], [9.9, 10.0]])
    sd.sweep_results = {'umap_2': {'data': data}}
    return sd


def test_representative_is_cluster_medoid_mapped_to_frame():
    sd = _make_discovery()
    reps = sd.get_cluster_representative_frames()
    # cluster 0 centroid ~ (1.73, 1.67): point idx1 (0.2,0) is closest, not the
    #   (5,5) outlier -> frame _cluster_sample_indices[1] == 3
    # cluster 1 centroid ~ (10.0, 10.0): point idx3 (10,10) closest
    #   -> frame _cluster_sample_indices[3] == 1
    assert reps == {0: 3, 1: 1}


def test_representative_empty_before_fit():
    assert StateDiscovery().get_cluster_representative_frames() == {}


def test_no_subsample_uses_identity_indices():
    sd = _make_discovery()
    sd._cluster_sample_indices = np.arange(6)  # no subsampling -> identity
    reps = sd.get_cluster_representative_frames()
    assert reps == {0: 1, 1: 3}


@pytest.mark.parametrize("frame_r,expected", [
    (3, (0, 6)),   # file 0, strided in-file frame 3*stride
    (1, (0, 2)),
    (4, (1, 0)),   # first frame of file 1 (boundary)
    (9, (1, 10)),
])
def test_frame_to_file_mapping(frame_r, expected):
    # boundaries in strided-frame space: file0 = frames 0-3, file1 = frames 4-9
    boundaries = np.array([0, 4, 10])
    stride = 2
    t = int(np.searchsorted(boundaries, frame_r, side='right') - 1)
    in_file = int((frame_r - boundaries[t]) * stride)
    assert (t, in_file) == expected
