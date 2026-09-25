from otter.test_files import test_case

OK_FORMAT = False

name = "q3d"
points = 2

@test_case(points=None, hidden=False)
def test_q3d_public(gmm, gmm_labels, responsibilities, N_COMPONENTS, X_reduced):
    import numpy as np
    assert gmm_labels.shape == (8000,)
    assert responsibilities.shape == (8000, N_COMPONENTS)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1, atol=1e-06)
    np.testing.assert_array_equal(responsibilities.argmax(axis=1), gmm_labels)

