from otter.test_files import test_case

OK_FORMAT = False

name = "q3c"
points = 1

@test_case(points=None, hidden=False)
def test_q3c_public(svd, X_tfidf, X_reduced, N_SVD_COMPONENTS):
    import numpy as np
    assert X_reduced.shape == (8000, N_SVD_COMPONENTS)
    assert svd.components_.shape == (N_SVD_COMPONENTS, X_tfidf.shape[1])
    assert np.isfinite(X_reduced).all()

