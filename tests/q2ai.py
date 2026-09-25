from otter.test_files import test_case

OK_FORMAT = False

name = "q2ai"
points = 2

@test_case(points=None, hidden=False)
def test_compute_pairwise_win_fraction(compute_pairwise_win_fraction):
    import numpy as np
    import pandas as pd
    battles_df = pd.DataFrame({'model_a': ['A', 'B', 'A', 'A', 'C', 'B', 'C', 'B'], 'model_b': ['B', 'A', 'B', 'C', 'A', 'C', 'B', 'C'], 'winner': ['model_a', 'model_b', 'model_b', 'model_a', 'model_b', 'model_a', 'model_b', 'model_b']})
    df = compute_pairwise_win_fraction(battles_df)
    assert isinstance(df, pd.DataFrame), 'Output must be a pandas DataFrame.'
    assert len(df.index) == len(df.columns) > 0, 'Output must be a non-empty square matrix.'
    assert (df.index == df.columns).all(), 'Row and column labels must match (same models in same order).'
    assert all((pd.api.types.is_float_dtype(dtype) for dtype in df.dtypes)), 'All entries in the returned DataFrame must have a floating-point dtype.'
    vals = df.values
    finite_mask = np.isfinite(vals)
    assert np.all((vals[finite_mask] >= 0) & (vals[finite_mask] <= 1)), 'All finite entries must be within [0, 1].'
    assert df.index.tolist() == ['A', 'B', 'C'], 'Rows and columns must be sorted by decreasing average win fraction.'
    expected = np.array([[np.nan, 2 / 3, 1.0], [1 / 3, np.nan, 2 / 3], [0.0, 1 / 3, np.nan]], dtype=float)
    np.testing.assert_allclose(df.to_numpy(), expected, rtol=1e-08, atol=1e-10, equal_nan=True)

