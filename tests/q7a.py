from otter.test_files import test_case

OK_FORMAT = False

name = "q7a"
points = 4

@test_case(points=None, hidden=False)
def test_q7a_custom_feature(custom_style_value, X_custom_style, X):
    import numpy as np
    sample = [{'role': 'user', 'content': 'Explain recursion.'}, {'role': 'assistant', 'content': 'For example, a function can call itself.'}]
    value = custom_style_value(sample)
    assert isinstance(value, (int, float, np.integer, np.floating))
    assert np.isfinite(value) and value >= 0
    assert X_custom_style.shape == (len(X), 1)
    np.testing.assert_allclose(X_custom_style[1::2], -X_custom_style[0::2], atol=1e-12)

