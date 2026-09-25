from otter.test_files import test_case

OK_FORMAT = False

name = "q6b"
points = 2

@test_case(points=None, hidden=False)
def test_q6b_style_differences(normalized_style_difference, make_style_feature_matrix, extract_style_metrics):
    import numpy as np
    import pandas as pd
    assert normalized_style_difference(0, 0) == 0
    assert normalized_style_difference(3, 1) == 0.5
    assert normalized_style_difference(1, 3) == -0.5
    tiny = pd.DataFrame({'conv_metadata': [{'bold_count_a': {'turn_1': 3}, 'bold_count_b': {'turn_1': 1}, 'header_count_a': {'turn_1': 0}, 'header_count_b': {'turn_1': 0}, 'list_count_a': {'turn_1': 1}, 'list_count_b': {'turn_1': 3}, 'sum_assistant_a_tokens': 120, 'sum_assistant_b_tokens': 80}]})
    out = make_style_feature_matrix(tiny, extract_style_metrics(tiny))
    np.testing.assert_allclose(out[0], [0.5, 0.0, -0.5, 0.2])
    np.testing.assert_allclose(out[1], -out[0], atol=1e-12)

