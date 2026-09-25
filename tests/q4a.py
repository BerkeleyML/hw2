from otter.test_files import test_case

OK_FORMAT = False

name = "q4a"
points = 2

@test_case(points=None, hidden=False)
def test_q4a_public(turn_into_features, selected_battles_no_ties, selected_lmarena_models, X):
    X_test, y_test = turn_into_features(selected_battles_no_ties, selected_lmarena_models)
    expected_rows = 2 * len(selected_battles_no_ties)
    expected_cols = len(selected_lmarena_models)
    assert X_test.shape == (expected_rows, expected_cols), f'shape of X does not match ({expected_rows}, {expected_cols})'
    assert y_test.shape == (expected_rows,), f'shape of y does not match ({expected_rows},)'
    assert len(X) == expected_rows, 'Each battle should generate exactly 2 examples'

