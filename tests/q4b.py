from otter.test_files import test_case

OK_FORMAT = False

name = "q4b"
points = 2

@test_case(points=None, hidden=False)
def test_q4b_public(q4b_results_df):
    import numpy as np
    assert q4b_results_df.shape == (20, 2), 'result_df shape does not match (20,2)'
    assert list(q4b_results_df.columns) == ['Model', 'Score'], 'results_df does not match column: [Model, Score]'
    scores = q4b_results_df['Score'].to_numpy()
    assert np.all(np.diff(scores) <= 1e-12), 'Scores must be sorted in descending order.'

