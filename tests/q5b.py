from otter.test_files import test_case

OK_FORMAT = False

name = "q5b"
points = 2

@test_case(points=None, hidden=False)
def test_q5b_public(env):
    assert 'results_df' in env, 'Expected a DataFrame named `results_df`.'
    results_df = env['results_df']
    required_cols = {'Model', 'Lower Bound', 'Upper Bound', 'Rank', 'Category'}
    assert required_cols.issubset(results_df.columns), f'Missing columns: {required_cols - set(results_df.columns)}'

