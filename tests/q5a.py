from otter.test_files import test_case

OK_FORMAT = False

name = "q5a"
points = 2

@test_case(points=None, hidden=False)
def test_get_bootstrapped_score_outputs(get_bootstrapped_score, X, y, selected_lmarena_models, SEED):
    """
      - Run get_bootstrapped_score on provided (X, y, selected_lmarena_models).
      - Check shapes of outputs.
      - Check that results_df has expected columns and type.
      - Check that confidence intervals are valid (LB <= UB).
    """
    import numpy as np
    import pandas as pd
    np.random.seed(SEED)
    results_df, mean_scores, confidence_intervals = get_bootstrapped_score(X, y, selected_lmarena_models, n_bootstrap=10)
    assert isinstance(results_df, pd.DataFrame), 'results_df should be a DataFrame.'
    for c in ['Feature', 'Average Score', 'Lower Bound', 'Upper Bound', 'Category']:
        assert c in results_df.columns, f'Missing column: {c}'
    n_models = len(selected_lmarena_models)
    assert results_df.shape[0] == n_models, f'Expected {n_models} rows in results_df, got {results_df.shape[0]}'
    assert mean_scores.shape == (n_models,), f'mean_scores must be shape ({n_models},)'
    assert confidence_intervals.shape == (2, n_models), f'confidence_intervals must be (2, {n_models})'
    lb, ub = confidence_intervals
    assert (lb <= ub).all(), 'Every lower bound must be <= upper bound.'
    assert ((lb <= mean_scores) & (mean_scores <= ub)).all(), 'Each mean score should lie within its CI.'

