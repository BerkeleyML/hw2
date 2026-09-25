from otter.test_files import test_case

OK_FORMAT = False

name = "q6c"
points = 3

@test_case(points=None, hidden=False)
def test_q6c_results(style_model_results, selected_lmarena_models, style_coefficient_results, STYLE_FEATURE_NAMES, style_rank_comparison, fig_q6c):
    assert set(style_model_results['Feature']) == set(selected_lmarena_models)
    assert set(style_coefficient_results['Feature']) == set(STYLE_FEATURE_NAMES)
    assert set(style_rank_comparison.columns) == {'Model', 'Baseline Rank', 'Controlled Rank'}
    assert hasattr(fig_q6c, 'savefig')

