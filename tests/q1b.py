from otter.test_files import test_case

OK_FORMAT = False

name = "q1b"
points = 1

@test_case(points=None, hidden=False)
def test_selected_battles_filter(battles, selected_lmarena_models, subselect_battles):
    import pandas as pd
    assert isinstance(battles, pd.DataFrame), '`battles` should be a pandas DataFrame.'
    expected_selected = battles[battles['model_a'].isin(selected_lmarena_models) & battles['model_b'].isin(selected_lmarena_models)]
    expected_no_ties = expected_selected[~expected_selected['winner'].astype(str).str.contains('tie')]
    selected_battles_no_ties = subselect_battles(battles, selected_lmarena_models)
    pd.testing.assert_frame_equal(selected_battles_no_ties.reset_index(drop=True), expected_no_ties.reset_index(drop=True), check_dtype=False)
    assert not selected_battles_no_ties['winner'].astype(str).str.contains('tie', case=False, na=False).any(), '`selected_battles_no_ties` must not contain any ties.'
    assert set(selected_battles_no_ties.columns) == set(battles.columns), '`selected_battles_no_ties` should preserve the same columns as `battles`.'

