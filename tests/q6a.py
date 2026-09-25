from otter.test_files import test_case

OK_FORMAT = False

name = "q6a"
points = 2

@test_case(points=None, hidden=False)
def test_q6a_extract_metrics(extract_style_metrics, STYLE_METRIC_COLUMNS):
    import pandas as pd
    tiny = pd.DataFrame({'conv_metadata': [{'bold_count_a': {'turn_1': 3, 'turn_2': 1}, 'bold_count_b': {'turn_1': 2}, 'header_count_a': {'turn_1': 0}, 'header_count_b': {'turn_1': 1}, 'list_count_a': {'turn_1': 2}, 'list_count_b': {'turn_1': 4}, 'sum_assistant_a_tokens': 120, 'sum_assistant_b_tokens': 80}]})
    out = extract_style_metrics(tiny)
    assert list(out.columns) == STYLE_METRIC_COLUMNS
    assert out.iloc[0].tolist() == [4, 2, 0, 1, 2, 4, 120, 80]
    assert list(tiny.columns) == ['conv_metadata']

