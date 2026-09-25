from otter.test_files import test_case

OK_FORMAT = False

name = "q1a"
points = 2

@test_case(points=None, hidden=False)
def test_q1a_public(selected_lmarena_models):
    assert isinstance(selected_lmarena_models, list), '`selected_lmarena_models` should be a list of LLM model names.'
    assert len(selected_lmarena_models) == 20, '`selected_lmarena_models` should contain 20 elements'

