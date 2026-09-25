from otter.test_files import test_case

OK_FORMAT = False

name = "q3b"
points = 4

@test_case(points=None, hidden=False)
def test_q3b_public(proportions, rank_dataframes):
    assert isinstance(proportions, dict)
    assert set(proportions) == {'creativity', 'technical_accuracy', 'instruction_following', 'math', 'is_code'}
    for val in proportions.values():
        assert 0 <= val <= 1
    assert len(rank_dataframes) == 120

