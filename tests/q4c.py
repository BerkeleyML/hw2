from otter.test_files import test_case

OK_FORMAT = False

name = "q4c"
points = 3

@test_case(points=None, hidden=False)
def test_q4c_public(selected_battles_no_ties, train_battles, test_battles, q4c_metrics):
    import numpy as np
    assert set(train_battles.index).isdisjoint(test_battles.index), 'Training and test battles must be disjoint.'
    assert len(train_battles) + len(test_battles) == len(selected_battles_no_ties), 'The split must include every selected battle exactly once.'
    assert list(q4c_metrics.columns) == ['Method', 'Accuracy', 'Log Loss']
    assert q4c_metrics['Method'].tolist() == ['Bradley-Terry', 'Pairwise win-rate baseline']
    assert np.isfinite(q4c_metrics[['Accuracy', 'Log Loss']].to_numpy()).all()
    assert q4c_metrics['Accuracy'].between(0, 1).all()
    assert (q4c_metrics['Log Loss'] >= 0).all()

