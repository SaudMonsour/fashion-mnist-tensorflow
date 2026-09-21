"""Verify recorded outputs against source bytes and reloaded TensorFlow weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import train
import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cache', type=Path, default=train.HERE.parents[1] / '.cache' / 'fashion-mnist')
    args = parser.parse_args()
    root = args.project.resolve()
    read = lambda name: json.loads((root / name).read_text())
    config, metrics, execution = read('study_config.json'), read('metrics.json'), read('execution.json')
    train.tf.config.threading.set_intra_op_parallelism_threads(4)
    train.tf.config.threading.set_inter_op_parallelism_threads(1)
    train.tf.config.experimental.enable_op_determinism()
    x, y, test_x, test_y, sources = train.load_data(config, args.cache)
    assert sources == read('data_sources.json')
    keep, exclusions, audit = train.audit_and_clean(x, y, test_x, test_y)
    assert audit == read('audit.json')
    assert len(exclusions) == len(pd.read_csv(root / 'excluded_training_rows.csv'))
    assignment = pd.read_csv(root / 'cv_assignments.csv')
    assert np.array_equal(assignment.source_train_row_id, keep)
    cv = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['seed'])
    for fold, (tr, va) in enumerate(cv.split(keep, y[keep]), 1):
        assert not set(tr).intersection(va)
        assert (assignment.iloc[va].validation_fold == fold).all()
    for record in metrics['leaderboard']:
        assert len(record['fold_scores']) == config['cv_folds']
        assert np.isclose(np.mean(record['fold_scores']), record['cv_mean'], atol=1e-12)
        assert np.isclose(np.std(record['fold_scores']), record['cv_std'], atol=1e-12)
    selection = read('selection.json')
    assert selection['selected_model'] == metrics['selected_model']
    assert selection['leaderboard'] == metrics['leaderboard']
    assert selection['selection_used_test'] is False
    assert metrics['selected_model'] == max(metrics['leaderboard'], key=lambda r: r['cv_mean'])['model']
    logs = pd.read_csv(root / 'training_history.csv')
    cv_logs = logs[logs.phase == 'cv']
    assert len(cv_logs) == 2 * config['cv_folds'] * config['epochs']
    assert (cv_logs.groupby(['model', 'fold']).epoch.max() == config['epochs']).all()
    assert len(logs[logs.phase == 'final']) == config['epochs']
    pred = pd.read_csv(root / 'predictions.csv')
    assert np.array_equal(pred.test_row_id, np.arange(len(test_y)))
    assert np.array_equal(pred.actual, test_y)
    assert np.array_equal(pred.correct, pred.actual == pred.predicted)
    actual = train.scores(test_y, pred.predicted.to_numpy())
    assert actual == metrics['test_metrics'][metrics['selected_model']]
    dummy = np.full(len(test_y), np.bincount(y[keep], minlength=10).argmax())
    assert train.scores(test_y, dummy) == metrics['test_metrics']['Dummy baseline']
    pd.testing.assert_frame_equal(pred[~pred.correct].reset_index(drop=True), pd.read_csv(root / 'error_analysis.csv'))
    assert int((~pred.correct).sum()) == metrics['test_errors']
    report = classification_report(test_y, pred.predicted, labels=range(10), target_names=train.NAMES,
                                   output_dict=True, zero_division=0)
    assert report == metrics['per_class']
    probability = np.load(root / 'test_probabilities.npy', allow_pickle=False)
    assert probability.shape == (len(test_y), 10) and probability.dtype == np.float32
    assert np.isfinite(probability).all() and (probability >= 0).all() and (probability <= 1).all()
    assert np.allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    assert np.array_equal(probability.argmax(axis=1), pred.predicted)
    assert np.allclose(probability.max(axis=1), pred.confidence, atol=1e-7)
    inspected = []
    with h5py.File(root / 'model.weights.h5', 'r') as handle:
        def inspect(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                assert np.issubdtype(data.dtype, np.number) and np.isfinite(data).all()
                inspected.append({'path': name, 'shape': list(data.shape), 'dtype': str(data.dtype)})
        handle.visititems(inspect)
    assert inspected
    model = train.tf.keras.models.model_from_json((root / 'model_architecture.json').read_text())
    if getattr(model, 'optimizer', None) is not None:
        model.optimizer.build(model.trainable_variables)
    model.load_weights(root / 'model.weights.h5')
    reloaded = train.probabilities(model, test_x)
    max_difference = float(np.max(np.abs(reloaded - probability)))
    assert np.allclose(reloaded, probability, rtol=1e-6, atol=1e-7)
    assert np.array_equal(reloaded.argmax(axis=1), pred.predicted)
    assert execution['train_script_sha256'] == train.digest((root / 'train.py').read_bytes())
    assert execution['config_sha256'] == train.digest((root / 'study_config.json').read_bytes())
    train.write_json(root / 'verification.json', {
        'status': 'passed', 'source_files_fully_validated': len(sources),
        'source_train_rows': len(x), 'source_test_rows': len(test_y),
        'holdout_metrics_recomputed_from_predictions': actual,
        'dummy_metrics_recomputed': True, 'cv_mean_and_sd_recomputed': True,
        'saved_fold_assignments_match_seeded_split': True,
        'source_labels_match_all_predictions': True, 'all_errors_match_csv': True,
        'reloaded_weight_predictions_match_all_test_rows': True,
        'max_probability_difference_after_weight_reload': max_difference,
        'hdf5_numeric_datasets_inspected': inspected,
        'execution_code_and_config_hashes_match': True,
        'scope': 'Independent artifact checks and full holdout inference; training was not rerun.'})
    print('VERIFIED', json.dumps(actual), 'reload max difference:', max_difference)


if __name__ == '__main__':
    main()
