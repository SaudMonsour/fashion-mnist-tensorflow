"""One auditable TensorFlow Fashion-MNIST study; no test-driven model selection."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
TEAL, NAVY, GOLD = '#087f8c', '#14253f', '#df9b36'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_data(config, cache):
    """Validate every decompressed byte, shape, label and published source checksum."""
    cache.mkdir(parents=True, exist_ok=True)
    def load(item):
        name, spec = item
        url = f"https://raw.githubusercontent.com/{config['source_repository']}/{config['source_commit']}/data/fashion/{name}"
        path = cache / name
        if not path.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'DailyDataScienceLab/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read(35_000_001)
            if len(raw) > 35_000_000:
                raise ValueError('Source exceeds download limit')
            if hashlib.md5(raw).hexdigest() != spec['md5']:
                raise ValueError('Published source checksum mismatch: ' + name)
            path.write_bytes(raw)
        raw = path.read_bytes()
        if hashlib.md5(raw).hexdigest() != spec['md5']:
            raise ValueError('Cached checksum mismatch: ' + name)
        with gzip.open(path, 'rb') as stream:
            expanded = stream.read(50_000_001)
        magic, count = struct.unpack('>II', expanded[:8])
        assert count == spec['count']
        if spec['kind'] == 'images':
            assert magic == 2051 and struct.unpack('>II', expanded[8:16]) == (28, 28)
            assert len(expanded) == 16 + count * 784
            array = np.frombuffer(expanded, dtype=np.uint8, offset=16).reshape(count, 28, 28).copy()
        else:
            assert magic == 2049 and len(expanded) == 8 + count
            array = np.frombuffer(expanded, dtype=np.uint8, offset=8).copy()
            assert set(np.unique(array)) == set(range(10))
        return name, array, {'url': url, 'published_md5': spec['md5'], 'sha256': digest(raw),
                             'decompressed_sha256': digest(expanded), 'decompressed_bytes': len(expanded),
                             'shape': list(array.shape), 'dtype': str(array.dtype),
                             'minimum': int(array.min()), 'maximum': int(array.max()),
                             'all_decompressed_bytes_validated': True}
    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(load, config['files'].items()))
    arrays = {name: arr for name, arr, _ in loaded}
    sources = {name: audit for name, _, audit in loaded}
    return (arrays['train-images-idx3-ubyte.gz'], arrays['train-labels-idx1-ubyte.gz'],
            arrays['t10k-images-idx3-ubyte.gz'], arrays['t10k-labels-idx1-ubyte.gz'], sources)


def audit_and_clean(x, y, test_x, test_y):
    train_hash = [digest(row.tobytes()) for row in x]
    test_hash = [digest(row.tobytes()) for row in test_x]
    test_set = set(test_hash)
    labels = defaultdict(set)
    for h, label in zip(train_hash, y):
        labels[h].add(int(label))
    conflicts = {h for h, values in labels.items() if len(values) > 1}
    seen, keep, excluded = set(), [], []
    reasons = Counter()
    for row_id, h in enumerate(train_hash):
        reason = ('train_test_pixel_overlap' if h in test_set else
                  'conflicting_training_labels' if h in conflicts else
                  'duplicate_training_image' if h in seen else None)
        if reason:
            excluded.append({'train_row_id': row_id, 'reason': reason})
            reasons[reason] += 1
        else:
            keep.append(row_id)
            seen.add(h)
    assert not seen.intersection(test_set)
    audit = {'original_training_rows': len(x), 'official_test_rows': len(test_x),
             'usable_training_rows': len(keep), 'features': 784, 'image_shape': [28, 28, 1],
             'missing_pixels': 0, 'pixel_dtype': 'uint8', 'pixel_range': [int(x.min()), int(x.max())],
             'training_exclusions': dict(reasons), 'excluded_training_rows': len(excluded),
             'test_duplicate_images_retained': len(test_hash) - len(test_set),
             'train_test_exact_overlap_after_cleaning': 0,
             'training_class_counts': np.bincount(y[keep], minlength=10).tolist(),
             'test_class_counts': np.bincount(test_y, minlength=10).tolist(),
             'note': 'Official test set is preserved. Test pixels are used only for exact-overlap auditing before fitting. Test outcomes never select a model or an epoch. Near duplicates and product identity groups cannot be ruled out.'}
    return np.asarray(keep), excluded, audit


def build_model(name, config):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config['seed'])
    layers = [tf.keras.layers.Input((28, 28, 1)), tf.keras.layers.Rescaling(1.0 / 255)]
    if name == 'CNN':
        layers += [tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same'),
                   tf.keras.layers.MaxPooling2D(),
                   tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
                   tf.keras.layers.MaxPooling2D(), tf.keras.layers.Flatten(),
                   tf.keras.layers.Dense(64, activation='relu')]
    elif name == 'Dense MLP':
        layers += [tf.keras.layers.Flatten(), tf.keras.layers.Dense(128, activation='relu'),
                   tf.keras.layers.Dense(64, activation='relu')]
    else:
        raise ValueError(name)
    layers += [tf.keras.layers.Dropout(config['dropout']), tf.keras.layers.Dense(10, activation='softmax')]
    model = tf.keras.Sequential(layers, name=name.replace(' ', '_'))
    model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def dataset(x, y, config, training=False):
    ds = tf.data.Dataset.from_tensor_slices((x[..., None], y))
    if training:
        ds = ds.shuffle(len(x), seed=config['seed'], reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return ds.batch(config['batch_size']).with_options(options).prefetch(1)


def probabilities(model, x, batch=512):
    return np.concatenate([model(x[i:i + batch, ..., None], training=False).numpy()
                           for i in range(0, len(x), batch)])


def scores(y, prediction):
    return {'macro_f1': float(f1_score(y, prediction, average='macro', zero_division=0)),
            'accuracy': float(accuracy_score(y, prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction))}


def make_figures(out, x, y, test_x, test_y, predictions, confidence, audit, metrics, logs):
    figures = out / 'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'text.color': NAVY})
    def save(fig, name):
        fig.savefig(figures / (name + '.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5), layout='constrained')
    for label, ax in enumerate(axes.flat):
        ax.imshow(x[np.flatnonzero(y == label)[0]], cmap='gray', vmin=0, vmax=255)
        ax.set_title(NAMES[label]); ax.axis('off')
    fig.suptitle('Fashion-MNIST · first retained training image per class')
    save(fig, 'dataset-examples')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    axes[0].barh(NAMES, audit['training_class_counts'], color=TEAL)
    axes[0].set(title='Retained training labels', xlabel='Images')
    axes[1].axis('off')
    text = f"{len(x):,} retained training images\n10,000 official test images\n{audit['excluded_training_rows']} training images excluded\n0 exact train/test overlaps after cleaning\n0 missing pixels; uint8 range 0–255"
    axes[1].text(0.02, 0.78, text, va='top', fontsize=13, linespacing=1.9)
    fig.suptitle('Data quality and class balance')
    save(fig, 'data-quality')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    axes[0].hist(x.mean(axis=(1, 2)), bins=35, color=TEAL, edgecolor='white')
    axes[0].set(title='Mean image intensity · training', xlabel='Mean uint8 intensity', ylabel='Images')
    mean = axes[1].imshow(x.mean(axis=0), cmap='magma')
    axes[1].set(title='Mean training image', xlabel='Pixel column', ylabel='Pixel row')
    fig.colorbar(mean, ax=axes[1], label='Mean uint8 intensity')
    save(fig, 'distributions')
    board = metrics['leaderboard'][::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    ax.barh([v['model'] for v in board], [v['cv_mean'] for v in board],
            xerr=[v['cv_std'] for v in board], capsize=4,
            color=[GOLD if v['model'] == 'Dummy baseline' else TEAL for v in board])
    ax.set(title='Model comparison · three training CV folds', xlabel='Macro-F1 · bars show fold SD', xlim=(0, 1))
    save(fig, 'model-comparison')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    for name, color in [('CNN', TEAL), ('Dense MLP', GOLD)]:
        subset = [r for r in logs if r['model'] == name and r['phase'] == 'cv']
        for key, ax in [('accuracy', axes[0]), ('loss', axes[1])]:
            for part, style in [('', '-'), ('val_', '--')]:
                values = np.array([[r[part + key] for r in subset if r['fold'] == fold] for fold in (1, 2, 3)])
                ax.plot(np.arange(1, values.shape[1] + 1), values.mean(axis=0), style,
                        color=color, label=name + (' validation' if part else ' training'))
            ax.set(xlabel='Epoch', ylabel=key.title(), title='Mean CV ' + key)
            ax.legend(fontsize=8)
    save(fig, 'learning-curves')
    cm = confusion_matrix(test_y, predictions, labels=np.arange(10))
    fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
    im = ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8,
                    color='white' if cm[i, j] > cm.max() / 2 else NAVY)
    ax.set(xticks=range(10), yticks=range(10), xticklabels=NAMES, yticklabels=NAMES,
           title=metrics['selected_model'] + ' · official test confusion matrix', xlabel='Predicted', ylabel='Actual')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.colorbar(im, ax=ax, label='Images')
    save(fig, 'confusion-matrix')
    wrong = np.flatnonzero(predictions != test_y)[:12]
    fig, axes = plt.subplots(2, 6, figsize=(14, 5.3), layout='constrained')
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(test_x[i], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"True: {NAMES[test_y[i]]}\nPred: {NAMES[predictions[i]]}\np={confidence[i]:.2f}", fontsize=9)
        ax.axis('off')
    for ax in list(axes.flat)[len(wrong):]: ax.axis('off')
    fig.suptitle('First 12 test errors by source row · not cherry-picked')
    save(fig, 'error-examples')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=HERE.parents[1] / '.cache' / 'fashion-mnist')
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        raise ValueError('Refusing to overwrite recorded results; use a new output directory')
    config = json.loads((HERE / 'study_config.json').read_text())
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    start = time.monotonic()
    print('TensorFlow', tf.__version__, 'Downloading and validating source files', flush=True)
    raw_x, raw_y, test_x, test_y, sources = load_data(config, args.cache)
    write_json(out / 'data_sources.json', sources)
    keep, exclusions, audit = audit_and_clean(raw_x, raw_y, test_x, test_y)
    x, y = raw_x[keep], raw_y[keep]
    del raw_x, raw_y
    write_json(out / 'audit.json', audit)
    pd.DataFrame(exclusions, columns=['train_row_id', 'reason']).to_csv(out / 'excluded_training_rows.csv', index=False)
    print('Data audit:', json.dumps(audit), flush=True)
    cv = list(StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['seed']).split(keep, y))
    assignments = np.empty(len(keep), dtype=int)
    for fold, (tr, va) in enumerate(cv, 1):
        assert not set(tr).intersection(va)
        assignments[va] = fold
    pd.DataFrame({'source_train_row_id': keep, 'validation_fold': assignments}).to_csv(out / 'cv_assignments.csv', index=False)
    write_json(out / 'splits.json', {'strategy': 'Fixed official test split; exact overlap removal; stratified 3-fold CV on retained training images',
               'seed': config['seed'], 'training_rows': len(x), 'test_rows': len(test_x),
               'test_source_row_ids': '0 through 9999 inclusive', 'cv_assignments': 'cv_assignments.csv'})
    records, logs, parameters = [], [], {}
    for name in config['models']:
        fold_scores = []
        for fold, (tr, va) in enumerate(cv, 1):
            if name == 'Dummy baseline':
                prediction = np.full(len(va), np.bincount(y[tr], minlength=10).argmax())
            else:
                model = build_model(name, config)
                parameters[name] = model.count_params()
                print(f'{name} fold {fold}/3', flush=True)
                history = model.fit(dataset(x[tr], y[tr], config, True),
                                    validation_data=dataset(x[va], y[va], config), epochs=config['epochs'], verbose=2)
                for epoch in range(config['epochs']):
                    logs.append({'model': name, 'phase': 'cv', 'fold': fold, 'epoch': epoch + 1,
                                 **{k: float(v[epoch]) for k, v in history.history.items()}})
                prediction = probabilities(model, x[va]).argmax(axis=1)
            score = scores(y[va], prediction)
            fold_scores.append(score['macro_f1'])
            print(name, 'fold', fold, score, flush=True)
        records.append({'model': name, 'cv_mean': float(np.mean(fold_scores)),
                        'cv_std': float(np.std(fold_scores)), 'fold_scores': fold_scores})
        pd.DataFrame(logs).to_csv(out / 'training_history.csv', index=False)
        write_json(out / 'cv_progress.json', records)
    records.sort(key=lambda r: r['cv_mean'], reverse=True)
    winner = records[0]['model']
    print('Frozen CV winner:', winner, flush=True)
    write_json(out / 'selection.json', {'selected_model': winner, 'leaderboard': records,
               'selection_used_test': False, 'fixed_epochs': config['epochs']})
    dummy = np.full(len(test_y), np.bincount(y, minlength=10).argmax())
    if winner == 'Dummy baseline':
        prediction, confidence = dummy, np.ones(len(dummy))
    else:
        model = build_model(winner, config)
        history = model.fit(dataset(x, y, config, True), epochs=config['epochs'], verbose=2)
        for epoch in range(config['epochs']):
            logs.append({'model': winner, 'phase': 'final', 'fold': 0, 'epoch': epoch + 1,
                         **{k: float(v[epoch]) for k, v in history.history.items()}})
        prob = probabilities(model, test_x)
        prediction, confidence = prob.argmax(axis=1), prob.max(axis=1)
        model.save_weights(out / 'model.weights.h5')
        (out / 'model_architecture.json').write_text(model.to_json() + '\n')
        lines = []; model.summary(print_fn=lines.append)
        (out / 'model_summary.txt').write_text('\n'.join(lines) + '\n')
        np.save(out / 'test_probabilities.npy', prob, allow_pickle=False)
    metrics = {'task': 'classification', 'framework': 'TensorFlow', 'primary_metric': 'macro_f1',
               'higher_is_better': True, 'selected_model': winner, 'leaderboard': records,
               'test_metrics': {winner: scores(test_y, prediction), 'Dummy baseline': scores(test_y, dummy)},
               'train_rows': len(x), 'test_rows': len(test_x), 'selection_used_test': False,
               'per_class': classification_report(test_y, prediction, labels=range(10), target_names=NAMES,
                                                  output_dict=True, zero_division=0),
               'parameter_counts': parameters, 'epochs': config['epochs'], 'seed': config['seed'],
               'test_errors': int((prediction != test_y).sum()), 'elapsed_seconds': time.monotonic() - start}
    write_json(out / 'metrics.json', metrics)
    pred = pd.DataFrame({'test_row_id': np.arange(len(test_y)), 'actual': test_y,
                         'predicted': prediction, 'correct': test_y == prediction, 'confidence': confidence})
    pred.to_csv(out / 'predictions.csv', index=False)
    pred[~pred.correct].to_csv(out / 'error_analysis.csv', index=False)
    pd.DataFrame(logs).to_csv(out / 'training_history.csv', index=False)
    pd.DataFrame({'class_id': range(10), 'label': NAMES}).to_csv(out / 'class_names.csv', index=False)
    pd.DataFrame([{'column': f'pixel_{r}_{c}', 'dtype': 'uint8', 'minimum': 0, 'maximum': 255, 'role': 'image pixel'}
                  for r in range(28) for c in range(28)] +
                 [{'column': 'label', 'dtype': 'uint8', 'minimum': 0, 'maximum': 9, 'role': 'target'}]).to_csv(out / 'data_dictionary.csv', index=False)
    pd.DataFrame(x.reshape(len(x), -1), columns=[f'pixel_{r}_{c}' for r in range(28) for c in range(28)]).describe().to_csv(out / 'descriptive_statistics.csv')
    make_figures(out, x, y, test_x, test_y, prediction, confidence, audit, metrics, logs)
    write_json(out / 'execution.json', {'created_at': datetime.now(ZoneInfo('Asia/Riyadh')).isoformat(),
               'python': platform.python_version(), 'versions': {n: importlib.metadata.version(n) for n in
               ['tensorflow-cpu', 'keras', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib']},
               'train_script_sha256': digest(Path(__file__).read_bytes()),
               'config_sha256': digest((HERE / 'study_config.json').read_bytes()),
               'determinism_enabled': True, 'oneDNN_enabled': False, 'device': 'CPU',
               'thread_limits': {'intra_op': 4, 'inter_op': 1},
               'limitation': 'Exact numeric equality is only expected with the same software and hardware environment.'})
    print('COMPLETE', json.dumps(metrics['test_metrics']), flush=True)


if __name__ == '__main__':
    main()
