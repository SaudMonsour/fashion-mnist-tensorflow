# Source and provenance

This study uses [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), created by Han Xiao, Kashif Rasul, and Roland Vollgraf at Zalando Research. Dataset copyright © 2017 Zalando SE; distributed under the [MIT license](https://github.com/zalandoresearch/fashion-mnist/blob/b2617bb6d3ffa2e429640350f613e3291e10b141/LICENSE). The complete notice is preserved in `DATA_LICENSE.txt`. Example images in the study figures come from this dataset.

Reference: Xiao, H., Rasul, K., and Vollgraf, R. (2017), [Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747).

The four original gzip files were retrieved and validated on 2026-09-20 from source commit `b2617bb6d3ffa2e429640350f613e3291e10b141`. `data_sources.json` records each pinned URL, compressed SHA-256, decompressed SHA-256, shape, dtype, bounds, and exact decompressed length. Their MD5 values match the checksum table published by the dataset authors:

| File | Published MD5 |
| :--- | :--- |
| `train-images-idx3-ubyte.gz` | `8d4fb7e6c68d591d4c3dfef9ec88bf0d` |
| `train-labels-idx1-ubyte.gz` | `25c81989df183df01b3e8a0aad5dffbe` |
| `t10k-images-idx3-ubyte.gz` | `bef4ecab320f06d8554ea6380940ec79` |
| `t10k-labels-idx1-ubyte.gz` | `bb300cfdad3c16e7a12a480ee83cd310` |

All decompressed payloads were read in full and validated as IDX data. Training contains 60,000 images and testing 10,000 images; each is a 28 × 28 uint8 array with a label in 0–9. SHA-256 comparison of every image found no exact within-partition duplicates and no exact train/test overlaps. No source rows were excluded. The model uses the official split; stratified training fold IDs are stored in `cv_assignments.csv`.

The raw dataset archives are not copied into this project. The pinned public sources, source hashes, and row IDs identify the complete input data. The project dataset fingerprint is SHA-256 of the UTF-8 JSON mapping from filename to decompressed SHA-256, with keys sorted and separators `(',', ':')`.

The project-specific `train.py` handles image tensors and TensorFlow. It is separate from the lab's tabular daily agent; the recurring agent configuration was not changed. Settings were fixed before scoring. This was one automated study, with no paid LLM commentary and no claims of human review. `environment.lock.txt` records the training environment; notebook-only packages are listed separately. Deterministic TensorFlow operations and seed 42 were used; exact numeric agreement across hardware or library versions is not guaranteed, as explained in the [TensorFlow determinism documentation](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism).
