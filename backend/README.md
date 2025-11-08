# Data pipeline and model trainer

## Data flow

1. `DatasetBuilder.create(...)`

   - For each doodle class:
     - Download `.npy` into `.cache/quickdraw/<name>.npy` (skip if exists).
     - Parse NPY header (shape, dtype, offset).
     - Shuffle indices.
     - Read selected samples directly from `.npy` (one image at a time) and write:
       - `frontend/public/data/<name>_train.bin`
       - `frontend/public/data/<name>_test.bin`
     - Record byte offsets `[0, 784, 1568, ...]` per split.
   - Writes a manifest JSON (e.g. `frontend/public/data/quickdraw-3.json`).
   - Returns a `QuickdrawDataset` instance that wraps:
     - The manifest.
     - A `NodeFileSource` that can read from `.bin` on demand.

2. `QuickdrawDataset.createBatchSampler(split, batchSize)`

   - Produces a `BatchSampler` bound to the manifest and `NodeFileSource`.

3. `Trainer`

   - Takes a `NeuralNetwork`, a `QuickdrawDataset`, and a scheduler.
   - For each epoch:
     - Creates a fresh `BatchSampler`.
     - Iterates batches (`for await`), performing per-sample `nn.train(xi, yi)`.
     - Tracks loss/accuracy; invokes callbacks.
     - Saves checkpoints as JSON files using `nn.checkpoint()`.

Everything is built around:

- `.bin` = compact `Uint8` images (no labels, 784 bytes/sample).
- Manifest = where bins live + offsets + class ids.
- `NodeFileSource` = reads + normalizes (u8→f32/255) on the fly.
