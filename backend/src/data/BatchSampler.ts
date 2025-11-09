import type { ClassInfo } from "./manifest";
import type { IDataSource, Batch } from "./NodeFileSource";

interface BatchSamplerOptions {
  split: "train" | "test";
  imgSize: number; // 28*28
}

/**
 * Shuffles across all classes and yields mini-batches.
 */
export class BatchSampler {
  private classes: ClassInfo[];
  private data: IDataSource;
  private batchSize: number;
  private opts: BatchSamplerOptions;

  constructor(
    classes: ClassInfo[],
    data: IDataSource,
    batchSize: number,
    opts: BatchSamplerOptions
  ) {
    this.classes = classes;
    this.data = data;
    this.batchSize = batchSize;
    this.opts = opts;
  }

  numClasses(): number {
    return this.classes.length;
  }

  async *batches(): AsyncGenerator<Batch> {
    const IMG = this.opts.imgSize;
    const C = this.classes.length;
    const order: Array<{ c: ClassInfo; off: number }> = [];

    for (const c of this.classes) {
      const split = c[this.opts.split];
      for (let i = 0; i < split.count; i++) {
        const off = split.offsets[i];
        order.push({ c, off });
      }
    }

    // Shuffle each epoch
    for (let i = order.length - 1; i > 0; i--) {
      const j = (Math.random() * (i + 1)) | 0;
      [order[i], order[j]] = [order[j], order[i]];
    }

    const B = this.batchSize;
    for (let i = 0; i < order.length; i += B) {
      const slice = order.slice(i, i + B);
      const effB = slice.length;
      const x = new Float32Array(effB * IMG);
      const y = new Float32Array(effB * C);

      for (let b = 0; b < effB; b++) {
        const { c, off } = slice[b];
        const split = c[this.opts.split];
        const fv = await this.data.readSample(split.bin, off, IMG);
        x.set(fv, b * IMG);
        y[b * C + c.id] = 1;
      }

      yield { x, y };
    }
  }
}
