import fs from "node:fs/promises";
import path from "node:path";
import { type NeuralNetwork, Tensor4D } from "@doodle/lib";
import type { NNCheckpoint } from "@doodle/lib";
import type { QuickdrawDataset } from "../data/DatasetBuilder";
import type { IScheduler } from "./Scheduler";

export interface TrainerCallbacks {
  onEpochStart?(epoch: number): void | Promise<void>;
  onEpochEnd?(
    epoch: number,
    metrics: { loss: number; acc?: number }
  ): void | Promise<void>;
  onBatchEnd?(
    batchIndex: number,
    metrics: { loss: number }
  ): void | Promise<void>;
  onCheckpoint?(epoch: number, ckpt: NNCheckpoint): void | Promise<void>;
}

export interface TrainerOptions {
  batchSize: number;
  epochs: number;
  batchesPerYield?: number;
  checkpointEvery?: number;
  // future: validationSplit?
}

export class Trainer {
  constructor(
    private nn: NeuralNetwork,
    private dataset: QuickdrawDataset,
    private scheduler: IScheduler,
    private opts: TrainerOptions
  ) {}

  private async saveCheckpointToFile(
    epoch: number,
    filePath: string
  ): Promise<void> {
    const ckpt = this.nn.checkpoint();
    const abs = path.resolve(process.cwd(), filePath);
    await fs.mkdir(path.dirname(abs), { recursive: true });
    await fs.writeFile(abs, JSON.stringify(ckpt, null, 2), "utf-8");
  }

  /**
   * Public helper if you just want to save the current model state.
   */
  async saveCheckpoint(filePath: string): Promise<void> {
    await this.saveCheckpointToFile(-1, filePath);
  }

  async train(
    callbacks: TrainerCallbacks = {},
    checkpointPath?: string
  ): Promise<void> {
    const {
      epochs,
      batchSize,
      batchesPerYield = 10,
      checkpointEvery = 1,
    } = this.opts;

    const C = this.dataset.numClasses();

    for (let e = 0; e < epochs; e++) {
      await callbacks.onEpochStart?.(e);

      const sampler = this.dataset.createBatchSampler("train", batchSize);
      let batchIndex = 0;
      let lossSum = 0;
      let sampleCount = 0;
      let correct = 0;

      for await (const { x, y } of sampler.batches()) {
        const IMG = x.length / (y.length / C); // x: [B*IMG], y: [B*C]
        const B = Math.floor(x.length / IMG);

        for (let i = 0; i < B; i++) {
          const xi = x.subarray(i * IMG, (i + 1) * IMG);
          const yi = y.subarray(i * C, (i + 1) * C);

          // Train on single sample
          const input = new Tensor4D([1, 28, 28, 1], xi);
          this.nn.train(input, yi);

          // Optional: metrics
          const yhat = this.nn.guess(input) as Float32Array;
          const l = (this.nn as any)["lossFn"].f(yhat, yi); // use internal loss entry
          lossSum += l;
          sampleCount++;

          // Accuracy: argmax predicted vs argmax target
          let p = 0;
          let pm = -Infinity;
          for (let k = 0; k < C; k++) {
            const v = yhat[k];
            if (v > pm) {
              pm = v;
              p = k;
            }
          }
          let t = 0;
          for (let k = 0; k < C; k++) {
            if (yi[k] === 1) {
              t = k;
              break;
            }
          }
          if (p === t) correct++;
        }

        await callbacks.onBatchEnd?.(batchIndex, {
          loss: lossSum / Math.max(1, sampleCount),
        });
        batchIndex++;

        if (batchIndex % batchesPerYield === 0) {
          await this.scheduler.tick();
        }
      }

      const metrics = {
        loss: lossSum / Math.max(1, sampleCount),
        acc: correct / Math.max(1, sampleCount),
      };
      await callbacks.onEpochEnd?.(e, metrics);

      if ((e + 1) % checkpointEvery === 0) {
        const ckpt = this.nn.checkpoint();
        await callbacks.onCheckpoint?.(e, ckpt);
        if (checkpointPath) {
          const file = checkpointPath.replace("{epoch}", `${e}`);
          await this.saveCheckpointToFile(e, file);
        }
      }
    }
  }
}
