import { NeuralNetwork, shuffleArray } from "@doodle/lib";
import { DatasetBuilder } from "./data/DatasetBuilder";
import { NodeScheduler } from "./train/Scheduler";
import { Trainer } from "./train/Trainer";
import fs from "node:fs/promises";
import { getCLIOptions, USAGE } from "./cli";
import path from "node:path";

async function getClassList(path: string): Promise<string[]> {
  const res = await fs.readFile(path);
  const txt = res.toString();
  return txt.split("\n");
}

function timeString(ms: number): string {
  if (ms < 1000) {
    return `${ms}ms`;
  }
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  const adjSec = seconds - minutes * 60;
  if (minutes < 60) {
    return `${minutes}m ${adjSec}s`;
  }
  const hours = Math.floor(minutes / 60);
  const adjMins = minutes - hours * 60;
  if (hours < 60) {
    return `${hours}h ${adjMins}m ${adjSec}s`;
  }
  const days = Math.floor(hours / 24);
  const adjHours = hours - days * 24;
  return `${days}d ${adjHours}h ${adjMins}m ${adjSec}s`;
}

async function main() {
  const opts = getCLIOptions();
  if (opts.help) {
    console.log(USAGE);
    return;
  }

  const classes = await getClassList(opts.classList);
  const dataset = await DatasetBuilder.create(classes, {
    manifestPath: path.join(".cache", "public", "doodle-manifest.json"),
    trainCount: opts.trainCount,
    testCount: opts.testCount,
  });

  const nn = new NeuralNetwork(
    [
      { type: "input", shape: [1, 28, 28, 1] },
      { type: "conv2d", kernel: [3, 3], filters: 8, activation: "relu" },
      { type: "pool", size: [2, 2] },
      { type: "conv2d", kernel: [3, 3], filters: 16, activation: "relu" },
      { type: "pool", size: [2, 2] },
      { type: "flatten" },
      { type: "dense", size: 64, activation: "relu" },
      { type: "dense", size: classes.length, activation: "softmax" },
    ],
    { learningRate: opts.learnRate, loss: "categoricalCrossEntropy" }
  );

  const scheduler = new NodeScheduler();
  const trainer = new Trainer(nn, dataset, scheduler, {
    batchSize: 32,
    epochs: opts.epochs,
    batchesPerYield: 50,
    checkpointEvery: 1,
  });

  await trainer.train(
    {
      onEpochStart: (e) => {
        console.log(`Epoch ${e + 1} starting`);
      },
      onEpochEnd: (e, m) => {
        console.log(
          `Epoch ${e + 1} done: loss=${m.loss.toFixed(4)} acc=${(
            (m.acc ?? 0) * 100
          ).toFixed(1)}%\nTime taken -- ${timeString(m.runtime)}`
        );
      },
      onCheckpoint: async (e, ckpt) => {
        console.log(`Checkpoint saved: completed: ${e}`);
      },
    },
    path.join(".cache", "checkpoints", "model_epoch-{epoch}.json")
  );

  await fs.writeFile(
    opts.modelOutput,
    JSON.stringify(nn.checkpoint(), null, 2),
    "utf-8"
  );
  console.log(`Final model saved to ${opts.modelOutput}`);

  await dataset.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
