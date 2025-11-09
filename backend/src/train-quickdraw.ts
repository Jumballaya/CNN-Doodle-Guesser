import { NeuralNetwork, shuffleArray } from "@doodle/lib";
import { DatasetBuilder } from "./data/DatasetBuilder";
import { NodeScheduler } from "./train/Scheduler";
import { Trainer } from "./train/Trainer";
import fs from "node:fs/promises";
import { getCLIOptions, USAGE } from "./cli";

async function getClassList(path: string): Promise<string[]> {
  const res = await fs.readFile(path);
  const txt = res.toString();
  return txt.split("\n");
}

async function main() {
  const opts = getCLIOptions();
  if (opts.help) {
    console.log(USAGE);
    return;
  }
  // Get CLI inputs

  // Load class list and build data set
  const classes = await getClassList(opts.classList);
  const dataset = await DatasetBuilder.create(classes, {
    manifestPath: ".cache/public/quickdraw.json",
    trainCount: opts.trainCount,
    testCount: opts.testCount,
  });

  const nn = new NeuralNetwork(
    [
      { type: "input", shape: [1, 28, 28, 1] },
      { type: "conv2d", kernel: [3, 3], filters: 8, activation: "leakyRelu" },
      { type: "pool", size: [2, 2] },
      { type: "conv2d", kernel: [3, 3], filters: 16, activation: "leakyRelu" },
      { type: "pool", size: [2, 2] },
      { type: "flatten" },
      { type: "dense", size: 64, activation: "leakyRelu" },
      { type: "dense", size: classes.length, activation: "softmax" },
    ],
    { learningRate: 0.005, loss: "categoricalCrossEntropy" }
  );

  // Training
  const scheduler = new NodeScheduler();
  const trainer = new Trainer(nn, dataset, scheduler, {
    batchSize: 32,
    epochs: 30,
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
          ).toFixed(1)}%`
        );
      },
      onCheckpoint: async (e, ckpt) => {
        console.log(`Checkpoint saved: completed: ${e}`);
      },
    },
    ".cache/checkpoints/model_epoch-{epoch}.json"
  );

  // Final model output
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
