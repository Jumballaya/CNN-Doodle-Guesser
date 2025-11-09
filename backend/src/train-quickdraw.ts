// backend/src/train-quickdraw3.ts
import { NeuralNetwork } from "@doodle/lib";
import { DatasetBuilder } from "./data/DatasetBuilder";
import { NodeScheduler } from "./train/Scheduler";
import { Trainer } from "./train/Trainer";
import fs from "node:fs/promises";

async function main() {
  // 1. Build (or load) dataset
  const dataset = await DatasetBuilder.create(
    [
      "cat",
      "butterfly",
      "rainbow",
      "banana",
      "flower",
      "ladder",
      "mushroom",
      "snowman",
      "sword",
      "nose",
    ].map((name) => ({ name, trainCount: 4000, testCount: 800 })),
    {}
  );

  // 2. Define model
  const nn = new NeuralNetwork(
    [
      { type: "input", shape: [1, 28, 28, 1] },
      { type: "conv2d", kernel: [3, 3], filters: 8, activation: "leakyRelu" },
      { type: "pool", size: [2, 2] },
      { type: "conv2d", kernel: [3, 3], filters: 16, activation: "leakyRelu" },
      { type: "pool", size: [2, 2] },
      { type: "flatten" },
      { type: "dense", size: 64, activation: "leakyRelu" },
      { type: "dense", size: 10, activation: "softmax" },
    ],
    { learningRate: 0.001, loss: "categoricalCrossEntropy" }
  );

  // 3. Create trainer
  const scheduler = new NodeScheduler();
  const trainer = new Trainer(nn, dataset, scheduler, {
    batchSize: 32,
    epochs: 30,
    batchesPerYield: 50,
    checkpointEvery: 1,
  });

  // 4. Train
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

  // 5. Optionally save final model
  const finalPath = "frontend/public/doodle-guesser.json";
  await fs.writeFile(
    finalPath,
    JSON.stringify(nn.checkpoint(), null, 2),
    "utf-8"
  );
  console.log(`Final model saved to ${finalPath}`);

  await dataset.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
