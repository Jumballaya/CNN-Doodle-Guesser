import { DrawingApp } from "./drawingApp/DrawingApp";
import { NeuralNetwork } from "./nn/NeuralNetwork";
import type { ActivationValue } from "./nn/nn.types";
import "./style.css";

type TrainingEntry = {
  data: Float32Array;
  label: number;
};

type DataEntry = {
  train: Array<TrainingEntry>;
  test: Array<TrainingEntry>;
};

// returns the index of the max value
const argMax = (arr: ActivationValue): number => {
  if (!(arr instanceof Float32Array)) {
    arr = arr.flatten()[0];
  }
  let idx = 0;
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max) {
      max = arr[i];
      idx = i;
    }
  }

  return idx;
};

const IMG = 28 * 28;

const normalizeDataView = (bytes: Uint8Array): Float32Array => {
  const out = new Float32Array(IMG);
  for (let i = 0; i < IMG; i++) out[i] = 1 - bytes[i] / 255;
  return out;
};

const loadBytes = async (path: string): Promise<Uint8Array> => {
  const res = await fetch(path);
  return new Uint8Array(await res.arrayBuffer());
};

const loadData = async (
  dataset: string,
  label: number
): Promise<[Array<TrainingEntry>, Array<TrainingEntry>]> => {
  const train_path = `data/${dataset}_train.bin`;
  const test_path = `data/${dataset}_test.bin`;

  const train_bytes = await loadBytes(train_path);
  const test_bytes = await loadBytes(test_path);

  const numTrain = Math.floor(train_bytes.length / IMG);
  const numTest = Math.floor(test_bytes.length / IMG);

  const train_array: Array<TrainingEntry> = [];
  for (let i = 0; i < numTrain; i++) {
    const start = i * IMG;
    const slice = train_bytes.slice(start, start + IMG);
    train_array.push({ data: normalizeDataView(slice), label });
  }

  const test_array: Array<TrainingEntry> = [];
  for (let i = 0; i < numTest; i++) {
    const start = i * IMG;
    const slice = test_bytes.slice(start, start + IMG);
    test_array.push({ data: normalizeDataView(slice), label });
  }

  return [test_array, train_array];
};

const loadTrainingData = async (
  datasets: string[]
): Promise<Record<string, DataEntry>> => {
  const out: Record<string, DataEntry> = {};

  for (let i = 0; i < datasets.length; i++) {
    const ds = datasets[i];
    const [test, train] = await loadData(ds, i);
    out[ds] = { test, train };
  }

  return out;
};

const shuffleArray = <T = unknown>(array: Array<T>): Array<T> => {
  let currentIndex = array.length;
  let randomIndex;

  while (currentIndex !== 0) {
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex],
      array[currentIndex],
    ];
  }

  return array;
};

async function collectData(): Promise<
  [TrainingEntry[], Record<string, DataEntry>]
> {
  const data = await loadTrainingData(["butterfly", "cat", "rainbow"]);

  let training: TrainingEntry[] = [];
  training = training.concat(data.butterfly.train);
  training = training.concat(data.cat.train);
  training = training.concat(data.rainbow.train);
  shuffleArray(training);

  return [training, data];
}

function* generateNN(
  training: TrainingEntry[],
  epochCount = 1
): Generator<void, NeuralNetwork, void> {
  const nn = new NeuralNetwork(
    [
      { type: "input", shape: [1, 28, 28, 1] },
      { type: "dense", size: 128, activation: "relu" },
      { type: "dense", size: 32, activation: "relu" },
      { type: "dense", size: 3, activation: "softmax" },
    ],
    { learningRate: 0.005, loss: "categoricalCrossEntropy" }
  );

  for (let e = 0; e < epochCount; e++) {
    shuffleArray(training);
    for (let i = 0; i < training.length; i++) {
      const { data, label } = training[i];
      switch (label) {
        case 0: {
          nn.train(new Float32Array(data), new Float32Array([1, 0, 0]));
          break;
        }
        case 1: {
          nn.train(new Float32Array(data), new Float32Array([0, 1, 0]));
          break;
        }
        case 2: {
          nn.train(new Float32Array(data), new Float32Array([0, 0, 1]));
          break;
        }
      }
    }
    yield;
  }

  return nn;
}

async function trainNN(
  epochCount = 1
): Promise<[NeuralNetwork, Record<string, DataEntry>]> {
  return new Promise(async (res) => {
    const [training, allData] = await collectData();
    const gen = generateNN(training, epochCount);
    let current = gen.next();

    let eCount = 0;
    const loop = () => {
      if (current.done) {
        res([current.value, allData]);
        return;
      } else {
        console.log(`Epoch Trained: ${eCount++}`);
        current = gen.next();
      }
      requestAnimationFrame(loop);
    };
    loop();
  });
}

export async function doodle() {
  const drawApp = new DrawingApp();

  const [nn, data] = await trainNN(8);

  let testing_data: TrainingEntry[] = [];
  testing_data = testing_data.concat(data.butterfly.test);
  testing_data = testing_data.concat(data.cat.test);
  testing_data = testing_data.concat(data.rainbow.test);

  shuffleArray(testing_data);

  let correct = 0;
  for (let i = 0; i < testing_data.length; i++) {
    const { data, label } = testing_data[i];
    const yhat = nn.guess(new Float32Array(data));
    const pred = argMax(yhat);
    if (pred === label) correct++;
  }
  const percent = (correct / testing_data.length) * 100;
  console.log(`${percent.toFixed(2)}% correct`);

  const guessBtn = document.createElement("button");
  guessBtn.innerText = "Guess!";
  document.body.appendChild(guessBtn);
  guessBtn.addEventListener("click", (e) => {
    e.preventDefault();
    const data = drawApp.getData();
    const guess = nn.guess(new Float32Array(data));
    const output = argMax(guess);
    if (output === 0) {
      console.log("Butterfly?");
    } else if (output === 1) {
      console.log("Cat?");
    } else if (output === 2) {
      console.log("Rainbow?");
    }
  });
}
