import { DrawingApp } from "./drawingApp/DrawingApp";
import { NeuralNetwork } from "./nn/NeuralNetwork";
import "./style.css";

type TrainingEntry = {
  data: number[];
  label: number;
};

type DataEntry = {
  train: Array<TrainingEntry>;
  test: Array<TrainingEntry>;
};

// returns the index of the max value
const argMax = (arr: number[]): number => {
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

const loadBytes = async (path: string): Promise<Uint8Array> => {
  const res = await fetch(path);
  return await res.bytes();
};

const normalizeData = (d: Uint8Array): number[] => {
  const inputs = new Array(784);
  for (let j = 0; j < 784; j++) {
    inputs[j] = 1.0 - d[j] / 255.0;
  }
  return inputs;
};

const loadData = async (
  dataset: string,
  label: number
): Promise<[Array<TrainingEntry>, Array<TrainingEntry>]> => {
  const imgSize = 28 * 28;
  const train_path = `data/${dataset}_train.bin`;
  const test_path = `data/${dataset}_test.bin`;

  const train_bytes = await loadBytes(train_path);
  const test_bytes = await loadBytes(test_path);

  const train_array: Array<TrainingEntry> = [];
  for (let i = 0; i < 800; i++) {
    train_array.push({
      data: normalizeData(train_bytes.slice(imgSize * i, imgSize * (i + 1))),
      label,
    });
  }

  const test_array: Array<TrainingEntry> = [];
  for (let i = 0; i < 200; i++) {
    test_array.push({
      data: normalizeData(test_bytes.slice(imgSize * i, imgSize * (i + 1))),
      label,
    });
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
  const nn = new NeuralNetwork([
    { type: "input", shape: [1, 28, 28, 1], activation: "sigmoid" },
    { type: "dense", size: 64, activation: "sigmoid" },
    { type: "dense", size: 3, activation: "sigmoid" },
  ]);

  for (let e = 0; e < epochCount; e++) {
    for (let i = 0; i < training.length; i++) {
      const { data, label } = training[i];
      const inputs = new Array(784);
      for (let j = 0; j < 784; j++) {
        inputs[j] = data[j] / 255.0;
      }

      switch (label) {
        case 0: {
          nn.train(inputs, [1, 0, 0]);
          break;
        }
        case 1: {
          nn.train(inputs, [0, 1, 0]);
          break;
        }
        case 2: {
          nn.train(inputs, [0, 0, 1]);
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

async function main() {
  const drawApp = new DrawingApp();

  const [nn, data] = await trainNN(8);

  // let testing_data: TrainingEntry[] = [];
  // testing_data = testing_data.concat(data.butterfly.test);
  // testing_data = testing_data.concat(data.cat.test);
  // testing_data = testing_data.concat(data.rainbow.test);

  // shuffleArray(testing_data);

  // let guesses: number[] = [];
  // const len = testing_data.length;
  // for (let i = 0; i < len; i++) {
  //   const guess = nn.guess(testing_data[i].data);
  //   guesses.push(argMax(guess) === 1 ? 1 : 0);
  // }
  // const sum = guesses.reduce((acc, i) => acc + i, 0);
  // const percent = (sum / len) * 100;

  // console.log(`${percent}% correct`);

  // const guessBtn = document.createElement("button");
  // guessBtn.innerText = "Guess!";
  // document.body.appendChild(guessBtn);
  // guessBtn.addEventListener("click", (e) => {
  //   e.preventDefault();
  //   const data = drawApp.getData();
  //   const guess = nn.guess(data);
  //   const output = argMax(guess);
  //   if (output === 0) {
  //     console.log("Butterfly?");
  //   } else if (output === 1) {
  //     console.log("Cat?");
  //   } else if (output === 2) {
  //     console.log("Rainbow?");
  //   }
  // });
}
main();
