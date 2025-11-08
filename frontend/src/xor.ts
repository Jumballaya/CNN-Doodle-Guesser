import { NeuralNetwork } from "./nn/NeuralNetwork";
import "./style.css";

const trainingData = [
  [[0, 0], [0]],
  [[1, 1], [0]],
  [[1, 0], [1]],
  [[0, 1], [1]],
];

async function xor() {
  const nn = new NeuralNetwork(
    [
      { type: "input", shape: [1, 1, 1, 2] },
      { type: "dense", size: 4, activation: "relu" },
      { type: "dense", size: 4, activation: "relu" },
      { type: "dense", size: 1, activation: "relu" },
    ],
    { learningRate: 0.01, loss: "categoricalCrossEntropy" }
  );
  for (let i = 0; i < 500000; i++) {
    for (const [is, ts] of trainingData) {
      nn.train(new Float32Array(is), new Float32Array(ts));
    }
  }

  console.log([
    (nn.guess(new Float32Array([0, 0])) as Float32Array)[0],
    (nn.guess(new Float32Array([1, 1])) as Float32Array)[0],
    (nn.guess(new Float32Array([1, 0])) as Float32Array)[0],
    (nn.guess(new Float32Array([0, 1])) as Float32Array)[0],
  ]);
}
