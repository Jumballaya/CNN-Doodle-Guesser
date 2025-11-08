//
// Loss Functions Library
//
// Each entry defines f(ŷ, y) and df(ŷ, y)
// where ŷ = predicted, y = target
//

export type LossEntry = {
  f: (yhat: Float32Array, y: Float32Array) => number;
  df: (yhat: Float32Array, y: Float32Array) => Float32Array;
  name: string;
};

//
// 1. Mean Squared Error (MSE)
//
function mse(yhat: Float32Array, y: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < y.length; i++) {
    const d = yhat[i] - y[i];
    sum += d * d;
  }
  return (0.5 * sum) / y.length;
}

function dmse(yhat: Float32Array, y: Float32Array): Float32Array {
  const d = new Float32Array(y.length);
  for (let i = 0; i < y.length; i++) d[i] = yhat[i] - y[i];
  return d;
}

//
// 2. Binary Cross-Entropy (BCE)
//
function bce(yhat: Float32Array, y: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < y.length; i++) {
    const p = Math.min(Math.max(yhat[i], 1e-7), 1 - 1e-7);
    sum += -y[i] * Math.log(p) - (1 - y[i]) * Math.log(1 - p);
  }
  return sum / y.length;
}

function dbce(yhat: Float32Array, y: Float32Array): Float32Array {
  const d = new Float32Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const p = Math.min(Math.max(yhat[i], 1e-7), 1 - 1e-7);
    d[i] = (p - y[i]) / (p * (1 - p));
  }
  return d;
}

//
// 3. Categorical Cross-Entropy (Softmax outputs)
//
function categoricalCrossEntropy(yhat: Float32Array, y: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < y.length; i++) {
    const p = Math.min(Math.max(yhat[i], 1e-7), 1 - 1e-7);
    sum += -y[i] * Math.log(p);
  }
  return sum;
}

function dcategoricalCrossEntropy(
  yhat: Float32Array,
  y: Float32Array
): Float32Array {
  const d = new Float32Array(y.length);
  for (let i = 0; i < y.length; i++) d[i] = yhat[i] - y[i];
  return d;
}

//
// Registry
//
export const losses: Record<string, LossEntry> = {
  mse: { f: mse, df: dmse, name: "mse" },
  bce: { f: bce, df: dbce, name: "bce" },
  categoricalCrossEntropy: {
    f: categoricalCrossEntropy,
    df: dcategoricalCrossEntropy,
    name: "categoricalCrossEntropy",
  },
};
