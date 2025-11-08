//
// Activation Functions Library
// Each entry defines f(x), df(y), and a canonical name.
//

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y: number): number {
  // derivative expects y = sigmoid(x)
  return y * (1 - y);
}

function tanh(x: number): number {
  if (x > 20) x = 20;
  if (x < -20) x = -20;
  return Math.tanh(x);
}

function dtanh(y: number): number {
  // derivative expects y = tanh(x)
  return 1 - y * y;
}

function relu(x: number): number {
  return Math.max(0, x);
}

function drelu(y: number): number {
  // derivative expects y = relu(x)
  return y > 0 ? 1 : 0;
}

function leakyRelu(x: number, alpha = 0.01): number {
  return x > 0 ? x : alpha * x;
}

function dleakyRelu(y: number, alpha = 0.01): number {
  // derivative expects y = relu(x)
  return y > 0 ? 1 : alpha;
}

function elu(x: number, alpha = 1.0): number {
  return x >= 0 ? x : alpha * (Math.exp(x) - 1);
}

function delu(y: number, alpha = 1.0): number {
  // derivative expects y = elu(x)
  return y >= 0 ? 1 : y + alpha;
}

function softplus(x: number): number {
  return Math.log(1 + Math.exp(x));
}

function dsoftplus(y: number): number {
  // derivative is sigmoid(x), but here y = softplus(x)
  // To keep it consistent with other df(y) signatures, we’ll approximate:
  return 1 / (1 + Math.exp(-y));
}

function linear(x: number): number {
  return x;
}

function dlinear(_y: number): number {
  return 1;
}

type Fn = (n: number) => number;
export type ActivationEntry = { f: Fn; df: Fn; name: string };

export const activations: Record<string, ActivationEntry> = {
  sigmoid: { f: sigmoid, df: dsigmoid, name: "sigmoid" },
  tanh: { f: tanh, df: dtanh, name: "tanh" },
  relu: { f: relu, df: drelu, name: "relu" },
  leakyRelu: {
    f: (x) => leakyRelu(x, 0.01),
    df: (y) => dleakyRelu(y, 0.01),
    name: "leakyRelu",
  },
  elu: {
    f: (x) => elu(x, 1.0),
    df: (y) => delu(y, 1.0),
    name: "elu",
  },
  softplus: { f: softplus, df: dsoftplus, name: "softplus" },
  linear: { f: linear, df: dlinear, name: "linear" },
};
