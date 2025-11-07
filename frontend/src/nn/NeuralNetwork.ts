import { activations, type ActivationEntry } from "./activation";
import { Matrix } from "./Matrix";

//
//    CNN upgrade:
//
//        Kernel shape: [kH, kW, C_in, C_out]
//
//              K[ky][kx] = filter patch
//              K[ky][kx][c_in] = per-channel weight
//              K[ky][kx][c_in][f] = contrib to output channel f
//
//              bias: [C_out]
//
//
//
//
//      CNN Layer shape:
//
//          Input: [N, H_in, W_in, C_in]
//          Kernel: [kH, kW, C_in, C_out]
//          Stride: [sH, sW]
//          Padding: [padH, padW]
//
//          H_out = floor((H_in + padTop + padBottom - kH) / sH) + 1
//          W_out = floor((W_in + padLeft + padRight - kW) / sW) + 1
//          Output: [N, H_out, W_out, C_out]
//
//
//          2 padding modes needed: "valid" and "same"
//
//            valid -- no padding, convolution applies where the filter fully fits
//                    H_out = floor((H_in - kH) / sH) + 1
//
//            same -- output h/w == input h/w (when stride == 1)
//                    padNeededH = max(0, (H_in - 1)*sH + kH - H_in)
//                    padTop = floor(padNeededH / 2)
//                    padBottom = padNeededH - padTop
//
//
//
//
//
//        CNN Forward pass:
//
//            For each batch n:
//              For each output row y:
//                For each output col x:
//                  For each output channel f:
//                    accumulate over:
//                    kernel height ky
//                    kernel width kx
//                    input channels c
//
//
//
//      Current MLP process:
//
//            Matrix → Dense → Activation → Dense → Activation → Output
//
//      CNN process:
//
//            Tensor → Conv2D → Activation → Pool → Conv2D → Activation → Flatten → Dense → Dense
//
//
//
//
//      Example Pipeline for doodle detector:
//
//          [
//            { type:"input", shape:[28,28,1] },
//            { type:"conv2d", filters:8, kernel:[3,3], stride:1, padding:"same", activation:"relu" },
//            { type:"pool", size:[2,2] },
//            { type:"conv2d", filters:16, kernel:[3,3], activation:"relu" },
//            { type:"flatten" },
//            { type:"dense", size:64, activation:"relu" },
//            { type:"dense", size:10, activation:"softmax" }
//          ]
//

export type LayerType = "input" | "dense";

export interface LayerConfig {
  type: LayerType;
  size: number;
  activation?: keyof typeof activations;
}

export interface Layer {
  type: LayerType;
  size: number;
  activation?: ActivationEntry;
  weights?: Matrix;
  bias?: Matrix;
}

export interface NeuralNetworkOptions {
  learningRate?: number;
}

export class NeuralNetwork {
  private layers: Layer[] = [];
  private learningRate: number;

  constructor(configs: LayerConfig[], opts?: NeuralNetworkOptions) {
    if (configs.length < 2) {
      throw new Error(
        "NeuralNetwork requires at least input and output layers."
      );
    }
    this.learningRate = opts?.learningRate ?? 0.1;

    // Build layers
    for (let i = 0; i < configs.length; i++) {
      const cfg = configs[i];
      const layer: Layer = {
        type: cfg.type,
        size: cfg.size,
        activation: cfg.activation ? activations[cfg.activation] : undefined,
      };

      // Create weights/bias only for dense layers (skip input layer)
      if (cfg.type === "dense" && i > 0) {
        const prevSize = configs[i - 1].size;
        layer.weights = new Matrix(cfg.size, prevSize);
        layer.bias = new Matrix(cfg.size, 1);
        layer.weights.randomize();
        layer.bias.randomize();
      }

      this.layers.push(layer);
    }
  }

  public guess(inputs: number[]): number[] {
    if (inputs.length !== this.layers[0].size)
      throw new Error(
        `Expected ${this.layers[0].size} inputs, got ${inputs.length}`
      );

    const activations = this.feedForward(Matrix.fromArray(inputs));
    const output = activations[activations.length - 1];
    return Array.from(output.values);
  }

  public train(is: number[], ts: number[]) {
    const inputs = Matrix.fromArray(is);
    const targets = Matrix.fromArray(ts);

    // 1) Forward pass
    const acts = this.feedForward(inputs); // [a0, a1, ..., aL]
    const outputs = acts[acts.length - 1];

    // 2) Initial error at output
    let errors = Matrix.subtract(targets, outputs); // (target - output)

    // 3) Backward pass through all dense layers
    for (let i = this.layers.length - 1; i > 0; i--) {
      const layer = this.layers[i];
      if (layer.type !== "dense" || !layer.weights || !layer.bias) continue;

      const a = acts[i]; // activation of this layer
      const aPrev = acts[i - 1]; // activation of previous layer

      // Gradient = f'(a) ⊙ errors
      const df = layer.activation?.df ?? activations.sigmoid.df;
      const grad = a.clone(df);
      grad.mul(errors);
      grad.mulScalar(this.learningRate);

      // ΔW = grad × aPrevᵀ
      const aPrev_T = aPrev.transpose();
      const deltaW = Matrix.multiply(grad, aPrev_T);

      // Propagate error BEFORE updating weights
      const W_T = layer.weights.transpose();
      const prevErrors = Matrix.multiply(W_T, errors);

      // Apply updates
      layer.weights.add(deltaW);
      layer.bias.add(grad);

      // Move to previous layer
      errors = prevErrors;
    }
  }

  public trainEpochs(data: Array<[number[], number[]]>, epochs = 1000) {
    for (let e = 0; e < epochs; e++) {
      let totalError = 0;
      for (const [input, target] of data) {
        this.train(input, target);
        const guess = this.guess(input);
        const error = target.reduce(
          (sum, t, i) => sum + (t - guess[i]) ** 2,
          0
        );
        totalError += error;
      }
      if (e % 100 === 0) {
        console.log(
          `Epoch ${e} - MSE: ${(totalError / data.length).toFixed(6)}`
        );
      }
    }
  }

  public serialize() {
    return JSON.stringify({
      learningRate: this.learningRate,
      layers: this.layers.map((l) => ({
        type: l.type,
        size: l.size,
        activation: l.activation?.name,
        weights: l.weights?.values,
        bias: l.bias?.values,
      })),
    });
  }

  public static deserialize(json: string) {
    const data = JSON.parse(json);
    const layers = data.layers.map((l: any) => ({
      ...l,
      weights: l.weights ? Matrix.fromArray(l.weights) : undefined,
      bias: l.bias ? Matrix.fromArray(l.bias) : undefined,
      activation: l.activation ? activations[l.activation] : undefined,
    }));
    return new NeuralNetwork(layers, { learningRate: data.learningRate });
  }

  public mutate(rate: number = 0.1) {
    for (const layer of this.layers) {
      if (!layer.weights || !layer.bias) continue;
      layer.weights.map((v) => v + (Math.random() * 2 - 1) * rate);
      layer.bias.map((v) => v + (Math.random() * 2 - 1) * rate);
    }
  }

  // returns [hidden, outputs]
  private feedForward(inputs: Matrix): Matrix[] {
    const activations: Matrix[] = [inputs];

    let current = inputs;
    for (let i = 1; i < this.layers.length; i++) {
      const layer = this.layers[i];
      if (layer.type !== "dense" || !layer.weights || !layer.bias) {
        continue;
      }

      // linear transform: z = W * a_prev + b
      current = Matrix.multiply(layer.weights, current);
      current.add(layer.bias);

      // non-linear transform: a = f(z)
      if (layer.activation) {
        current.map(layer.activation.f);
      }

      activations.push(current);
    }

    return activations; // last element is output layer activation
  }
}
