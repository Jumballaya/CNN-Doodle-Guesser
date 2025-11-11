import { clip } from "./utils";
import { activations, softmax } from "./activation";
import type {
  ActivationValue,
  LayerCache,
  Conv2DCache,
  Conv2DConfig,
  Conv2DLayer,
  DenseConfig,
  DenseLayer,
  InputConfig,
  InputLayer,
  Layer,
  LayerConfig,
  NeuralNetworkOptions,
  PoolCache,
  PoolConfig,
  PoolLayer,
  NNCheckpoint,
  LayerSerialized,
} from "./nn.types";
import { Tensor4D } from "./Tensor";
import { losses, type LossEntry } from "./losses";

export class NeuralNetwork {
  private layers: Layer[] = [];
  private learningRate: number;
  private debug = false;
  private lossFn: LossEntry;

  constructor(configs: LayerConfig[], opts?: NeuralNetworkOptions) {
    if (configs.length < 2) {
      throw new Error(
        "NeuralNetwork requires at least input and output layers."
      );
    }
    this.debug = opts?.debug ?? false;
    this.learningRate = opts?.learningRate ?? 0.1;
    this.lossFn = losses[opts?.loss ?? "mse"];

    // Build layers
    for (let i = 0; i < configs.length; i++) {
      const cfg = configs[i];
      const prev = this.layers[i - 1];

      let layer: Layer;
      switch (cfg.type) {
        case "input": {
          layer = this.buildInputLayer(cfg);
          break;
        }
        case "dense": {
          const prev = this.layers[i - 1];
          const inputSize = this.getOutputSizeOf(prev);
          layer = this.buildDenseLayer(cfg, inputSize);
          break;
        }
        case "conv2d": {
          if (
            !prev ||
            (prev.type !== "input" &&
              prev.type !== "conv2d" &&
              prev.type !== "pool")
          ) {
            throw new Error("Conv2D must follow an input/conv/pool");
          }
          layer = this.buildConv2DLayer(cfg, prev);
          break;
        }
        case "flatten": {
          const size = this.getOutputSizeOf(prev);
          layer = { type: "flatten", size };
          break;
        }
        case "pool": {
          layer = this.buildPoolLayer(cfg, prev);
          break;
        }
      }

      this.layers.push(layer);
    }
  }

  public guess(inputs: ActivationValue): ActivationValue {
    const ctx = this.forwardPass(inputs);
    return ctx[ctx.length - 1].out;
  }

  public train(input: ActivationValue, target: Float32Array) {
    const ctx = this.forwardPass(input);
    const output = ctx[ctx.length - 1].out;
    const outputLayer = ctx[ctx.length - 1].layer;

    // account for softmax
    let dOut: ActivationValue;
    if (
      outputLayer.type === "dense" &&
      outputLayer.activation?.name === "softmax" &&
      this.lossFn.name === "categoricalCrossEntropy"
    ) {
      const out = output as Float32Array;
      dOut = new Float32Array(out.length);
      for (let i = 0; i < out.length; i++) {
        dOut[i] = out[i] - target[i];
      }
    } else {
      dOut = this.lossFn.df(output as Float32Array, target);
    }

    // Backprop
    for (let i = ctx.length - 1; i >= 0; i--) {
      const { layer, cache } = ctx[i];

      // Diagnostics every N epochs/layers
      if (this.debug && i === ctx.length - 1) {
        const weights = (layer as any).weights;
        const bias = (layer as any).bias;
        const wNorm = weights
          ? Math.sqrt(weights.reduce((s: number, v: number) => s + v * v, 0))
          : 0;
        const bMean = bias
          ? bias.reduce((s: number, v: number) => s + v, 0) / bias.length
          : 0;

        console.log(
          `Layer: ${layer.type}, wNorm=${wNorm.toFixed(
            4
          )}, bMean=${bMean.toFixed(4)}, out[0]=${
            (output as any)[0].toFixed?.(4) ?? "?"
          }`
        );
      }

      dOut = this.backwardLayer(layer, dOut, cache);

      // --- NaN Guard ---
      if (
        (this.debug &&
          dOut instanceof Float32Array &&
          !Number.isFinite(dOut[0])) ||
        (dOut instanceof Float32Array && dOut.some(isNaN))
      ) {
        throw new Error(
          `NaN detected in backward pass in layer type: ${layer.type}`
        );
      }
    }
  }

  public checkpoint(): NNCheckpoint {
    return {
      learningRate: this.learningRate,
      layers: this.layers.map(this.serializeLayer.bind(this)),
    };
  }

  public static fromCheckpoint(ckpt: NNCheckpoint): NeuralNetwork {
    const rebuild: LayerConfig[] = ckpt.layers.map((l) => {
      switch (l.type) {
        case "input":
          return { type: "input", shape: l.shape };
        case "flatten":
          return { type: "flatten" };
        case "pool":
          return {
            type: "pool",
            size: [l.windowH, l.windowW],
            stride: [l.strideH, l.strideW],
          };
        case "dense":
          return {
            type: "dense",
            size: l.outputSize,
            activation: l.activation as any,
          };
        case "conv2d":
          return {
            type: "conv2d",
            filters: l.outShape[3],
            kernel: [l.kernelShape[0], l.kernelShape[1]],
            stride: [l.strideH, l.strideW],
            padding: "valid",
            activation: l.activation as any,
          };
      }
    });
    const nn = new NeuralNetwork(rebuild, { learningRate: ckpt.learningRate });

    // inject weights/biases
    for (let i = 0; i < nn["layers"].length; i++) {
      const a = ckpt.layers[i];
      const b = nn["layers"][i];
      if (a.type === "dense" && b.type === "dense") {
        b.weights.set(Float32Array.from(a.weights));
        b.bias.set(Float32Array.from(a.bias));
      }
      if (a.type === "conv2d" && b.type === "conv2d") {
        const kernel = Float32Array.from(a.kernel);
        // Tensor4D expects (shape, data)
        b.kernel = new Tensor4D(a.kernelShape, kernel);
        b.bias.set(Float32Array.from(a.bias));
      }
    }

    return nn;
  }

  private serializeLayer(l: Layer): LayerSerialized {
    switch (l.type) {
      case "input": {
        return {
          type: "input",
          shape: l.shape,
        };
      }
      case "flatten": {
        return {
          type: "flatten",
          size: l.size,
        };
      }
      case "pool": {
        return {
          type: "pool",
          windowH: l.windowH,
          windowW: l.windowW,
          strideH: l.strideH,
          strideW: l.strideW,
          outShape: l.outShape,
          channels: l.channels,
        };
      }
      case "dense": {
        return {
          type: "dense",
          inputSize: l.inputSize,
          outputSize: l.outputSize,
          activation: l.activation?.name,
          weights: Array.from(l.weights),
          bias: Array.from(l.bias),
        };
      }
      case "conv2d": {
        return {
          type: "conv2d",
          strideH: l.strideH,
          strideW: l.strideW,
          padTop: l.padTop,
          padBottom: l.padBottom,
          padLeft: l.padLeft,
          padRight: l.padRight,
          activation: l.activation?.name,
          kernel: Array.from(l.kernel.slice(0)),
          kernelShape: l.kernel.shape,
          bias: Array.from(l.bias),
          outShape: l.outShape,
        };
      }
    }
  }

  //
  //  Back propagation
  //

  private backwardLayer(
    layer: Layer,
    dOut: ActivationValue,
    cache: LayerCache
  ): ActivationValue {
    switch (layer.type) {
      case "input": {
        return dOut;
      }
      case "conv2d": {
        return this.conv2DBackward(dOut, layer, cache);
      }
      case "pool": {
        return this.poolBackward(dOut, cache);
      }
      case "flatten": {
        return this.flattenBackward(dOut, cache);
      }
      case "dense": {
        return this.denseBackward(dOut, layer, cache);
      }
    }
  }

  private denseBackward(
    dOut: ActivationValue,
    layer: DenseLayer,
    cache: LayerCache
  ): ActivationValue {
    if (cache.type !== "dense") {
      throw new Error(`densebackward cache type mismatch, got: ${cache.type}`);
    }
    if (!(dOut instanceof Float32Array)) {
      throw new Error("denseBackward expects Float32Array gradient");
    }

    const x = cache.input; // input vector
    const z = cache.z; // pre-activation
    const f = layer.activation;
    const inSize = layer.inputSize;
    const outSize = layer.outputSize;
    const W = layer.weights;
    const B = layer.bias;

    const isOutputSoftmax =
      layer.activation?.name === "softmax" &&
      this.lossFn?.name === "categoricalCrossEntropy";

    // Compute dZ = dOut ⊙ f'(z)
    const dZ = new Float32Array(outSize);
    if (isOutputSoftmax) {
      // Gradient already simplified to (ŷ - y)
      // dOut here is the loss derivative directly from dcategoricalCrossEntropy
      for (let i = 0; i < outSize; i++) {
        dZ[i] = clip(dOut[i]);
      }
    } else {
      for (let i = 0; i < outSize; i++) {
        const dz = f ? f.df(z[i]) : 1;
        dZ[i] = clip(dOut[i] * dz);
      }
    }

    if (this.debug) {
      if (!Number.isFinite(dZ[0])) {
        throw new Error("NaN in dZ");
      }

      for (let i = 0; i < outSize; i++) {
        if (!Number.isFinite(dZ[i])) {
          throw new Error("NaN in dZ source");
        }
      }
      for (let j = 0; j < inSize; j++) {
        if (!Number.isFinite(x[j])) {
          throw new Error("NaN in x source");
        }
      }
    }

    // Compute dW and dB
    const dW = new Float32Array(inSize * outSize);
    const dB = new Float32Array(outSize);

    for (let i = 0; i < outSize; i++) {
      dB[i] = clip(dZ[i]); // bias gradient
      for (let j = 0; j < inSize; j++) {
        dW[i * inSize + j] = dZ[i] * x[j];
      }
    }

    // Compute dX = Wᵀ * dZ
    const dX = new Float32Array(inSize);

    for (let j = 0; j < inSize; j++) {
      let sum = 0;
      for (let i = 0; i < outSize; i++) {
        sum += W[i * inSize + j] * dZ[i];
      }
      dX[j] = clip(sum);
    }

    if (this.debug) {
      if (!Number.isFinite(dX[0]) || dX.some((v) => !Number.isFinite(v))) {
        console.error("NaN/Inf in dX from layer:", layer.type, {
          isOutputSoftmax:
            layer.activation?.name === "softmax" &&
            this.lossFn?.name === "categoricalCrossEntropy",
          dZ: Array.from(dZ.slice(0, 10)),
          W0: Array.from(W.slice(0, Math.min(10, W.length))),
        });
        throw new Error("NaN in dX");
      }
    }

    // Apply gradient descent update
    const lr = this.learningRate;
    for (let i = 0; i < outSize; i++) {
      B[i] -= lr * dB[i];
      for (let j = 0; j < inSize; j++) {
        const idx = i * inSize + j;
        W[idx] -= lr * dW[idx];
      }
    }

    return dX;
  }

  private flattenBackward(
    dOut: ActivationValue,
    cache: LayerCache
  ): ActivationValue {
    if (!(dOut instanceof Float32Array)) {
      throw new Error("flattenBackward expects Float32Array gradient");
    }
    if (cache.type !== "flatten") {
      throw new Error(
        `flattenbackward cache type mismatch, got: ${cache.type}`
      );
    }

    const { N, H, W, C } = cache;
    const out = new Tensor4D([N, H, W, C]);

    let idx = 0;
    for (let n = 0; n < N; n++) {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          for (let c = 0; c < C; c++) {
            out.set(n, y, x, c, dOut[idx++]);
          }
        }
      }
    }

    return out;
  }

  private conv2DBackward(
    dOut: ActivationValue,
    layer: Conv2DLayer,
    cache: LayerCache
  ): ActivationValue {
    if (!(dOut instanceof Tensor4D)) {
      throw new Error("conv2DBackward expects Tensor4D");
    }
    if (cache.type !== "conv2d") {
      throw new Error("conv2DBackward cache mismatch");
    }

    const {
      input,
      padded,
      kernel,
      outH,
      outW,
      strideH,
      strideW,
      padTop,
      padLeft,
      postAct,
    } = cache as Conv2DCache;

    const act = layer.activation;
    const N = input.getN(),
      H = padded.getH(),
      W = padded.getW();
    const kH = kernel.shape[0],
      kW = kernel.shape[1],
      C_in = kernel.shape[2],
      C_out = kernel.shape[3];

    // dZ = dOut ⊙ f'(postAct)
    const dZ = new Tensor4D(dOut.shape as [number, number, number, number]);
    for (let n = 0; n < N; n++)
      for (let oy = 0; oy < outH; oy++)
        for (let ox = 0; ox < outW; ox++)
          for (let f = 0; f < C_out; f++) {
            const g = dOut.get(n, oy, ox, f);
            const a = postAct.get(n, oy, ox, f); // post-activation output from forward
            dZ.set(n, oy, ox, f, act ? g * act.df(a) : g);
          }

    // Accumulate grads
    const dKernel = new Tensor4D([kH, kW, C_in, C_out]);
    const dBias = new Float32Array(C_out);
    const dPad = new Tensor4D([N, H, W, C_in]);

    for (let n = 0; n < N; n++) {
      for (let oy = 0; oy < outH; oy++) {
        const iyBase = oy * strideH;
        for (let ox = 0; ox < outW; ox++) {
          const ixBase = ox * strideW;
          for (let f = 0; f < C_out; f++) {
            const grad = dZ.get(n, oy, ox, f);
            dBias[f] += grad;

            for (let ky = 0; ky < kH; ky++) {
              const iy = iyBase + ky;
              for (let kx = 0; kx < kW; kx++) {
                const ix = ixBase + kx;
                for (let c = 0; c < C_in; c++) {
                  // dKernel += input_patch * grad
                  const dkOld = dKernel.get(ky, kx, c, f);
                  dKernel.set(
                    ky,
                    kx,
                    c,
                    f,
                    dkOld + padded.get(n, iy, ix, c) * grad
                  );
                  // dPad += kernel * grad
                  const dpOld = dPad.get(n, iy, ix, c);
                  dPad.set(
                    n,
                    iy,
                    ix,
                    c,
                    dpOld + kernel.get(ky, kx, c, f) * grad
                  );
                }
              }
            }
          }
        }
      }
    }

    // Crop dPad --> dInput
    const dInput = new Tensor4D(input.shape);
    for (let n = 0; n < N; n++) {
      for (let y = 0; y < input.getH(); y++) {
        for (let x = 0; x < input.getW(); x++) {
          for (let c = 0; c < C_in; c++) {
            dInput.set(n, y, x, c, dPad.get(n, y + padTop, x + padLeft, c));
          }
        }
      }
    }

    // SGD updates (use get/set, not flatten)
    const lr = this.learningRate;
    for (let f = 0; f < C_out; f++) {
      layer.bias[f] -= lr * dBias[f];
    }
    for (let ky = 0; ky < kH; ky++) {
      for (let kx = 0; kx < kW; kx++) {
        for (let c = 0; c < C_in; c++) {
          for (let f = 0; f < C_out; f++) {
            const newW =
              kernel.get(ky, kx, c, f) - lr * dKernel.get(ky, kx, c, f);
            kernel.set(ky, kx, c, f, newW);
          }
        }
      }
    }

    return dInput;
  }

  private poolBackward(
    dOut: ActivationValue,
    cache: LayerCache
  ): ActivationValue {
    if (!(dOut instanceof Tensor4D)) {
      throw new Error("poolBackward expects Tensor4D");
    }
    if (cache.type !== "pool") {
      throw new Error("poolBackward cache mismatch");
    }

    const { input, outH, outW, mask } = cache;

    const N = input.getN();
    const H = input.getH();
    const W = input.getW();
    const C = input.getC();

    const dX = new Tensor4D([N, H, W, C]);

    const sN = input["strides"][0];
    const sH = input["strides"][1];
    const sW = input["strides"][2];

    let m = 0;

    for (let n = 0; n < N; n++) {
      for (let oy = 0; oy < outH; oy++) {
        for (let ox = 0; ox < outW; ox++) {
          for (let c = 0; c < C; c++) {
            const grad = dOut.get(n, oy, ox, c);
            const idx = mask[m++];

            // Decode flat index
            const nn = (idx / sN) | 0;
            const rem1 = idx % sN;
            const yy = (rem1 / sH) | 0;
            const rem2 = rem1 % sH;
            const xx = (rem2 / sW) | 0;
            const cc = rem2 % sW;

            const prev = dX.get(nn, yy, xx, cc);
            dX.set(nn, yy, xx, cc, prev + grad);
          }
        }
      }
    }

    return dX;
  }

  //
  //  Feed forward
  //
  private forwardPass(input: ActivationValue): Array<{
    out: ActivationValue;
    cache: LayerCache;
    layer: Layer;
  }> {
    const ctx: Array<{
      out: ActivationValue;
      layer: Layer;
      cache: LayerCache;
    }> = []; // array of { layer, out, cache }
    let act = input;

    for (const layer of this.layers) {
      const { out, cache } = this.forwardLayer(layer, act);
      ctx.push({ layer, out, cache });
      act = out;
    }

    return ctx;
  }

  private forwardLayer(
    layer: Layer,
    input: ActivationValue
  ): { out: ActivationValue; cache: LayerCache } {
    switch (layer.type) {
      case "input": {
        return {
          out: input,
          cache: {
            type: "input",
            shape: layer.shape,
          },
        };
      }
      case "conv2d": {
        return this.conv2DForward(input, layer);
      }
      case "pool": {
        return this.poolForward(input, layer);
      }
      case "flatten": {
        return this.flattenForward(input);
      }
      case "dense": {
        return this.denseForward(input, layer);
      }
    }
  }

  private denseForward(
    current: ActivationValue,
    layer: DenseLayer
  ): { out: ActivationValue; cache: LayerCache } {
    // If NHWC tensor, flatten to 1D vector
    if (current instanceof Tensor4D) {
      const arr = current.flatten()[0]; // batch=1
      current = arr;
    }

    if (!(current instanceof Float32Array)) {
      throw new Error("Dense layer expects a Float32Array input");
    }

    const z = new Float32Array(layer.outputSize);
    for (let i = 0; i < layer.outputSize; i++) {
      let sum = layer.bias[i];
      for (let j = 0; j < layer.inputSize; j++) {
        sum += layer.weights[i * layer.inputSize + j] * current[j];
      }
      z[i] = sum;
    }

    let out: Float32Array;
    if (layer.activation?.name === "softmax") {
      out = softmax(z);
    } else {
      out = new Float32Array(z.length);
      for (let i = 0; i < z.length; i++) {
        out[i] = layer.activation ? layer.activation.f(z[i]) : z[i];
      }
    }

    // DEBUG
    if (this.debug) {
      for (const val of out) {
        if (Number.isNaN(val))
          throw new Error("NaN detected in denseForward output");
      }
      for (const val of z) {
        if (Number.isNaN(val))
          throw new Error("NaN detected in denseForward z");
      }
    }

    return {
      out,
      cache: {
        type: "dense",
        input: current,
        z,
      },
    };
  }

  //
  //  Slightly more memory efficient than using Tensor4D.sliceWindow()
  //
  private conv2DForward(
    current: ActivationValue,
    layer: Conv2DLayer
  ): { out: ActivationValue; cache: LayerCache } {
    if (!(current instanceof Tensor4D)) {
      throw new Error("Conv2D expects Tensor4D input");
    }

    // compute padding
    const padded = current.pad(
      layer.padTop,
      layer.padBottom,
      layer.padLeft,
      layer.padRight
    );

    const [N, outH, outW, C_out] = layer.outShape;
    const out = new Tensor4D([N, outH, outW, C_out]);

    const k = layer.kernel;
    const kH = k.shape[0];
    const kW = k.shape[1];
    const C_in = k.shape[2];

    const sH = layer.strideH;
    const sW = layer.strideW;

    for (let n = 0; n < N; n++) {
      for (let oy = 0; oy < outH; oy++) {
        const iyBase = oy * sH;
        for (let ox = 0; ox < outW; ox++) {
          const ixBase = ox * sW;

          for (let f = 0; f < C_out; f++) {
            let sum = layer.bias[f];

            for (let ky = 0; ky < kH; ky++) {
              for (let kx = 0; kx < kW; kx++) {
                for (let c = 0; c < C_in; c++) {
                  const v = padded.get(n, iyBase + ky, ixBase + kx, c);
                  sum += v * k.get(ky, kx, c, f);
                }
              }
            }

            out.set(
              n,
              oy,
              ox,
              f,
              layer.activation ? layer.activation.f(sum) : sum
            );
          }
        }
      }
    }

    const cache: Conv2DCache = {
      type: "conv2d",
      input: current,
      padded,
      kernel: layer.kernel,
      outH,
      outW,
      postAct: out,
      strideH: sH,
      strideW: sW,
      padTop: layer.padTop,
      padLeft: layer.padLeft,
    };

    return { out, cache };
  }

  private poolForward(
    current: ActivationValue,
    layer: PoolLayer
  ): { out: ActivationValue; cache: LayerCache } {
    if (!(current instanceof Tensor4D)) {
      throw new Error("Pool layer expects a Tensor4D input");
    }

    const N = current.getN();
    const C = current.getC();
    const [, outH, outW] = layer.outShape;

    const out = new Tensor4D([N, outH, outW, C]);
    const mask = new Int32Array(N * outH * outW * C);

    let m = 0;

    for (let n = 0; n < N; n++) {
      for (let oy = 0; oy < outH; oy++) {
        const baseY = oy * layer.strideH;
        for (let ox = 0; ox < outW; ox++) {
          const baseX = ox * layer.strideW;

          for (let c = 0; c < C; c++) {
            let best = -Infinity;
            let bestIdx = -1;

            for (let wy = 0; wy < layer.windowH; wy++) {
              for (let wx = 0; wx < layer.windowW; wx++) {
                const iy = baseY + wy;
                const ix = baseX + wx;

                const v = current.get(n, iy, ix, c);
                if (v > best) {
                  best = v;
                  bestIdx = current.indexOf(n, iy, ix, c);
                }
              }
            }

            out.set(n, oy, ox, c, best);
            mask[m++] = bestIdx;
          }
        }
      }
    }

    const cache: PoolCache = {
      type: "pool",
      input: current,
      windowH: layer.windowH,
      windowW: layer.windowW,
      strideH: layer.strideH,
      strideW: layer.strideW,
      outH,
      outW,
      mask,
    };

    return { out, cache };
  }

  private flattenForward(current: ActivationValue): {
    out: ActivationValue;
    cache: LayerCache;
  } {
    if (!(current instanceof Tensor4D)) {
      throw new Error("Flatten expects a Tensor4D");
    }

    const [N, H, W, C] = current.shape;
    const cache: LayerCache = { type: "flatten", N, H, W, C };

    if (N !== 1) {
      throw new Error("Flatten only supports a batch size of 1");
    }
    const out = current.slice(0, H * W * C);

    return { out, cache };
  }

  // Build Layers

  private buildInputLayer(cfg: InputConfig): InputLayer {
    return {
      type: "input",
      shape: cfg.shape,
    };
  }

  private buildDenseLayer(cfg: DenseConfig, inputSize: number): DenseLayer {
    const outputSize = cfg.size;
    const weights = new Float32Array(inputSize * outputSize);
    const bias = new Float32Array(outputSize);

    // Xavier init
    const scale = Math.sqrt(2 / (inputSize + outputSize)); // sigmoid
    for (let i = 0; i < weights.length; i++) {
      weights[i] = (Math.random() - 0.5) * scale;
    }

    for (let i = 0; i < bias.length; i++) {
      bias[i] = (Math.random() - 0.5) * 0.1;
    }

    return {
      type: "dense",
      inputSize,
      outputSize,
      weights,
      bias,
      activation: cfg.activation ? activations[cfg.activation] : undefined,
    };
  }

  private buildPoolLayer(cfg: PoolConfig, prev: Layer): PoolLayer {
    const [N, H, W, C] = this.inferOutputShape(prev);

    const windowH = cfg.size[0];
    const windowW = cfg.size[1];
    const strideH = cfg.stride?.[0] ?? windowH;
    const strideW = cfg.stride?.[1] ?? windowW;

    const outH = Math.floor((H - windowH) / strideH) + 1;
    const outW = Math.floor((W - windowW) / strideW) + 1;

    return {
      type: "pool",
      windowH,
      windowW,
      strideH,
      strideW,
      channels: C,
      outShape: [N, outH, outW, C],
    };
  }

  private buildConv2DLayer(cfg: Conv2DConfig, prev: Layer): Conv2DLayer {
    const [N, H_in, W_in, C_in] = this.inferOutputShape(prev);
    const [kH, kW] = cfg.kernel;
    const C_out = cfg.filters;
    const strideH = cfg.stride?.[0] ?? 1;
    const strideW = cfg.stride?.[1] ?? 1;

    let padTop = 0;
    let padBottom = 0;
    let padLeft = 0;
    let padRight = 0;

    if (cfg.padding === "same") {
      const padH = Math.max(0, (H_in - 1) * strideH + kH - H_in);
      const padW = Math.max(0, (W_in - 1) * strideW + kW - W_in);
      padTop = Math.floor(padH / 2);
      padBottom = padH - padTop;
      padLeft = Math.floor(padW / 2);
      padRight = padW - padLeft;
    }

    const H_out = Math.floor((H_in + padTop + padBottom - kH) / strideH) + 1;
    const W_out = Math.floor((W_in + padLeft + padRight - kW) / strideW) + 1;

    const kernel = new Tensor4D([kH, kW, C_in, C_out]);
    const bias = new Float32Array(C_out);

    for (let i = 0; i < bias.length; i++) {
      bias[i] = 0;
    }

    kernel.setEach(() => (Math.random() - 0.5) * 0.1);

    return {
      type: "conv2d",
      kernel,
      bias,
      strideH,
      strideW,
      padTop,
      padBottom,
      padLeft,
      padRight,
      activation: cfg.activation ? activations[cfg.activation] : undefined,
      outShape: [N, H_out, W_out, C_out],
    };
  }

  // Helpers

  private getOutputSizeOf(layer: Layer): number {
    switch (layer.type) {
      case "input": {
        const [_, H, W, C] = layer.shape;
        return H * W * C;
      }
      case "dense": {
        return layer.outputSize;
      }
      case "flatten": {
        return layer.size;
      }
      case "conv2d":
      case "pool": {
        const [_, H, W, C] = layer.outShape;
        return H * W * C;
      }
    }
  }

  private inferOutputShape(layer: Layer): [number, number, number, number] {
    switch (layer.type) {
      case "input": {
        return layer.shape;
      }
      case "dense": {
        return [1, 1, 1, layer.outputSize];
      }
      case "conv2d": {
        return layer.outShape;
      }
      case "flatten": {
        return [1, 1, 1, layer.size];
      }
      case "pool": {
        return layer.outShape;
      }
    }
  }
}
