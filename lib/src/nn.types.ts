import type { ActivationEntry, activations } from "./activation";
import type { losses } from "./losses";
import type { Tensor4D } from "./Tensor";

export type Vec4 = [number, number, number, number];

export type LayerType = "input" | "dense" | "conv2d" | "pool" | "flatten";
export type ActivationValue = Tensor4D | Float32Array;

//
//  Configs
//

export interface InputConfig {
  type: "input";
  shape: Vec4;
  activation?: keyof typeof activations;
}

export interface DenseConfig {
  type: "dense";
  size: number;
  activation?: keyof typeof activations;
}

export interface Conv2DConfig {
  type: "conv2d";
  filters: number;
  kernel: [number, number];
  stride?: [number, number];
  padding?: "valid" | "same";
  activation?: keyof typeof activations;
}

export interface PoolConfig {
  type: "pool";
  size: [number, number];
  stride?: [number, number];
}

export interface FlattenConfig {
  type: "flatten";
}

export type LayerConfig =
  | InputConfig
  | DenseConfig
  | Conv2DConfig
  | PoolConfig
  | FlattenConfig;

//
// Layers
//

export interface InputLayer {
  type: "input";
  shape: Vec4; // N, H, W, C
}

export interface DenseLayer {
  type: "dense";
  inputSize: number;
  outputSize: number;
  weights: Float32Array;
  bias: Float32Array;
  activation?: ActivationEntry;
}

export interface Conv2DLayer {
  type: "conv2d";
  kernel: Tensor4D;
  bias: Float32Array;
  strideH: number;
  strideW: number;
  padTop: number;
  padBottom: number;
  padLeft: number;
  padRight: number;
  outShape: Vec4;
  activation?: ActivationEntry;
}

export interface PoolLayer {
  type: "pool";
  windowH: number;
  windowW: number;
  strideH: number;
  strideW: number;
  channels: number;
  outShape: Vec4;
}

export interface FlattenLayer {
  type: "flatten";
  size: number;
}

export type Layer =
  | DenseLayer
  | Conv2DLayer
  | PoolLayer
  | FlattenLayer
  | InputLayer;

export interface NeuralNetworkOptions {
  learningRate?: number;
  loss?: keyof typeof losses;
  debug?: boolean;
}

//
// Cache
//

export interface InputCache {
  type: "input";
  shape: Vec4;
}

export interface DenseCache {
  type: "dense";
  input: Float32Array; // input vector to the dense layer
  z: Float32Array; // pre-activation values
}

export interface FlattenCache {
  type: "flatten";
  N: number;
  H: number;
  W: number;
  C: number;
}

export interface PoolCache {
  type: "pool";
  input: Tensor4D;
  windowH: number;
  windowW: number;
  strideH: number;
  strideW: number;
  outH: number;
  outW: number;
  mask: Int32Array; // length: N*outH*outW*C
}

export interface Conv2DCache {
  type: "conv2d";
  input: Tensor4D; // original input (un-padded)
  padded: Tensor4D; // padded input used for forward
  kernel: Tensor4D; // weights
  postAct: Tensor4D;
  outH: number;
  outW: number;
  strideH: number;
  strideW: number;
  padTop: number;
  padLeft: number;
}

export type LayerCache =
  | InputCache
  | DenseCache
  | FlattenCache
  | PoolCache
  | Conv2DCache;

//
//  Serialization
//
//      @TODO: Build a single base input, dense, etc. types
//             that the config, layer, cache and serialized
//             objects can build on.
//

export interface InputSerialized {
  type: "input";
  shape: Vec4;
}

export interface DenseSerialized {
  type: "dense";
  inputSize: number;
  outputSize: number;
  activation?: string;
  weights: number[]; // row major [out, in]
  bias: number[];
}

export interface Conv2DSerialized {
  type: "conv2d";
  strideH: number;
  strideW: number;
  padTop: number;
  padBottom: number;
  padLeft: number;
  padRight: number;
  activation?: string;
  kernel: number[]; // [kH, kW, C_in, C_out]
  kernelShape: Vec4;
  bias: number[];
  outShape: Vec4;
}

export interface PoolSerialized {
  type: "pool";
  windowH: number;
  windowW: number;
  strideH: number;
  strideW: number;
  channels: number;
  outShape: Vec4;
}

export interface FlattenSerialized {
  type: "flatten";
  size: number;
}

export type LayerSerialized =
  | InputSerialized
  | DenseSerialized
  | Conv2DSerialized
  | PoolSerialized
  | FlattenSerialized;

export type NNCheckpoint = {
  learningRate: number;
  layers: Array<LayerSerialized>;
};
