export type ImageSize = [number, number, number]; // [H, W, C]

export interface SplitInfo {
  count: number;
  bin: string; // e.g. "frontend/public/data/cat_train.bin"
  dtype: "u8";
  imageSize: ImageSize; // [28, 28, 1]
  offsets: number[]; // byte offsets into .bin
}

export interface SourceInfo {
  url: string; // QuickDraw .npy URL
  cacheFile: string; // relative cache file e.g. ".cache/quickdraw/cat.npy"
}

export interface ClassInfo {
  id: number;
  name: string; // "cat"
  displayName: string; // "Cat"
  oneHot: number[]; // length = #classes
  train: SplitInfo;
  test: SplitInfo;
  source: SourceInfo;
}

export interface DatasetManifest {
  version: number;
  classes: ClassInfo[];
}
