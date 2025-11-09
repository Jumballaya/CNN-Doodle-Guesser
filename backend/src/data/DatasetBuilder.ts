import fs from "node:fs/promises";
import { open as openFd } from "node:fs/promises";
import { Buffer } from "node:buffer";
import path from "node:path";

import type { DatasetManifest, ClassInfo, SplitInfo } from "./manifest";
import { readNpyHeader } from "./npy.js";
import { NodeFileSource } from "./NodeFileSource.js";
import { BatchSampler } from "./BatchSampler";

const IMG_H = 28;
const IMG_W = 28;
const IMG_SIZE = IMG_H * IMG_W;
const DEFAULT_TRAIN_COUNT = 800;
const DEFAULT_TEST_COUNT = 200;
const DEFAULT_CACHE_DIR = ".cache/quickdraw";
const DEFAULT_DATA_DIR = ".cache/public";
const DEFAULT_MANIFEST = ".cache/public/quickdraw-3.json";

export interface ClassSpec {
  name: string; // "cat"
  trainCount: number; // e.g. 800
  testCount: number; // e.g. 200
  displayName?: string; // optional override
}

export interface DatasetBuilderOptions {
  cacheDir?: string;
  dataDir?: string;
  manifestPath?: string;
  trainCount?: number;
  testCount?: number;
}

/**
 * Format class display name, e.g. cat -> Cat, aircraft carrier -> Aircraft Carrier
 */
const capitalize = (s: string) => `${s[0].toUpperCase()}${s.slice(1)}`;
const displayName = (s: string) => s.split("_").map(capitalize).join(" ");
const snakeCase = (s: string) => s.replace(" ", "_");

/**
 * Convenience wrapper over manifest + NodeFileSource.
 */
export class QuickdrawDataset {
  public manifest: DatasetManifest;
  private source;

  constructor(
    manifest: DatasetManifest,
    rootDir = process.cwd(),
    source = new NodeFileSource(rootDir)
  ) {
    this.manifest = manifest;
    this.source = source;
  }

  numClasses(): number {
    return this.manifest.classes.length;
  }

  createBatchSampler(split: "train" | "test", batchSize: number): BatchSampler {
    return new BatchSampler(this.manifest.classes, this.source, batchSize, {
      split,
      imgSize: IMG_SIZE,
    });
  }

  async close(): Promise<void> {
    await this.source.close?.();
  }
}

async function ensureDir(dir: string) {
  await fs.mkdir(dir, { recursive: true });
}

function quickdrawUrl(name: string): string {
  return `https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/${name.replace(
    "_",
    " "
  )}.npy`;
}

async function downloadIfNeeded(url: string, dest: string): Promise<void> {
  try {
    await fs.access(dest);
    return; // already exists
  } catch {
    // continue
  }

  await ensureDir(path.dirname(dest));
  const res = await fetch(url);
  if (!res.ok || !res.body) {
    throw new Error(
      `Failed to download ${url}: ${res.status} ${res.statusText}`
    );
  }

  // Simpler: buffer in memory once, then write
  const ab = await res.arrayBuffer();
  const buf = Buffer.from(ab);
  await fs.writeFile(dest, buf);
}

/**
 * Build the train/test .bin files and manifest for one class.
 */
async function buildClassBins(
  spec: ClassSpec,
  id: number,
  oneHotDim: number,
  opts: Required<DatasetBuilderOptions>
): Promise<ClassInfo> {
  const cacheDir = opts.cacheDir;
  const dataDir = opts.dataDir;

  const npyUrl = quickdrawUrl(spec.name);
  const cacheFile = path.join(cacheDir, `${spec.name}.npy`);
  await downloadIfNeeded(npyUrl, cacheFile);

  const header = await readNpyHeader(cacheFile);
  const [numSamples, cols] = header.shape;
  if (cols !== IMG_SIZE) {
    throw new Error(
      `Unexpected cols=${cols} in ${cacheFile}, expected ${IMG_SIZE}`
    );
  }

  const needed = spec.trainCount + spec.testCount;
  if (needed > numSamples) {
    throw new Error(
      `Not enough samples in ${spec.name}: requested ${needed}, have ${numSamples}`
    );
  }

  // Shuffle indices
  const indices = Array.from({ length: numSamples }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = (Math.random() * (i + 1)) | 0;
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const trainIdx = indices.slice(0, spec.trainCount);
  const testIdx = indices.slice(
    spec.trainCount,
    spec.trainCount + spec.testCount
  );

  const trainRel = path.join(dataDir, `${spec.name}_train.bin`);
  const testRel = path.join(dataDir, `${spec.name}_test.bin`);

  const trainAbs = path.resolve(process.cwd(), trainRel);
  const testAbs = path.resolve(process.cwd(), testRel);
  await ensureDir(path.dirname(trainAbs));
  await ensureDir(path.dirname(testAbs));

  const trainOffsets: number[] = [];
  const testOffsets: number[] = [];

  // Keep a file handle on the .npy so we can random-access images
  const fh = await openFd(cacheFile, "r");
  try {
    const sampleBuf = Buffer.allocUnsafe(IMG_SIZE);

    // helper to copy + invert one image into a writable file
    async function writeSplit(
      filePath: string,
      idxList: number[],
      offsets: number[]
    ) {
      const handle = await openFd(filePath, "w");
      try {
        let localOffset = 0;
        for (const idx of idxList) {
          const samplePos = header.dataOffset + idx * IMG_SIZE;
          await fh.read(sampleBuf, 0, IMG_SIZE, samplePos);
          // invert colors: 255 - pixel
          for (let i = 0; i < IMG_SIZE; i++) {
            sampleBuf[i] = 255 - sampleBuf[i];
          }
          offsets.push(localOffset);
          await handle.write(sampleBuf, 0, IMG_SIZE, localOffset);
          localOffset += IMG_SIZE;
        }
      } finally {
        await handle.close();
      }
    }

    await writeSplit(trainAbs, trainIdx, trainOffsets);
    await writeSplit(testAbs, testIdx, testOffsets);
  } finally {
    await fh.close();
  }

  const oneHot = Array(oneHotDim).fill(0);
  oneHot[id] = 1;

  const mkSplit = (
    relPath: string,
    count: number,
    offsets: number[]
  ): SplitInfo => ({
    count,
    bin: relPath.replace(/\\/g, "/"),
    dtype: "u8",
    imageSize: [IMG_H, IMG_W, 1],
    offsets,
  });

  const classInfo: ClassInfo = {
    id,
    name: spec.name,
    displayName: spec.displayName ?? displayName(spec.name),
    oneHot,
    train: mkSplit(trainRel, spec.trainCount, trainOffsets),
    test: mkSplit(testRel, spec.testCount, testOffsets),
    source: {
      url: npyUrl,
      cacheFile: cacheFile.replace(/\\/g, "/"),
    },
  };

  return classInfo;
}

export class DatasetBuilder {
  /**
   * Create datasets and manifest for the given class specs.
   * Overload: accepts either string[] or ClassSpec[].
   */
  static async create(
    namesOrSpecs: Array<string | ClassSpec>,
    options: DatasetBuilderOptions = {}
  ): Promise<QuickdrawDataset> {
    const {
      cacheDir = DEFAULT_CACHE_DIR,
      dataDir = DEFAULT_DATA_DIR,
      manifestPath = DEFAULT_MANIFEST,
    } = options;
    const trainCount = options.trainCount ?? DEFAULT_TRAIN_COUNT;
    const testCount = options.testCount ?? DEFAULT_TEST_COUNT;

    const specs: ClassSpec[] = namesOrSpecs.map((n) =>
      typeof n === "string"
        ? { name: snakeCase(n), trainCount, testCount } // default counts
        : n
    );

    const oneHotDim = specs.length;
    const classes: ClassInfo[] = [];
    for (let i = 0; i < specs.length; i++) {
      const info = await buildClassBins(specs[i], i, oneHotDim, {
        cacheDir,
        dataDir,
        manifestPath,
        trainCount,
        testCount,
      });
      classes.push(info);
    }

    const manifest: DatasetManifest = {
      version: 1,
      classes,
    };

    const manifestAbs = path.resolve(process.cwd(), manifestPath);
    await ensureDir(path.dirname(manifestAbs));
    await fs.writeFile(manifestAbs, JSON.stringify(manifest, null, 2), "utf-8");

    return new QuickdrawDataset(manifest);
  }

  /**
   * Load an existing manifest from disk and create a dataset.
   */
  static async fromManifest(
    manifestPath: string,
    rootDir = process.cwd()
  ): Promise<QuickdrawDataset> {
    const manifestAbs = path.resolve(rootDir, manifestPath);
    const raw = await fs.readFile(manifestAbs, "utf-8");
    const manifest = JSON.parse(raw) as DatasetManifest;
    return new QuickdrawDataset(manifest, rootDir);
  }
}
