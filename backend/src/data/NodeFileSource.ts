import fs from "node:fs/promises";
import { Buffer } from "node:buffer";
import path from "node:path";

//
//  @TODO: NodeJS only, no running the trainer in the
//         browser so I don't need this level of abstraction
//

export type SampleView = Float32Array;

export interface Batch {
  x: Float32Array; // [B, IMG_SIZE] flattened
  y: Float32Array; // [B, C] one-hot labels
}

export interface IDataSource {
  readSample(
    binPath: string,
    offset: number,
    len: number
  ): Promise<Float32Array>;
  close?(): Promise<void>;
}

/**
 * Node-only source that:
 *   - Caches open file descriptors
 *   - Reads Uint8 data from .bin
 *   - Normalizes to [0,1]
 */
export class NodeFileSource implements IDataSource {
  private fhCache = new Map<string, fs.FileHandle>();

  constructor(private rootDir: string = process.cwd()) {}

  private resolvePath(p: string): string {
    return path.isAbsolute(p) ? p : path.resolve(this.rootDir, p);
  }

  async readSample(
    p: string,
    offset: number,
    len: number
  ): Promise<Float32Array> {
    const full = this.resolvePath(p);
    let fh = this.fhCache.get(full);
    if (!fh) {
      fh = await fs.open(full, "r");
      this.fhCache.set(full, fh);
    }

    const buf = Buffer.allocUnsafe(len);
    await fh.read(buf, 0, len, offset);

    const out = new Float32Array(len);
    for (let i = 0; i < len; i++) out[i] = buf[i] / 255;
    return out;
  }

  async close(): Promise<void> {
    for (const fh of this.fhCache.values()) {
      try {
        await fh.close();
      } catch {
        /* ignore */
      }
    }
    this.fhCache.clear();
  }
}
