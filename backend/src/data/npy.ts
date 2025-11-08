import fs from "node:fs/promises";

export interface NpyHeader {
  descr: string;
  fortranOrder: boolean;
  shape: [number, number];
  dataOffset: number;
}

/**
 * Skip the 80-byte header; treat remainder as uint8 [*, 784].
 */
export async function readNpyHeader(filePath: string): Promise<NpyHeader> {
  const stats = await fs.stat(filePath);
  const HEADER_BYTES = 80;
  const IMG_SIZE = 28 * 28;
  const dataBytes = stats.size - HEADER_BYTES;

  if (dataBytes % IMG_SIZE !== 0) {
    throw new Error(
      `Unexpected file length for ${filePath}: ${stats.size} bytes (not multiple of 784 after header)`
    );
  }

  const numSamples = dataBytes / IMG_SIZE;

  return {
    descr: "|u1",
    fortranOrder: false,
    shape: [numSamples, IMG_SIZE],
    dataOffset: HEADER_BYTES,
  };
}
