export function random(): number {
  const r = Math.random();
  return r * 2 - 1;
}

export function sign(n: number): number {
  if (n >= 0) {
    return 1;
  }
  return -1;
}

export function remap(
  r1: [number, number],
  r2: [number, number],
  v: number
): number {
  return ((v - r1[0]) / (r1[1] - r1[0])) * (r2[1] - r2[0]) + r2[0];
}

// Return [H][W][C]
export function extractChannels(
  imageData: ImageData,
  channels: number
): number[][][] {
  const h = imageData.height;
  const w = imageData.width;
  const out: number[][][] = [];
  for (let i = 0; i < h * w; i += 4) {}
}
