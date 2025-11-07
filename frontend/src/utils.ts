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
// @TODO: Return a 3D Tensor?
// then merge that into a larger 4D tensor?
export function extractChannels(
  h: number,
  w: number,
  chan: number,
  data: ImageData // @TODO: use a uint8 array?
): number[][][] {
  const out: number[][][] = [];
  for (let y = 0; y < h; y++) {
    out[y] = [];
    for (let x = 0; x < w; x++) {
      out[y][x] = [];
      const pIdx = 4 * (y * w + x);
      for (let c = 0; c < chan; c++) {
        out[y][x][c] = data.data[pIdx + c];
      }
    }
  }
  return out;
}
