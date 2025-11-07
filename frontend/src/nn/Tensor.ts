type Vec4 = [number, number, number, number];

export class Tensor4D {
  private buffer: Float32Array;
  private shape: Vec4;
  private strides: Vec4;

  constructor(shape: Vec4, data?: Float32Array) {
    this.shape = [shape[0], shape[1], shape[2], shape[3]];
    const f32Size = shape[0] * shape[1] * shape[2] * shape[3];
    if (data && data.length !== f32Size) {
      throw new Error("buffer size mismatch");
    }
    this.buffer = data ? data.slice(0) : new Float32Array(f32Size);

    const [_, h, w, c] = this.shape;
    this.strides = [h * w * c, w * c, c, 1];
  }

  public get(n: number, y: number, x: number, c: number): number {
    return this.buffer[this.idx(n, y, x, c)];
  }

  public set(n: number, y: number, x: number, c: number, v: number): void {
    this.buffer[this.idx(n, y, x, c)] = v;
  }

  public indexOf(n: number, y: number, x: number, c: number): number {
    return this.idx(n, y, x, c);
  }

  public pool2d(
    windowH: number,
    windowW: number,
    strideH: number,
    strideW: number,
    type: "max" | "avg"
  ): Tensor4D {
    const [N, H, W, C] = this.shape;
    const outH = Math.floor((H - windowH) / strideH) + 1;
    const outW = Math.floor((W - windowW) / strideW) + 1;
    const out = new Tensor4D([N, outH, outW, C]);

    for (let n = 0; n < N; n++) {
      for (let y = 0; y < outH; y++) {
        for (let x = 0; x < outW; x++) {
          const patch = this.sliceWindow(
            n,
            y * strideH,
            x * strideW,
            windowH,
            windowW
          );
          for (let c = 0; c < C; c++) {
            let v = type === "max" ? -Infinity : 0;
            for (let i = c; i < patch.length; i += C) {
              v = type === "max" ? Math.max(v, patch[i]) : v + patch[i];
            }
            out.set(n, y, x, c, type === "avg" ? v / (windowH * windowW) : v);
          }
        }
      }
    }

    return out;
  }

  // Math Operations

  public fill(value: number): void {
    this.buffer.fill(value);
  }

  public clone(): Tensor4D {
    const t = new Tensor4D(this.shape, this.buffer);
    return t;
  }

  // Shape operations

  // flatten to an array of batch data
  public flatten(): Float32Array[] {
    const out: Float32Array[] = [];

    for (let i = 0; i < this.shape[0]; i++) {
      out.push(
        this.buffer.slice(i * this.strides[0], (i + 1) * this.strides[0])
      );
    }

    return out;
  }

  public pad(
    padTop: number,
    padBottom: number,
    padLeft: number,
    padRight: number
  ): Tensor4D {
    const [N, H, W, C] = this.shape;
    const newH = H + padTop + padBottom;
    const newW = W + padLeft + padRight;

    const out = new Tensor4D([N, newH, newW, C]);
    for (let n = 0; n < N; n++) {
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          for (let c = 0; c < C; c++) {
            out.set(n, y + padTop, x + padLeft, c, this.get(n, y, x, c));
          }
        }
      }
    }

    return out;
  }

  // This grabs a window (grid) of data from the buffer
  // and puts it into a flat array.
  public sliceWindow(
    n: number,
    y: number,
    x: number,
    windowH: number,
    windowW: number
  ): Float32Array {
    const [_, _H, _W, C] = this.shape;
    const out = new Float32Array(windowH * windowW * C);

    let i = 0;
    for (let dy = 0; dy < windowH; dy++) {
      for (let dx = 0; dx < windowW; dx++) {
        const iy = y + dy;
        const ix = x + dx;
        for (let c = 0; c < C; c++) {
          out[i++] = this.get(n, iy, ix, c);
        }
      }
    }

    return out;
  }

  //
  // @TODO: Later ----
  //            transpose
  //            broadcasting
  //            elementwise add/mul
  //            reduce max
  //            reduce mean
  //

  private idx(n: number, y: number, x: number, c: number): number {
    const [sN, sH, sW, sC] = this.strides;
    return n * sN + y * sH + x * sW + c * sC;
  }
}
