export class Matrix {
  private rows: number;
  private cols: number;
  private data: Float32Array;

  constructor(rows: number, cols: number) {
    this.rows = rows;
    this.cols = cols;
    this.data = new Float32Array(rows * cols);
  }

  public get values() {
    return this.data;
  }

  public static fromArray(arr: number[]): Matrix {
    const m = new Matrix(arr.length, 1);
    for (let i = 0; i < arr.length; i++) {
      m.data[i] = arr[i];
    }
    return m;
  }

  public static multiply(a: Matrix, b: Matrix): Matrix {
    if (a.cols !== b.rows) throw new Error("size mismatch");

    const result = new Matrix(a.rows, b.cols);

    const aRows = a.rows;
    const aCols = a.cols;
    const bCols = b.cols;

    const aData = a.data;
    const bData = b.data;
    const rData = result.data;

    for (let i = 0; i < aRows; i++) {
      const rowOffset = i * aCols;
      for (let j = 0; j < bCols; j++) {
        let sum = 0;
        for (let k = 0; k < aCols; k++) {
          sum += aData[rowOffset + k] * bData[k * bCols + j];
        }
        rData[i * bCols + j] = sum;
      }
    }

    return result;
  }

  public static subtract(a: Matrix, b: Matrix): Matrix {
    const { rows, cols } = a;
    if (a.rows !== b.rows) {
      throw new Error("unable to subtract matrices: size mismatch");
    }
    if (a.cols !== b.cols) {
      throw new Error("unable to subtract matrices: size mismatch");
    }

    const result = new Matrix(a.rows, a.cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result.data[i * a.cols + j] =
          a.data[i * a.cols + j] - b.data[i * a.cols + j];
      }
    }

    return result;
  }

  public print() {
    console.table(this.data);
  }

  public map(fn: (n: number) => number): void {
    const { rows, cols } = this;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        this.data[i * cols + j] = fn(this.data[i * cols + j]);
      }
    }
  }

  public clone(fn?: (n: number) => number): Matrix {
    const { rows, cols } = this;
    const out = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        out.data[i * cols + j] = fn
          ? fn(this.data[i * cols + j])
          : this.data[i * cols + j];
      }
    }
    return out;
  }

  public randomize() {
    const { rows, cols } = this;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        this.data[i * cols + j] = Math.random() * 2 - 1;
      }
    }
  }

  public transpose(): Matrix {
    const { cols, rows } = this;
    const out = new Matrix(cols, rows);
    for (let i = 0; i < out.rows; i++) {
      for (let j = 0; j < out.cols; j++) {
        out.data[i * out.cols + j] = this.data[j * this.cols + i];
      }
    }
    return out;
  }

  public mulScalar(scalar: number): void {
    const { rows, cols } = this;
    for (let i = 0; i < rows * cols; i++) {
      this.data[i] *= scalar;
    }
  }

  public mul(m: Matrix) {
    const { rows, cols } = this;
    for (let i = 0; i < rows * cols; i++) {
      this.data[i] *= m.data[i];
    }
  }

  public add(m: Matrix) {
    const { rows, cols } = this;
    for (let i = 0; i < rows * cols; i++) {
      this.data[i] += m.data[i];
    }
  }

  public addScalar(scalar: number): void {
    const { rows, cols } = this;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        this.data[i * cols + j] += scalar;
      }
    }
  }
}
