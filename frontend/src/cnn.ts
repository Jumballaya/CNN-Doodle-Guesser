import "./style.css";

async function loadImage(path: string): Promise<HTMLImageElement> {
  const img = new Image();
  return new Promise((res) => {
    img.src = path;
    img.onload = () => {
      res(img);
    };
  });
}

function sampleClamped(
  data: ImageDataArray,
  width: number,
  height: number,
  x: number,
  y: number
) {
  x = Math.max(0, Math.min(width - 1, x));
  y = Math.max(0, Math.min(height - 1, y));
  return data[4 * (y * width + x)];
}

function convolute(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  kernel: number[][]
) {
  const kernelSize = kernel.length;
  const radius = Math.floor(kernelSize / 2);

  const imageData = ctx.getImageData(0, 0, width, height);
  const out = new Uint8ClampedArray(width * height * 4);

  // compute total kernel weight
  let kernelWeight = 0;
  for (let y = 0; y < kernelSize; y++) {
    for (let x = 0; x < kernelSize; x++) {
      kernelWeight += kernel[y][x];
    }
  }
  if (kernelWeight === 0) kernelWeight = 1;

  let outIndex = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;

      for (let ky = 0; ky < kernelSize; ky++) {
        for (let kx = 0; kx < kernelSize; kx++) {
          const sampleY = y + ky - radius;
          const sampleX = x + kx - radius;

          const sample = sampleClamped(
            imageData.data,
            width,
            height,
            sampleX,
            sampleY
          );

          sum += sample * kernel[ky][kx];
        }
      }

      const value = sum / kernelWeight;

      out[outIndex++] = value;
      out[outIndex++] = value;
      out[outIndex++] = value;
      out[outIndex++] = 255;
    }
  }

  ctx.putImageData(new ImageData(out, width, height), 0, 0);
}

async function main() {
  const width = 256;
  const height = 256;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const birdsImage = await loadImage("birds.png");
  ctx.drawImage(birdsImage, 0, 0);
  document.body.appendChild(canvas);

  const gaussian3 = [
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1],
  ];

  const gaussian5 = [
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1],
  ];

  const gaussian7 = [
    [0, 0, 1, 2, 1, 0, 0],
    [0, 3, 13, 22, 13, 3, 0],
    [1, 13, 59, 97, 59, 13, 1],
    [2, 22, 97, 159, 97, 22, 2],
    [1, 13, 59, 97, 59, 13, 1],
    [0, 3, 13, 22, 13, 3, 0],
    [0, 0, 1, 2, 1, 0, 0],
  ];

  const box5 = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
  ];

  const sharpen3 = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ];

  const sharpen5 = [
    [0, 0, -1, 0, 0],
    [0, -1, -2, -1, 0],
    [-1, -2, 25, -2, -1],
    [0, -1, -2, -1, 0],
    [0, 0, -1, 0, 0],
  ];

  const emboss3 = [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
  ];

  const sobelX = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ];

  const sobelY = [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
  ];

  const laplacian8 = [
    [1, 1, 1],
    [1, -8, 1],
    [1, 1, 1],
  ];

  const gaussian9 = [
    [1, 4, 7, 10, 13, 10, 7, 4, 1],
    [4, 12, 26, 33, 39, 33, 26, 12, 4],
    [7, 26, 55, 71, 83, 71, 55, 26, 7],
    [10, 33, 71, 91, 106, 91, 71, 33, 10],
    [13, 39, 83, 106, 123, 106, 83, 39, 13],
    [10, 33, 71, 91, 106, 91, 71, 33, 10],
    [7, 26, 55, 71, 83, 71, 55, 26, 7],
    [4, 12, 26, 33, 39, 33, 26, 12, 4],
    [1, 4, 7, 10, 13, 10, 7, 4, 1],
  ];

  convolute(ctx, width, height, gaussian9);
}
main();
