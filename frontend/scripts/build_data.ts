import * as fs from "fs/promises";
import path from "path";

//
//    This builds the binary training and testing data.
//    Each file is a list of bitmap images (28x28) from
//    the google quickdraw dataset.
//
//    I invert the colors so it is black lines on a white
//    background. and then I stitch random x amount of
//    doodles together into a single binary.
//
//    The output binary files are still 1 byte per pixel
//    bitmap images to train/test against.
//
//
//

const imgW = 28;
const imgH = 28;
const imgByteSize = imgW * imgH;
const npyHeaderSizeBytes = 80;
async function extractDoodles(
  file: string,
  trainingCount: number,
  testingCount: number
): Promise<[Uint8Array, Uint8Array]> {
  const imageData = (await fs.readFile(file)).buffer;
  const bytes = new Uint8Array(imageData);
  const imgCount = (bytes.byteLength - npyHeaderSizeBytes) / imgByteSize;
  if (imgCount < 1) {
    throw new Error(`File: ${file} does not contain data`);
  }
  if (trainingCount + testingCount > imgCount) {
    throw new Error(
      `data set are too large for file: ${file} at count: ${imgCount}, and sizes: train: "${trainingCount}", test: "${testingCount}"`
    );
  }
  const out_test = new Uint8Array(testingCount * imgByteSize);
  const seen: number[] = [];
  const out_train = new Uint8Array(trainingCount * imgByteSize);

  // Training Data
  for (let i = 0; i < trainingCount; i++) {
    let doodleId = 0;
    while (seen.includes(doodleId)) {
      doodleId = Math.floor(Math.random() * imgCount);
    }
    seen.push(doodleId);
    const firstImage = bytes.slice(
      npyHeaderSizeBytes + doodleId * imgByteSize,
      npyHeaderSizeBytes + doodleId * imgByteSize + imgByteSize
    );
    const offset = i * imgByteSize;
    for (let p = 0; p < imgByteSize; p++) {
      out_train[offset + p] = 255 - firstImage[p];
    }
  }

  // Test Data
  for (let i = 0; i < testingCount; i++) {
    let doodleId = 0;
    while (seen.includes(doodleId)) {
      doodleId = Math.floor(Math.random() * imgCount);
    }
    seen.push(doodleId);
    const firstImage = bytes.slice(
      npyHeaderSizeBytes + doodleId * imgByteSize,
      npyHeaderSizeBytes + doodleId * imgByteSize + imgByteSize
    );
    const offset = i * imgByteSize;
    for (let p = 0; p < imgByteSize; p++) {
      out_test[offset + p] = 255 - firstImage[p];
    }
  }

  return [out_test, out_train];
}

async function prepareDoodleSet(
  name: string,
  trainingCount: number,
  testingCount: number
) {
  const cwd = process.cwd();
  const dataFile = path.join(cwd, "datasets", name + ".npy");
  const [test, train] = await extractDoodles(
    dataFile,
    trainingCount,
    testingCount
  );

  const trainFile = path.join(cwd, "public/data", name + "_train.bin");
  const testFile = path.join(cwd, "public/data", name + "_test.bin");

  await fs.writeFile(testFile, test);
  await fs.writeFile(trainFile, train);
}

async function main() {
  const data: Array<[string, number, number]> = [
    ["butterfly", 800, 200],
    ["rainbow", 800, 200],
    ["cat", 800, 200],
  ];

  for (const [name, train, test] of data) {
    await prepareDoodleSet(name, train, test);
  }
}
main();
