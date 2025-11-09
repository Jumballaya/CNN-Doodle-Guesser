import {
  NeuralNetwork,
  Tensor4D,
  type ActivationValue,
  type NNCheckpoint,
} from "@doodle/lib";

const classes = [
  "cat",
  "butterfly",
  "rainbow",
  "banana",
  "flower",
  "ladder",
  "mushroom",
  "snowman",
  "sword",
  "nose",
];

export class DrawingApp {
  public ctx: CanvasRenderingContext2D;
  public width: number;
  public height: number;

  private mouseDown = false;
  private mouseX = 0;
  private mouseY = 0;

  private snapshots: ImageData[] = [];
  private redosnaps: ImageData[] = [];

  private resizeCanvas: HTMLCanvasElement;
  private resizeCtx: CanvasRenderingContext2D;

  private nn: NeuralNetwork;

  // elements
  public canvas: HTMLCanvasElement;
  private guessBtn: HTMLButtonElement;
  private undoBtn: HTMLButtonElement;
  private redoBtn: HTMLButtonElement;
  private container: HTMLElement;

  constructor(nn: NeuralNetwork) {
    this.nn = nn;
    this.width = 28 * 10 * 2;
    this.height = 28 * 10 * 2;

    this.guessBtn = this.createGuessBtn();
    [this.canvas, this.resizeCanvas] = this.createCanvas();
    this.container = this.createContainer();
    this.undoBtn = this.createUndoBtn();
    this.redoBtn = this.createRedoBtn();
    const nav = document.createElement("div");
    nav.classList.add("nav-container");
    nav.appendChild(this.undoBtn);
    nav.appendChild(this.redoBtn);
    document.body.appendChild(this.container);
    this.container.appendChild(nav);
    this.container.appendChild(this.canvas);
    this.container.appendChild(this.guessBtn);
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true })!;
    this.resizeCtx = this.resizeCanvas.getContext("2d", {
      willReadFrequently: true,
    })!;

    document.body.addEventListener("mousedown", (e) => {
      if (e.target !== this.canvas) {
        return;
      }
      this.mouseDown = true;
      this.snapshots.push(this.ctx.getImageData(0, 0, this.width, this.height));
      this.undoBtn.disabled = false;
      this.redoBtn.disabled = true;
      this.redosnaps = [];
    });
    document.body.addEventListener("mouseup", () => {
      this.mouseDown = false;
    });

    let controlPressed = false;
    document.body.addEventListener("keydown", (e) => {
      if (e.key === "Control") {
        controlPressed = true;
      }
      if (e.key === "z" && controlPressed) {
        this.undo();
      }
      if (e.key === "y" && controlPressed) {
        this.redo();
      }
    });
    document.body.addEventListener("keyup", (e) => {
      if (e.key === "Control") {
        controlPressed = false;
      }
    });

    this.canvas.addEventListener("mousemove", this.onMoveMouse.bind(this));
  }

  public static async FromSerialized(path: string) {
    try {
      const data: NNCheckpoint = await (await fetch(path)).json();
      const nn = NeuralNetwork.fromCheckpoint(data);
      return new DrawingApp(nn);
    } catch (e) {
      throw e;
    }
  }

  public getData() {
    const data: number[] = new Array(28 * 28);
    this.resizeCtx.clearRect(0, 0, 28, 28);
    this.resizeCtx.drawImage(
      this.canvas,
      0,
      0,
      this.width,
      this.height,
      0,
      0,
      28,
      28
    );
    const imageData = this.resizeCtx.getImageData(0, 0, 28, 28).data;
    for (let i = 0; i < 28 * 28; i++) {
      data[i] = (255 - imageData[i * 4 + 3]) / 255;
    }
    console.log(data);
    return data;
  }

  private onMoveMouse(e: MouseEvent) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouseX = e.clientX - rect.x;
    this.mouseY = e.clientY - rect.y;
    if (this.mouseDown) {
      this.ctx.fillStyle = "black";
      this.ctx.beginPath();
      this.ctx.arc(this.mouseX, this.mouseY, 10, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  private undo() {
    const screen = this.ctx.getImageData(0, 0, this.width, this.height);
    const snap = this.snapshots.pop();
    if (snap) {
      this.ctx.putImageData(snap, 0, 0);
      this.redosnaps.push(screen);
      this.redoBtn.disabled = false;
    }
    if (this.snapshots.length === 0) {
      this.undoBtn.disabled = true;
    }
  }

  private redo() {
    const screen = this.ctx.getImageData(0, 0, this.width, this.height);
    const last = this.redosnaps.pop();
    if (last) {
      this.ctx.putImageData(last, 0, 0);
      this.snapshots.push(screen);
      this.undoBtn.disabled = false;
    }
    if (this.redosnaps.length === 0) {
      this.redoBtn.disabled = true;
    }
  }

  private createGuessBtn(): HTMLButtonElement {
    const guessBtn = document.createElement("button");
    guessBtn.innerText = "Guess!";
    guessBtn.classList.add("btn");
    guessBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const data = this.getData();
      const guess = this.nn.guess(
        new Tensor4D([1, 28, 28, 1], new Float32Array(data))
      );
      console.log(guess);
      const output = argMax(guess);
      console.log(`${classes[output]}?`);
    });
    return guessBtn;
  }

  private createRedoBtn(): HTMLButtonElement {
    const undoBtn = document.createElement("button");
    undoBtn.innerText = "Re-do >>";
    undoBtn.classList.add("btn");
    undoBtn.disabled = this.snapshots.length === 0;
    undoBtn.addEventListener("click", (e) => {
      e.preventDefault();
      this.redo();
    });
    return undoBtn;
  }

  private createUndoBtn(): HTMLButtonElement {
    const redoBtn = document.createElement("button");
    redoBtn.innerText = "<< Un-do";
    redoBtn.classList.add("btn");
    redoBtn.disabled = this.redosnaps.length === 0;

    redoBtn.addEventListener("click", (e) => {
      e.preventDefault();
      this.undo();
    });
    return redoBtn;
  }

  private createContainer(): HTMLElement {
    const container = document.createElement("main");
    container.classList.add("drawing-app");
    return container;
  }

  private createCanvas(): [HTMLCanvasElement, HTMLCanvasElement] {
    const canvas = document.createElement("canvas");
    const resizeCanvas = document.createElement("canvas");
    resizeCanvas.width = 28;
    resizeCanvas.height = 28;
    canvas.width = this.width;
    canvas.height = this.height;
    return [canvas, resizeCanvas];
  }
}

const argMax = (arr: ActivationValue): number => {
  if (!(arr instanceof Float32Array)) {
    arr = arr.flatten()[0];
  }
  let idx = 0;
  let max = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > max) {
      max = arr[i];
      idx = i;
    }
  }

  return idx;
};
