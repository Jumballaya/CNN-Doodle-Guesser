import {
  NeuralNetwork,
  Tensor4D,
  type ActivationValue,
  type NNCheckpoint,
} from "@doodle/lib";
import { Manifest } from "./Manifest";
import { GuessModal } from "./GuessModal";

export class DrawingApp {
  public ctx: CanvasRenderingContext2D;
  public width: number;
  public height: number;

  private manifest: Manifest;

  private mouseDown = false;
  private touchDown = false;
  private mouseX = 0;
  private mouseY = 0;
  private radius = 10;

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
  private modal: GuessModal;

  constructor(nn: NeuralNetwork, manifest: Manifest) {
    this.nn = nn;
    this.manifest = manifest;
    this.width = 28 * 10 * 2;
    this.height = 28 * 10 * 2;

    this.guessBtn = this.createGuessBtn();
    [this.canvas, this.resizeCanvas] = this.createCanvas();
    this.container = this.createContainer();
    this.undoBtn = this.createUndoBtn();
    this.redoBtn = this.createRedoBtn();
    this.createCursor();
    this.modal = new GuessModal();
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

    document.body.addEventListener("touchstart", this.onTouchStart.bind(this));
    document.body.addEventListener("touchend", () => {
      this.mouseDown = false;
    });
    document.body.addEventListener("mousedown", this.onMouseDown.bind(this));
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
    this.canvas.addEventListener("touchmove", this.onTouchMove.bind(this));
  }

  public static async FromSerialized({
    modelFile,
    manifestFile,
  }: {
    modelFile: string;
    manifestFile: string;
  }) {
    try {
      const data: NNCheckpoint = await (await fetch(modelFile)).json();
      const nn = NeuralNetwork.fromCheckpoint(data);
      const man = await Manifest.FromFile(manifestFile);
      return new DrawingApp(nn, man);
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
    return data;
  }

  private onMouseDown(e: MouseEvent) {
    if (this.touchDown) {
      return;
    }
    if (e.target !== this.canvas) {
      return;
    }
    this.mouseDown = true;
    this.snapshots.push(this.ctx.getImageData(0, 0, this.width, this.height));
    this.undoBtn.disabled = false;
    this.redoBtn.disabled = true;
    this.redosnaps = [];
    this.makeMark();
  }

  private onMoveMouse(e: MouseEvent) {
    if (this.touchDown) {
      return;
    }
    const rect = this.canvas.getBoundingClientRect();
    this.mouseX = e.clientX - rect.x;
    this.mouseY = e.clientY - rect.y;
    if (this.mouseDown) {
      this.makeMark();
    }
  }

  private onTouchStart(e: TouchEvent) {
    if (e.target !== this.canvas) {
      return;
    }
    this.touchDown = true;
    this.snapshots.push(this.ctx.getImageData(0, 0, this.width, this.height));
    this.undoBtn.disabled = false;
    this.redoBtn.disabled = true;
    this.redosnaps = [];
    this.makeMark();
  }

  private onTouchMove(e: TouchEvent) {
    const rect = this.canvas.getBoundingClientRect();
    this.mouseX = e.changedTouches[0].clientX - rect.x;
    this.mouseY = e.changedTouches[0].clientY - rect.y;
    if (this.touchDown) {
      this.makeMark();
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

  private makeMark() {
    this.ctx.fillStyle = "black";
    this.ctx.beginPath();
    this.ctx.arc(this.mouseX, this.mouseY, this.radius, 0, Math.PI * 2);
    this.ctx.fill();
  }

  private checkGuess(guess: string | undefined) {
    if (!guess) {
      return;
    }
    this.modal.showGuess(guess);
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
      const output = argMax(guess);

      this.checkGuess(this.manifest.getDisplayName(output));
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

  private createCursor(): HTMLDivElement {
    const cursor = document.createElement("div");
    cursor.classList.add("cursor");
    cursor.style.width = `${this.radius * 2}px`;
    cursor.style.height = `${this.radius * 2}px`;

    document.body.appendChild(cursor);

    let posX = 0;
    let posY = 0;
    document.body.addEventListener("mousemove", (e) => {
      posX = e.clientX;
      posY = e.clientY;
      cursor.style.left = `${e.clientX - this.radius}px`;
      cursor.style.top = `${e.clientY - this.radius}px`;
    });

    this.canvas.addEventListener("mouseenter", () => {
      cursor.style.opacity = `0.5`;
    });

    this.canvas.addEventListener("mouseleave", () => {
      cursor.style.opacity = `0`;
    });

    document.body.addEventListener("wheel", (e) => {
      const dir = e.deltaY >= 0 ? 1 : -1;
      this.radius += dir;
      this.radius = Math.min(60, Math.max(10, this.radius));
      cursor.style.width = `${this.radius * 2}px`;
      cursor.style.height = `${this.radius * 2}px`;
      cursor.style.left = `${posX - this.radius}px`;
      cursor.style.top = `${posY - this.radius}px`;
    });

    return cursor;
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
