export class DrawingApp {
  public canvas: HTMLCanvasElement;
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

  constructor() {
    this.canvas = document.createElement("canvas");
    this.resizeCanvas = document.createElement("canvas");
    this.width = 28 * 10 * 2;
    this.height = 28 * 10 * 2;
    this.resizeCanvas.width = 28;
    this.resizeCanvas.height = 28;
    this.canvas.width = this.width;
    this.canvas.height = this.height;
    document.body.appendChild(this.canvas);
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
    const imageData = this.resizeCtx.getImageData(0, 0, 28, 28);
    for (let i = 0; i < 28 * 28; i++) {
      data[i] = imageData.data[i * 4 + 3] / 255;
    }
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
    }
  }

  private redo() {
    const screen = this.ctx.getImageData(0, 0, this.width, this.height);
    const last = this.redosnaps.pop();
    if (last) {
      this.ctx.putImageData(last, 0, 0);
      this.snapshots.push(screen);
    }
  }
}
