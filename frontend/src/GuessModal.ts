const vowels = ["a", "e", "i", "o", "u"];

export class GuessModal {
  private dismissBtn: HTMLButtonElement;
  private modal: HTMLDivElement;
  private guessText: HTMLSpanElement;
  private guessPretext: HTMLSpanElement;

  constructor() {
    this.modal = document.createElement("div");
    this.modal.classList.add("guess-modal");
    this.dismissBtn = document.createElement("button");
    this.guessPretext = document.createElement("span");
    this.guessPretext.innerText = `My Guess:`;
    this.guessPretext.classList.add("guess-pretex");
    this.guessText = document.createElement("span");
    this.guessText.classList.add("guess");
    this.modal.appendChild(this.guessPretext);
    this.modal.appendChild(this.guessText);
    this.modal.appendChild(this.dismissBtn);

    this.dismissBtn.innerText = "Keep Drawing";
    this.dismissBtn.classList.add("btn");

    document.body.appendChild(this.modal);

    this.dismissBtn.addEventListener("click", (e) => {
      e.preventDefault();
      this.modal.style.visibility = "hidden";
    });
  }

  public showGuess(guess: string) {
    this.guessText.innerText = `${guess}`;
    this.show();
  }

  public show() {
    this.modal.style.visibility = "visible";
  }

  public hide() {
    this.modal.style.visibility = "hidden";
  }
}
