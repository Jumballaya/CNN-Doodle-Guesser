import { DrawingApp } from "./DrawingApp";
import "./style.css";

export async function doodle() {
  const drawApp = DrawingApp.FromSerialized("doodle-guesser.json");
}
