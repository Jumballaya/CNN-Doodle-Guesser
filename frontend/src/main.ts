import { DrawingApp } from "./DrawingApp";
import "./style.css";

async function main() {
  await DrawingApp.FromSerialized("model_epoch-25.json");
}
main();
