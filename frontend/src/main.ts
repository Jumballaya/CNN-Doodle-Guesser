import { DrawingApp } from "./DrawingApp";
import "./style.css";

async function main() {
  await DrawingApp.FromSerialized({
    modelFile: "doodle-all.json",
    manifestFile: "doodle-all-manifest.json",
  });
}
main();
