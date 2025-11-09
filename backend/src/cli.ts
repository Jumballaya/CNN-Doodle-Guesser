export type CLIOpts = {
  classList: string;
  testCount: number;
  trainCount: number;
  modelOutput: string;
  help: boolean;
};

export const USAGE = `Usage: doodle-trainer [optional_arg]
Builds data set and trains doodle detector CNN

Options:
  --help                    Prints out this help message
  --class-list="<file>"     Text file list of doodle types, newline delimited (default: src/categories-3.txt)
  --train-count=<number>    Number of doodles to train on, per doodle, per epoch
  --test-count=<number>     Number of doodles to test against per doodle
  --model-output="<file>"   Path to the final model file output 

Examples:
  npm run train --workspace=@doodle/backend -- --class-list="src/categories-3.txt" --model-output="doodle-detector.json"
`;

function argvCommands(): Record<string, string> {
  const cmds = process.argv.slice(2);
  const out: Record<string, string> = {};
  for (const cmd of cmds) {
    if (cmd.startsWith("--")) {
      const [k, v] = cmd.replace("--", "").split("=");
      out[k] = v;
      if (v === undefined) {
        out[k] = "true";
      }
    }
  }
  return out;
}

export function getCLIOptions(): CLIOpts {
  const defaultClasslist = "src/categories-3.txt";
  const defaultTrainCount = 800;
  const defaultTestCount = 200;
  const defaultModelOutput = "doodle-guesser.json";

  const commands = argvCommands();

  const classList = commands["class-list"] ?? defaultClasslist;
  const trainCount = commands["train-count"]
    ? parseInt(commands["train-count"])
    : defaultTrainCount;
  const testCount = commands["test-count"]
    ? parseInt(commands["test-count"])
    : defaultTestCount;
  const modelOutput = commands["model-output"] ?? defaultModelOutput;
  const help = commands["help"] === "true";

  return {
    classList,
    trainCount,
    testCount,
    modelOutput,
    help,
  };
}
