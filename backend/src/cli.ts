export type CLIOpts = {
  classList: string;
  testCount: number;
  trainCount: number;
  modelOutput: string;
  epochs: number;
  learnRate: number;
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
  --epochs=<number>         Number of epochs to train
  --learn-rate=<number>     Learning rate, defaults to 0.01

Examples:
  npm run train --workspace=@doodle/backend -- --class-list="src/categories-3.txt" --model-output="doodle-detector.json"
`;

function argvCommands() {
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
  return {
    getInt: (field: string, def: number): number => {
      return out[field] ? parseInt(out[field]) : def;
    },
    getFloat: (field: string, def: number): number => {
      return out[field] ? parseFloat(out[field]) : def;
    },
    getString: (field: string, def: string): string => {
      return out[field] ? out[field] : def;
    },
    getBool: (field: string): boolean => {
      return out[field] === "true";
    },
  };
}

export function getCLIOptions(): CLIOpts {
  const commands = argvCommands();

  const classList = commands.getString("class-list", "src/categories-3.txt");
  const trainCount = commands.getInt("train-count", 800);
  const testCount = commands.getInt("train-count", 200);
  const modelOutput = commands.getString("model-output", "doodle-guesser.json");
  const help = commands.getBool("help");
  const epochs = commands.getInt("epochs", 15);
  const learnRate = commands.getFloat("learn-rate", 0.01);

  return {
    classList,
    trainCount,
    testCount,
    modelOutput,
    help,
    epochs,
    learnRate,
  };
}
