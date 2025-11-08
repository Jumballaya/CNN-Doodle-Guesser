// backend/src/train/scheduler.ts
export interface IScheduler {
  tick(): Promise<void>;
}

export class MicrotaskScheduler implements IScheduler {
  tick(): Promise<void> {
    return Promise.resolve();
  }
}

/**
 * Node-friendly: yields back to the event loop using setImmediate.
 */
export class NodeScheduler implements IScheduler {
  tick(): Promise<void> {
    return new Promise((resolve) => setImmediate(resolve));
  }
}
