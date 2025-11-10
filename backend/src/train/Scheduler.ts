//
//  @TODO: NodeJS only, no running the trainer in the
//         browser so I don't need this level of abstraction
//

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
