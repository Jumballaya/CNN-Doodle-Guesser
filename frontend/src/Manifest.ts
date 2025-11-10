//
//  Types from backend @TODO --> share these types?
//
export interface ClassInfo {
  id: number;
  name: string;
  displayName: string;
}

export interface DatasetManifest {
  version: number;
  classes: ClassInfo[];
}

export class Manifest {
  private manifest: DatasetManifest;
  private classData: Map<number, ClassInfo> = new Map();

  constructor(manifest: DatasetManifest) {
    this.manifest = manifest;

    for (const cls of this.manifest.classes) {
      this.classData.set(cls.id, cls);
    }
  }

  public static async FromFile(path: string) {
    try {
      const res = await fetch(path);
      const data: DatasetManifest = await res.json();
      return new Manifest(data);
    } catch (e) {
      return new Manifest({ version: 1, classes: [] });
    }
  }

  public get version() {
    return this.manifest.version;
  }

  public getDisplayName(id: number) {
    return this.classData.get(id)?.displayName;
  }
}
