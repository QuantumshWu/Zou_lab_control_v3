export type TreeLeaf = { key: string; label: string; text: string };
export type TreeGroup = { heading: string; leaves: TreeLeaf[]; children?: TreeGroup[] };
export type FlatTreeGroup = { heading: string; leaves: TreeLeaf[]; depth: number; parent: number | null; path: string };

/** Preserve hierarchy and branch identity even when headings repeat. Existing
 * heading/leaves producers remain exactly one level deep. */
export function flattenTree(groups: readonly TreeGroup[]): FlatTreeGroup[] {
  const flat: FlatTreeGroup[] = [];
  const visit = (nodes: readonly TreeGroup[], parent: number | null, ancestors: string[]): void => {
    for (const node of nodes) {
      const path = [...ancestors, node.heading], index = flat.length;
      flat.push({ heading: node.heading, leaves: [...node.leaves], depth: ancestors.length, parent, path: JSON.stringify(path) });
      visit(node.children ?? [], index, path);
    }
  };
  visit(groups, null, []);
  return flat;
}
