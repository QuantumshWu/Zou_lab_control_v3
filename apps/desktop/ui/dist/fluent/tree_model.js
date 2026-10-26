/** Preserve hierarchy and branch identity even when headings repeat. Existing
 * heading/leaves producers remain exactly one level deep. */
export function flattenTree(groups) {
    const flat = [];
    const visit = (nodes, parent, ancestors) => {
        for (const node of nodes) {
            const path = [...ancestors, node.heading], index = flat.length;
            flat.push({ heading: node.heading, leaves: [...node.leaves], depth: ancestors.length, parent, path: JSON.stringify(path) });
            visit(node.children ?? [], index, path);
        }
    };
    visit(groups, null, []);
    return flat;
}
