import { DeriveVariablesEditor } from "./derive_variables.js";
import { FrameClaimsEditor } from "./frame_claims.js";
import { PairMatrixEditor } from "./pair_matrix.js";
import { PostSelectionEditor } from "./post_selection.js";
import { ScanPlanEditor } from "./scan_plan.js";
const EDITORS = {
    scan_plan: host => new ScanPlanEditor(host.edit),
    pair_matrix: host => new PairMatrixEditor(host.edit),
    derive_variables: host => new DeriveVariablesEditor(host.edit),
    frame_claims: host => new FrameClaimsEditor(host),
    post_selection: host => new PostSelectionEditor(host.edit),
};
/** The editor for a model kind, or null for a kind this UI does not draw
 * (the Edit page then shows the managed fields as plain rows). */
export function nodeEditor(kind, host) {
    return EDITORS[kind]?.(host) ?? null;
}
