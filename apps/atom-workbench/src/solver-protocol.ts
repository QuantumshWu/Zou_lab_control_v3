export const NATIVE_PROTOCOL = 'atom-lindblad-jsonl/3' as const;
export const HTTP_PROTOCOL = 'atom-workbench-http/2' as const;

/** The one model schema the native worker serves. v2 carries laser-derived
 * integer phase coordinates: an older reader must not silently turn their
 * roundoff-level frequency residuals into new beats. */
export const FULL_MODEL_SCHEMA = 'rb87-full/2' as const;
