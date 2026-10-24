/** Atomic models must negotiate this capability before any solve. A generic
 * C-only Lindblad solver is not a compatible substitute for the emitted-light
 * frequency bath, even if it accepts unknown JSON properties. */
export const ATOMIC_MODEL_SCHEMA = 'rb87-dipole/2' as const;
export const NATIVE_PROTOCOL = 'atom-lindblad-jsonl/2' as const;
export const HTTP_PROTOCOL = 'atom-workbench-http/2' as const;


/** v2 carries laser-derived integer phase coordinates. An older reader must
 * not silently turn their roundoff-level frequency residuals into new beats. */
export const FULL_MODEL_SCHEMA = 'rb87-full/2' as const;
/** Compatibility export for existing UI transport imports. */
export const EFFECTIVE_MODEL_SCHEMA = FULL_MODEL_SCHEMA;
