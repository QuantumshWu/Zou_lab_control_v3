/** Schema of the radiative-bath model built by master-model.ts. The native
 * solver no longer serves it (its capabilities are rb87-full/2 and the
 * generic-lindblad/1 warm-up); solves always use FULL_MODEL_SCHEMA below. */
export const ATOMIC_MODEL_SCHEMA = 'rb87-dipole/2' as const;
export const NATIVE_PROTOCOL = 'atom-lindblad-jsonl/2' as const;
export const HTTP_PROTOCOL = 'atom-workbench-http/2' as const;


/** v2 carries laser-derived integer phase coordinates. An older reader must
 * not silently turn their roundoff-level frequency residuals into new beats. */
export const FULL_MODEL_SCHEMA = 'rb87-full/2' as const;
/** Compatibility export for existing UI transport imports. */
export const EFFECTIVE_MODEL_SCHEMA = FULL_MODEL_SCHEMA;
