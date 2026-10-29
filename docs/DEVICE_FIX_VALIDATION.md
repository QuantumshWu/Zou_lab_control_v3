# Remote ownership, Pylon teardown and image navigation

Developed on `codex/device-lifecycle-image-zoom`, based on `6676a529`.
Integrated into `main` on 2026-10-28 after the user authorized the merge.

## Decisions and defects

- **Remote RF:** retain the generic fabric. RF already implements the common
  plugin contract; another RF-specific server would duplicate ownership and
  lifecycle code. SLM and sequencer retain their specialized protocols for
  phase revisions/presentation receipts and board ownership/SAFE/cancellation.
  See [the protocol rationale](PLUGIN_PROTOCOL.md).
  Publication, disconnect, detach, reconnect and withdrawal now wait for
  admitted operations to actually finish. A timeout alone is not completion.
  Releases confirm SAFE before transferring ownership; failed cleanup retains
  the reservation. Duplicate publication and late requests from dead sockets
  cannot replace an owner. An SLM withdrawal failure keeps its announcement.

- **Pylon:** cleanup previously relied on wrapper destruction after `Close`,
  and process exit could call device cleanup concurrently with unfinished
  instance-lane operations, including Open. The receiver, concurrent SDK calls
  and node runs now finish before lane-owned cleanup. Open failures and normal
  Close explicitly release the SDK transport object with `DestroyDevice`;
  failed destruction retains ownership for retry. Channel errors also run
  cleanup and wake reverse-RPC waiters.

- **Image navigation:** outward rounding to whole source pixels made small
  wheel movements invisible and imposed a one/two-cell magnification limit.
  Navigation and pan now retain continuous limits; sampling and ROI selection
  remain separate. Zoom retains the centered 1.1 exponential rule. Frontend
  and native queues combine adjacent same-direction wheel travel while
  preserving gesture order and independently answered remote requests.
  Numerical limits are computed at the zoom center, so a batched burst and
  repeated small steps retain the same reach, including a center at zero.
  Limits remain reversible with fractional wheel steps. Explicit
  rectangular viewports retain their exact ranges; the renderer preserves
  pixel aspect by adjusting the drawing box instead of widening the request.
  A second defect appeared in the real document waiting-frame regression:
  the presentation fast path compared accepted configuration but not the
  current previewed interaction state. Replaying waiting gestures could apply
  zoom twice. The fast path now also checks viewport, selectors, facet target
  and series presentation before deciding that restoration is unnecessary.

## Verification boundaries

- Browser DOM suite: **106 passed**, including 8 wheel ordering/batching cases.
- Python lifecycle suite: **15 passed**, also invoked by the Rust host suite.
  New tests reproduce premature shutdown during Open and missing explicit SDK
  destruction on the old implementation.
- Rust navigation: **14 passed**, including rendered RGBA, waiting-frame
  replay, presentation/reconstruction, pan, logarithmic and extreme numeric
  ranges, fractional reversal, and equivalent batched wheel travel.
- Rust interaction reference: **9 passed**. Native desktop: **24 passed**.
  Fabric unit/TCP integration: **8 passed**. Plugin host unit/integration and
  the virtual camera measurement reference also pass.
- Final Rust regression iterations used the portable MSVC toolchain with
  release-profile environment overrides `opt-level=1`, `codegen-units=16`,
  `lto=false` to shorten linking. Committed Cargo profiles are unchanged.
  Optional external-corpus/readout tests without their required environment
  are not evidence of hardware or external-dataset validation.
- No physical RF, SLM, sequencer or Pylon hardware was controlled. This machine
  has no Pylon SDK/camera. These fixes address demonstrated lifecycle defects;
  the other computer's precise enumeration failure still needs a hardware
  close/reopen/scan check. Basler documents that `Close` releases the connection
  and `DestroyDevice` destroys the SDK object; missing destruction alone does
  not prove that every successful Close prevents reconnection.

### Legacy replay corpus

The local, git-ignored v2 session captures are preserved, and no replay tests
are disabled. `committed_scenarios_replay` and
`committed_scenarios_native_renderer` do not match those old captures:

- Empty-panel target/size differences come from the earlier `aee32b63` change,
  which reserves authored geometry before data arrives. Its explicit
  `a_loaded_empty_panel_reserves_its_authored_size_before_start` regression
  verifies that behavior.
- `waiting_turns` records the former pixel-rounded viewport and renderer call
  sequence. Continuous zoom/pan intentionally produces different limits,
  pictures and subsequent recorded observations. The new navigation tests use
  real sessions and a real FigureDocument, including waiting-frame replay,
  instead of substituting actual output for those recorded expectations.

The interaction-number oracle retains exact comparisons for ordinary results.
Old zero-width results and outward recovery from zero-width input have explicit
precision contracts. Synthetic sample axes with repeated coordinates are
checked against their inverse interval; the real image entry point requires
strictly monotonic coordinates.

Basler references: [InstantCamera lifecycle](https://docs.baslerweb.com/pylonapi/cpp/class_pylon_1_1_c_instant_camera)
and [SDK samples](https://docs.baslerweb.com/pylonapi/pylon-sdk-samples-manual).
