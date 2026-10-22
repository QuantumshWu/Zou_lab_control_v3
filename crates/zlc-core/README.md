# Native scientific data and publication core

This crate owns immutable storage, typed scientific layout and finite/streaming
publication lifetime. It contains no plotting, Python, GUI, device calls or
application-level job executor.

## Storage and schema

`OwnedBuffer<T>` freezes a typed Vec without copying; its readers share an Arc.
The optional allocation pool recycles only after the last reader releases it.
`AnyBuffer` is the single dynamic owner for bool, signed/unsigned 8–64-bit
integers, f16/f32/f64 and complex64/128. Half and complex storage have explicit
padding-free layouts. A readonly byte view never includes unused capacity.

`DatasetSchema` has three scientific domains, Repeat, Point and Cell. All use
the same `Domain` and `CodeMapping`: a logical axis reads its coordinate code
from `base[(physical_flat_index / inner) % base_length]`. Dense tensor axes are
compact ranges using that same mapping, not another domain implementation.
Several axes may describe a carrier; alternative coordinates share the primary
mapping. Construction validates metadata once. Normal publications reuse the
schema Arc; independently constructed equal schemas remain valid.

Numeric coordinates retain their primitive dtype. Implicit integer coordinates
and origins plus offsets remain factored. Mixed scalar integer coordinates use
i128; larger arbitrary-precision Python integers currently fail explicitly.
`DataBlock` validates values, compact validity and optional sample sigma against
the schema. It does not attach a publication identity to storage or manufacture
a dense pixel mask for a uniformly valid image.

## Runtime contract

`Runtime::begin` registers an owner, schemas and one run record. A private
generation token authorizes commits and lifecycle changes. `commit` prechecks
every updated sibling before installing any of them. Updated siblings share
one event record and commit revision. A bundle may update a declared subset;
unchanged outputs do not acquire a fictitious new revision.

Every input supplies its actual ordinal. Gaps remain gaps. Recomputing the
latest ordinal creates a new revision of that sample, not an extra shot;
regressing an output's ordinal is rejected. Acquisition code owns the meaning
of the ordinal. Runtime does not infer it from Pulse or camera timing.

| Consumer need | Runtime behavior |
|---|---|
| Latest display | One replaceable pending delivery |
| Exact processing | Ordered queue with explicit event and referenced-byte limits; overflow terminates that subscription with an error, without stalling acquisition or pretending delivery stayed exact |
| Initial state | `Start::Current` returns a snapshot with `revision=None`; each member retains its true key, so mixed revisions cannot masquerade as an atomic commit |
| Latest-event output without a history lease | Only the latest value is retained by Runtime |
| History lease | Retention spans the maximum active requested window; a last-lease drop releases historical references immediately |
| Frozen analysis | Its acquired Arc reads remain valid independently of Runtime retention |

`Storage::Finite` declares the complete authored Repeat/Point layout. Each input
supplies a flat placement origin; the block's Repeat/Point sizes are its extent.
The event read exposes just that contribution, while `Publication::canonical()`
holds the immutable full-layout snapshot for display and terminal processing.
Unwritten positions remain holes. Overlapping placements are rejected, including
across a multi-signal commit: every sibling is checked before any store changes.
Finite accumulation does not require or consume an indexed-history lease.

`DatasetStore` uses shared pages of 64 placement references. Taking a snapshot is
constant-time; append copies at most the bounded tail page, not all earlier
segments or their numerical planes. Coverage counts acquired Repeat × Point
positions, independently of their scientific validity. Iterating a canonical
snapshot visits its actual segments; contiguous materialization is a consumer
operation rather than a mandatory acquisition stage.

History reads own a fixed-width effective schema with ordinary relative
coordinates `-H+1 .. 0`; new/absent positions are holes. The history-axis identity
is stable for the signal across restart. Source axes, code arrays, Cell/Repeat
domains and payloads are shared. Schema/view caches are bounded by active demand.
Changing the window changes view metadata, not the scientific revision.
Declared base and window schemas are available before the first sample.
An explicit one-record view can borrow latest without a history lease; it does
not retain any past events. This permits a one-point record plot while ordinary
Window=1 consumers continue to receive the original event schema.

`read_latest`, `read_at` and `read_window` return owned references while holding
one short state lock. Later processing never rechecks whether Runtime still
retains those pixels. Exact and latest queues also hold the same publications,
not copies of their payloads. Only subscribers of the affected generation are
visited, and only matching subscriptions are awakened. Metadata construction,
copying scientific data, serialization and user callbacks do not run inside a
publication callback. Large retired payloads are released outside the lock.
`read_views` captures a set of event/canonical/window reads under the same lock,
so same-source panels cannot accidentally select different latest publications.
`signals` projects declarations and retention counts; it is not another registry.

`watch_changes` is a coalesced metadata wake, not a scientific subscription. Its
one-slot notification contains no payload and cannot stall publishers. A consumer
clears the shared pending flag before acquiring its current reads. Exact processors
still receive ordered deliveries; `Subscription::wake` interrupts a wait for
control work without consuming or reordering one of those deliveries.

`Storage::Snapshot` declares a replacement result whose input records are already
inside its dataset (for example, a parameter table fitted over a retained window).
It remains a latest immutable value but does not admit another indexed history.
`history_capable` exposes this producer contract; panels need not infer storage
semantics from plot kinds or special axis names.

Stop is `seal`, not `retire`. A sealed run remains readable. Replacing a sealed
owner registers a new generation and releases Runtime's references to the old
one; obsolete commits and stops cannot affect it. `retire` removes an owner;
`close` ends Runtime and wakes waiting consumers. Neither revokes acquired data.

A Frozen derivation uses an independent generation, then seals its result. Its
parents are the exact owned publications supplied to the calculation. Output
lineage retains causal run/event metadata, but not ancestor pixels. It does not
claim membership in the live source's current-shot cohort.

## Verification and unfinished integration

Core tests exercise typed ownership, recycling, domain mapping, compact validity,
history gaps and replacement, demand release, Frozen lineage, restart, sibling
atomicity, overflow and concurrent delivery ordering. The production Runtime
benchmark lives in `research/runtime-native` and measures actual public APIs.

Device claims, processor scheduling and the complete live application chain are
subsequent integration work. The current Runtime stores immutable event blocks,
finite canonical snapshots and demanded windows; this crate is not yet a complete
replacement for every v2 Runtime role.
