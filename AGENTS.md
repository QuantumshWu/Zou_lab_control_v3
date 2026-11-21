# Rules for changes to this repository

These rules apply to every change, by a person or an agent.

## What is recorded

Persisted data and logs outlive the change that added them. A field nobody reads still costs
every event, every saved Figure and every reader. A 3210-shot scan once carried
`{"trigger_alignment": {"shot": n, "missing": []}}` on every shot. 3209 of those said nothing,
and the Figure Viewer froze on them.

- **Each fact has one home.**
  - Whether a value can be used: the data's validity.
  - What happened: the journal (`journal.note`).
  - How a run was configured: its run record.
  - How one event differs: its event record.

  Do not write a fact into a second home.
- **Record deviations only.** Never record a default or empty value per event. A value that is
  the same for every event of a run belongs in the run record.
- **Every event-record field is registered** in `crates/zlc-core/src/record_fields.json`, with
  its owner, its purpose and its readers: code outside the tests that reads it by name.
  - Register it in the same change that writes it.
  - A debug build refuses an unregistered field or one that says nothing (see
    `docs/PLUGIN_PROTOCOL.md` §5.5).
  - `record_fields::tests` fails when a reader no longer reads its field. Remove the field, or
    its registration, when its last reader goes.
  - Fields marked `unread` were written before the registry and are awaiting removal or a
    reader. Never add a field that way.
- **A journal event or status message** is written when something happens, not on every
  success. Name its reader before adding one.
- **A commit that adds a persisted field, a saved section or a journal event type** says in its
  message what reads it.

## How changes are made

- One topic per commit. Never mix a new record or log with an unrelated change.
- After a change, run only the specific tests that cover it: name the test targets or use name
  filters. Never run a whole crate's or the workspace's tests, `npm test`, or the golden,
  replay, configuration or smoke suites without asking first.
