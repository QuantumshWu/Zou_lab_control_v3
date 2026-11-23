// The window's own entry (dist/main.js) on a fake native host -- window.ipc out, chrome.webview's messages in --
// so every state goes the way the desktop's does: transport.attach's handler runs board.accept (whose changed()
// asks the status line) and then shell.accept. The host takes the window's commands in the order sent, as the
// workbench does: an edit moves the draft revision on and says it; a Start of another revision is refused. `viewer`: the Figure Viewer's window, else the Task Console's.
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

export async function startAfterEdit(viewer) {
  const wire = [], listeners = [], handled = [];
  window.chrome = { webview: { addEventListener(type, listener) { if (type === 'message') listeners.push(listener); } } };
  window.ipc = { postMessage(text) { wire.push(JSON.parse(text)); } };
  const deliver = message => { for (const listener of listeners) listener({ data: JSON.stringify(message) }); };
  const node = viewer ? 'figure-1-derive' : 'scan';
  const rows = [{ node_id: node, name: node, type_id: 'calibration', node_kind: viewer ? 'processor' : 'task', draft_revision: 1, values: { label: 'a' },
    controls: [{ name: 'label', label: 'Label', kind: 'text', value: 'a' }], source_signal: null, source_label: null, source_choices: [], device_keys: {},
    device_choices: {}, device_labels: {}, artifact_inputs: {}, artifact_specs: [], auto_preview: false, outputs: [], issues: [], can_start: true,
    can_stop: true, preview_offered: false, edited: false, state: 'running', status: 'running', phase: 'running', run: null, editor_facts: null }];
  const sections = () => viewer
    ? { viewer: { document: { path: 'C:\d\a.npz' }, panels: [], signals: [], plot_kinds: [], processor_types: [], logic: structuredClone(rows) } }
    : { logic: structuredClone(rows), board: { panels: [], paused: false, selectors: true, active_task: null, frozen: {} }, panels: [] };
  let taken = 0;
  /** The host takes the next node command sent and answers it on the state that carries its change. */
  const step = () => {
    while (taken < wire.length && !['edit_node', 'start_node', 'stop_node'].includes(wire[taken].action)) taken++;
    const command = wire[taken++];
    if (!command) return null;
    const row = rows[0];
    let result = null, error = null;
    if (command.action === 'edit_node') {
      if (command.values) { row.values = { ...row.values, ...command.values }; row.controls = row.controls.map(control => ({ ...control, value: row.values[control.name] })); row.draft_revision++; }
      result = { node, draft_revision: row.draft_revision };
    }
    if (command.action === 'start_node') {
      if (command.expected_draft_revision !== undefined && command.expected_draft_revision !== row.draft_revision) {
        error = `${node} changed (draft revision ${row.draft_revision} is not ${command.expected_draft_revision}); read it again`;
      } else result = { node };
    }
    handled.push({ action: command.action, expected: command.expected_draft_revision, ...(error ? { error } : {}) });
    deliver({ type: 'state', request: command.id, result, error, ...sections() });
    return { command, error };
  };
  if (viewer) history.replaceState(null, '', `${location.pathname}?window=figure-viewer`);
  await import('../dist/main.js');
  deliver({ type: 'state', ...sections() });
  await sleep(50);
  if (!viewer) [...document.querySelectorAll('.logic-row button')].find(button => button.textContent.trim() === 'Edit').click();
  await sleep(50);
  const page = document.querySelector('.logic-editor');
  const input = [...page.querySelectorAll('.field')].find(field => field.querySelector('.field-label')?.textContent.replace(/ \*$/, '') === 'Label').querySelector('input');
  const said = button => (button.querySelector('.words .current') ?? button).textContent.trim();
  const start = [...page.querySelectorAll('.logic-editor-actions button')].find(button => /^(Start|Restart|Starting…)\*?$/.test(said(button)));
  return { wire, handled, step, page, input, start, said, starts: () => wire.filter(command => command.action === 'start_node') };
}
