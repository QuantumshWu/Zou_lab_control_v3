// The Task Console's Diagnostics popup: what the application keeps for looking into a fault afterwards,
// and the archive that hands it over.
//
// Two things are kept, and they are not the same kind. The record -- what happened, in order, and the
// application's state -- is always kept for the last hour. Data -- what the signals carried -- is kept
// within a budget of memory, every signal for the same length of time; the popup says how far back that
// goes now, and the budget is the one thing set here.
import { Field, Popup, command, element, note, setText } from "../fluent/index.js";
/** What the popup says of the kept data, for a budget and how far back it goes. */
export function heldWords(state) {
    if (!state)
        return "";
    if (state.budget_mib === 0)
        return "No data is kept: an archive holds the record alone.";
    if (state.held_minutes === null)
        return "No data has been published yet.";
    const span = state.held_minutes < 1 ? "less than a minute" : state.held_minutes === 1 ? "1 minute" : `${state.held_minutes} minutes`;
    // Full, the budget is what limits it; else it is simply how long data has been coming.
    return state.full ? `Holds the last ${span} of every signal's data: the budget is full, and the oldest goes first.`
        : `Holds the last ${span} of every signal's data; the budget is not full yet.`;
}
export class DiagnosticsPopup {
    send;
    popup;
    budget;
    held = note("", true);
    state;
    constructor(anchor, send) {
        this.send = send;
        const body = element("div", "form");
        this.budget = new Field("diagnostics", this.spec(), {
            commit: value => this.send({ action: "set_diagnostics", budget_mib: Number(value) }, "Setting the data budget…"),
        });
        const actions = element("div", "form-actions");
        actions.append(command("Export record", () => this.export(false), { title: "The last hour's record, the board and the files it names, without data" }).root, command("Export with data", () => this.export(true), { role: "primary", title: "The record and the kept data of every signal, one file per signal" }).root);
        body.append(note("The last hour's record of commands, runs, board commands and state is always kept. Data is kept within this budget, every signal for the same length of time.", true), this.budget.root, this.held, actions);
        this.popup = new Popup(anchor, body, ["--diagnostics-min-w", "--diagnostics-min-h"]);
    }
    spec() {
        return { name: "budget_mib", label: "Data budget", kind: "integer", value: (this.state?.budget_mib ?? 0),
            minimum: 0, required: true, unit: "MiB", description: "The memory the data of recent publications is kept within; 0 keeps none." };
    }
    export(data) {
        this.send({ action: "export_diagnostics", data }, data ? "Exporting diagnostics with data…" : "Exporting diagnostics…");
        this.popup.close();
    }
    toggle() { this.popup.toggle(); }
    accept(state, request, failed = false) {
        if (!state)
            return;
        this.state = state;
        this.budget.update(this.spec(), request, failed);
        setText(this.held, heldWords(state));
    }
}
