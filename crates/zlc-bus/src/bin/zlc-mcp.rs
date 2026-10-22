//! `zlc-mcp`: the MCP entry point an AI client registers (stdio). It lists the
//! experiments running on this computer and attaches to one; it never starts
//! an instrument. `zlc-mcp --client-config claude|codex` prints the registration.
fn main() {
    let mut args = std::env::args().skip(1);
    match (args.next().as_deref(), args.next()) {
        (None, _) => if let Err(error) = zlc_bus::proxy::serve_stdio() { eprintln!("zlc-mcp: {error}"); std::process::exit(1); },
        (Some("--client-config"), Some(client)) => match zlc_bus::proxy::client_config(&client) {
            Ok(text) => println!("{text}"),
            Err(error) => { eprintln!("zlc-mcp: {error}"); std::process::exit(2); }
        },
        _ => { eprintln!("Usage: zlc-mcp [--client-config claude|codex]"); std::process::exit(2); }
    }
}
