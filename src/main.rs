use std::env;
use corrosiff;

fn main() {
    // Open a siff file from the argument
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <path_to_siff_file>", args[0]);
        return;
    }

    corrosiff::open_siff(
        &args[1]
    ).map(|siff| {
        println!("Filename: {}", siff.filename());
    }).unwrap_or_else(|e| {
        eprintln!("Error: {}", e);
    });
}