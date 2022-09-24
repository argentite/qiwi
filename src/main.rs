mod ast;
mod parser;

#[derive(clap::Parser, Debug)]
struct CommandLineArgs {
    /// The Qiwi source file to start compilation
    #[clap(value_name = "filename")]
    file_name: String,
    /// Print the Abstract Syntax Tree
    #[clap(long)]
    dump_ast: bool,
}

fn main() {
    let args = {
        use clap::Parser;
        CommandLineArgs::parse()
    };

    let base_file = std::fs::read_to_string(&args.file_name)
        .unwrap_or_else(|x| panic!("Error reading source file: {}: {}", args.file_name, x));
    let ast = match parser::parse_source(base_file.as_str()) {
        Ok(x) => x,
        Err(e) => {
            println!("Failed to produce AST: {:?}", e);
            return;
        }
    };

    if args.dump_ast {
        println!("{:?}", ast);
    }
}
