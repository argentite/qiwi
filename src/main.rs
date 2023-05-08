#[macro_use]
extern crate lazy_static;

mod ast;
mod codegen;
mod parser;
mod qasm;

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

    let mut context = codegen::Context::new();
    codegen::load_functions(&mut context, &ast);
    let mut output = qasm::QasmWriter {};
    match codegen::compile_function("main", &mut context, &mut output) {
        Ok(_result) => {}
        Err(error) => {
            eprintln!("Error: {}", error);
        }
    }
}
