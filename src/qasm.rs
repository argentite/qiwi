use crate::codegen;

#[derive(Debug)]
pub struct QasmWriter {}

impl codegen::Output for QasmWriter {
    fn emit_gate_apply(&mut self, name: &str, args: &[&str]) {
        let mut args = args.into_iter();

        // every gate has at least one argument
        print!("{} q[{}]", name, args.next().unwrap());

        while let Some(arg) = args.next() {
            print!(", q[{}]", arg);
        }

        println!(";")
    }
}
