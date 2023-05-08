use std::{collections::HashMap, rc::Rc};

use crate::ast;

pub trait Output {
    fn emit_gate_apply(&mut self, name: &str, args: &[&str]);
}

pub struct Context<'a> {
    parent: Option<&'a Context<'a>>,
    functions: HashMap<String, Rc<ast::FunctionDef<'a>>>,
    variables: HashMap<String, Vec<usize>>,
    comptime_variables: HashMap<String, usize>,
    allocated_qubits: usize,
}

impl Context<'_> {
    pub fn new() -> Self {
        Self {
            parent: None,
            functions: HashMap::new(),
            variables: HashMap::new(),
            comptime_variables: HashMap::new(),
            allocated_qubits: 0,
        }
    }

    pub fn alloc(&mut self, n: usize) -> Result<Vec<usize>, String> {
        self.allocated_qubits += n;
        Ok(((self.allocated_qubits - n)..self.allocated_qubits).collect())
    }

    pub fn scope(&self) -> Context {
        let mut context = Self::new();
        context.parent = Some(self);
        context
    }

    pub fn get_variable(&self, name: &str) -> Option<&Vec<usize>> {
        self.variables.get(name)
    }

    pub fn get_variable_mut(&mut self, name: &str) -> Option<&mut Vec<usize>> {
        self.variables.get_mut(name)
    }

    pub fn set_variable(&mut self, name: &str, value: Vec<usize>) -> Option<Vec<usize>> {
        self.variables.insert(name.to_string(), value)
    }

    pub fn get_comptime_variable(&self, name: &str) -> Option<usize> {
        self.comptime_variables.get(name).copied()
    }

    pub fn set_comptime_variable(&mut self, name: &str, value: usize) -> Option<usize> {
        self.comptime_variables.insert(name.to_string(), value)
    }

    pub fn get_function(&self, name: &str) -> Option<Rc<ast::FunctionDef>> {
        if let Some(parent) = self.parent {
            parent.get_function(name)
        } else {
            self.functions.get(name).cloned()
        }
    }
}

trait CompileExpr {
    fn compile(&self, context: &mut Context, output: &mut dyn Output)
        -> Result<Vec<usize>, String>;
}

trait CompileStatement {
    fn compile(&self, context: &mut Context, output: &mut dyn Output) -> Result<(), String>;
}

#[derive(Debug)]
struct BuiltinGate {
    symbol: &'static str,
    inputs: usize,
}

lazy_static! {
    static ref BUILTIN_GATES: HashMap<&'static str, BuiltinGate> = {
        HashMap::from([
            (
                "_x",
                BuiltinGate {
                    symbol: "x",
                    inputs: 1,
                },
            ),
            (
                "_h",
                BuiltinGate {
                    symbol: "h",
                    inputs: 1,
                },
            ),
            (
                "_cx",
                BuiltinGate {
                    symbol: "cx",
                    inputs: 2,
                },
            ),
        ])
    };
}

impl CompileExpr for ast::FuncExpr<'_> {
    fn compile(
        &self,
        context: &mut Context,
        output: &mut dyn Output,
    ) -> Result<Vec<usize>, String> {
        let args: Vec<Vec<usize>> = self
            .args
            .iter()
            .map(|arg| arg.compile(context, output))
            .collect::<Result<_, _>>()?;
        let func = match context.get_function(self.ident) {
            Some(func) => func,
            None => match BUILTIN_GATES.get(self.ident) {
                Some(gate) => {
                    let args: Vec<usize> = args.into_iter().flatten().collect();

                    if args.len() != gate.inputs {
                        return Err(format!("Argument length mismatch in builtin gate {}, expected {} qubits, found {}",
                            gate.symbol, gate.inputs, args.len()));
                    }

                    let args_str: Vec<String> = args.iter().map(|s| s.to_string()).collect();
                    let args_ref: Vec<&str> = args_str.iter().map(|s| s.as_str()).collect();
                    output.emit_gate_apply(gate.symbol, args_ref.as_slice());

                    // output for builtin gates is the same as input
                    return Ok(args);
                }
                None => return Err(format!("Undefined function: {}", self.ident)),
            },
        }
        .clone();

        // Create new scope for local variables
        let mut context = context.scope();
        // TODO: Check length
        if self.args.len() != func.param.len() {}
        for (arg, param) in args.iter().zip(func.param.iter()) {
            context.set_variable(param.name, arg.to_vec());
        }

        func.body.compile(&mut context, output)
    }
}

impl CompileExpr for ast::IntExpr {
    fn compile(
        &self,
        context: &mut Context,
        output: &mut dyn Output,
    ) -> Result<Vec<usize>, String> {
        let mut len = self.value.bits() as usize;
        if len == 0 {
            len = 1;
        }
        let qubits = context.alloc(len)?;
        for i in 0..len {
            if self.value.bit(i as u64) {
                output.emit_gate_apply("x", &[qubits[i].to_string().as_str()]);
            }
        }

        Ok(qubits)
    }
}

impl CompileExpr for ast::VarExpr<'_> {
    fn compile(
        &self,
        context: &mut Context,
        _output: &mut dyn Output,
    ) -> Result<Vec<usize>, String> {
        match context.get_variable(self.ident) {
            Some(qubits) => {
                if let Some(index) = &self.index {
                    let index: usize = match index.as_ref() {
                        ast::Expr::Int(x) => x.value.clone().try_into().expect("Too large index"),
                        ast::Expr::Var(v) => match context.get_comptime_variable(v.ident) {
                            Some(x) => x,
                            None => {
                                return Err(format!("Unknown compile time variable: {}", v.ident))
                            }
                        },
                        ast::Expr::Binary(_) | ast::Expr::Func(_) => todo!(),
                    };
                    Ok(vec![qubits[index]])
                } else {
                    Ok(qubits.to_vec())
                }
            }
            None => Err(format!("Undefined variable: {}", self.ident)),
        }
    }
}

impl CompileExpr for ast::Expr<'_> {
    fn compile(
        &self,
        context: &mut Context,
        output: &mut dyn Output,
    ) -> Result<Vec<usize>, String> {
        match self {
            ast::Expr::Int(i) => i.compile(context, output),
            ast::Expr::Var(v) => v.compile(context, output),
            ast::Expr::Binary(_) => todo!(),
            ast::Expr::Func(func) => func.compile(context, output),
        }
    }
}

impl CompileStatement for ast::Block<'_> {
    fn compile(&self, context: &mut Context, output: &mut dyn Output) -> Result<(), String> {
        for stmt in &self.stmts {
            stmt.compile(context, output);
        }
        Ok(())
    }
}

impl CompileStatement for ast::AssignmentStmt<'_> {
    fn compile(&self, context: &mut Context, output: &mut dyn Output) -> Result<(), String> {
        let mut rhs = self.rhs.compile(context, output)?;

        match self.lhs.as_ref() {
            ast::LhsExpr::Single(lhs) => {
                let lhs_size = match &self.lhs_type {
                    Some(lhs_type) => match lhs_type.size() {
                        Some(x) => x,
                        None => todo!(),
                    },
                    None => match context.get_variable(lhs.ident) {
                        Some(var) => var.len(),
                        None => {
                            return Err(format!("Assignment to undeclared variable: {}", lhs.ident))
                        }
                    },
                };

                let extra = (lhs_size as isize) - (rhs.len() as isize);
                if extra < 0 {
                    return Err(format!(
                        "RHS of assignment ({}) larger than LHS({})",
                        rhs.len(),
                        lhs_size,
                    ));
                } else if extra > 0 {
                    rhs.append(&mut context.alloc(extra as usize)?);
                }
                if let Some(index) = &lhs.index {
                    let index: usize = match index.as_ref() {
                        ast::Expr::Int(x) => x.value.clone().try_into().expect("Too large index"),
                        ast::Expr::Var(v) => match context.get_comptime_variable(v.ident) {
                            Some(x) => x,
                            None => {
                                return Err(format!("Unknown compile time variable: {}", v.ident))
                            }
                        },
                        ast::Expr::Binary(_) | ast::Expr::Func(_) => todo!(),
                    };
                    let variable = context.variables.get_mut(&lhs.ident.to_string()).unwrap();
                    if rhs.len() != 1 {
                        return Err(format!(
                            "{} qubits cannot be assigned to one qubit at {}[{}]",
                            rhs.len(),
                            lhs.ident,
                            index,
                        ));
                    }
                    variable[index] = rhs[0];
                } else {
                    // TODO: warn overwrite
                    context.variables.insert(lhs.ident.to_string(), rhs);
                }
            }
            ast::LhsExpr::Tuple(tuple) => {
                // TODO: Tuples currently do not support initialization, we can only assign to them
                // FIXME: We also don't resize or look inside them, we treat them just as a sequence of qubits
                let qubits: Vec<usize> = tuple
                    .elements
                    .iter()
                    .map(|var| -> Result<Vec<usize>, String> { var.compile(context, output) })
                    .collect::<Result<Vec<Vec<usize>>, _>>()?
                    .into_iter()
                    .flatten()
                    .collect();

                if qubits.len() != rhs.len() {
                    return Err(format!(
                        "Tuple length ({}) is not equal to RHS length ({}) in assignment",
                        qubits.len(),
                        rhs.len()
                    ));
                }

                let mut position = 0;
                for element in &tuple.elements {
                    if let Some(index) = &element.index {
                        let index: usize = match index.as_ref() {
                            ast::Expr::Int(x) => {
                                x.value.clone().try_into().expect("Too large index")
                            }
                            ast::Expr::Var(v) => match context.get_comptime_variable(v.ident) {
                                Some(x) => x,
                                None => {
                                    return Err(format!(
                                        "Unknown compile time variable: {}",
                                        v.ident
                                    ))
                                }
                            },
                            ast::Expr::Binary(_) | ast::Expr::Func(_) => todo!(),
                        };
                        let variable = match context.get_variable_mut(element.ident) {
                            Some(v) => v,
                            None => return Err(format!("Undefined variable: {}", element.ident)),
                        };
                        (*variable)[index] = rhs[position];
                        position += 1;
                    } else {
                        let variable = match context.get_variable_mut(element.ident) {
                            Some(v) => v,
                            None => return Err(format!("Undefined variable: {}", element.ident)),
                        };
                        *variable = rhs[position..(position + variable.len())].to_vec();
                    }
                }
            }
        }

        Ok(())
    }
}

impl CompileStatement for ast::ForStmt<'_> {
    fn compile(&self, context: &mut Context, output: &mut dyn Output) -> Result<(), String> {
        let count = match self.count.value.clone().try_into() {
            Ok(x) => x,
            Err(x) => return Err(format!("Failed to convert loop iteration count: {}", x)),
        };
        for i in 0..count {
            context.set_comptime_variable(self.variable.ident, i);
            self.body.compile(context, output)?;
        }

        Ok(())
    }
}

impl CompileStatement for ast::Stmt<'_> {
    fn compile(&self, context: &mut Context, output: &mut dyn Output) -> Result<(), String> {
        match self {
            ast::Stmt::Assign(s) => s.compile(context, output),
            ast::Stmt::For(f) => f.compile(context, output),
        }
    }
}

impl CompileExpr for ast::BlockExpr<'_> {
    fn compile(
        &self,
        context: &mut Context,
        output: &mut dyn Output,
    ) -> Result<Vec<usize>, String> {
        for statement in &self.stmts {
            statement.compile(context, output)?;
        }

        self.result.compile(context, output)
    }
}

impl CompileExpr for ast::FunctionDef<'_> {
    fn compile(
        &self,
        context: &mut Context,
        output: &mut dyn Output,
    ) -> Result<Vec<usize>, String> {
        self.body.compile(context, output)
    }
}

pub fn load_functions<'a>(context: &mut Context<'a>, root: &'a Vec<ast::Def>) {
    for def in root {
        match def {
            ast::Def::Func(func) => context
                .functions
                .insert(func.name.to_string(), func.clone()),
        };
    }
}

pub fn compile_function(
    name: &str,
    context: &mut Context,
    output: &mut dyn Output,
) -> Result<Vec<usize>, String> {
    let func = match context.functions.get(name) {
        Some(func) => func.clone(),
        None => return Err(format!("Cannot find definition for function: `{}`", name)),
    };
    func.compile(context, output)
}
