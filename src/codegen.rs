use crate::ast;

use num_bigint::BigInt;
use std::collections::HashMap;
use std::rc::Rc;

// A node represents an intermediate data in IR
#[derive(Debug, PartialEq)]
pub enum Node {
    Constant(BigInt),
    Parameter(u32),
    Index(Rc<Node>, u32),
    FuncCall(String, Vec<Rc<Node>>),
    Resize(Rc<Node>, u32),
}

#[derive(Debug)]
pub struct SymbolTable<'a> {
    parent: Option<&'a SymbolTable<'a>>,
    symbols: HashMap<String, Rc<Node>>,
}

impl<'a> SymbolTable<'a> {
    pub fn new() -> Self {
        Self {
            parent: None,
            symbols: HashMap::new(),
        }
    }

    pub fn extend(&'a self) -> Self {
        Self {
            parent: Some(self),
            symbols: HashMap::new(),
        }
    }

    pub fn register(&mut self, symbol: &str, node: Rc<Node>) -> Option<Rc<Node>> {
        self.symbols.insert(symbol.to_string(), node)
    }

    pub fn lookup(&self, symbol: &str) -> Option<Rc<Node>> {
        match self.symbols.get(symbol) {
            Some(x) => Some(x.clone()),
            None => match self.parent {
                Some(parent) => parent.lookup(symbol),
                None => None,
            },
        }
    }
}

pub trait CompileExpr {
    fn compile(&self, symbol_table: &mut SymbolTable) -> Result<Rc<Node>, String>;
}

impl CompileExpr for ast::IntExpr {
    fn compile(&self, _: &mut SymbolTable) -> Result<Rc<Node>, String> {
        Ok(Rc::new(Node::Constant(self.value.clone())))
    }
}

impl CompileExpr for ast::VarExpr<'_> {
    fn compile(&self, symbol_table: &mut SymbolTable) -> Result<Rc<Node>, String> {
        let variable = match symbol_table.lookup(self.ident) {
            Some(node) => node,
            None => return Err(format!("Unknown variable referenced: {}", self.ident)),
        };

        match self.index {
            Some(index) => Ok(Rc::new(Node::Index(variable, index))),
            None => Ok(variable),
        }
    }
}

impl CompileExpr for ast::FuncExpr<'_> {
    fn compile(&self, symbol_table: &mut SymbolTable) -> Result<Rc<Node>, String> {
        let arg_nodes = self
            .args
            .iter()
            .map(|x| x.compile(symbol_table))
            .collect::<Result<_, _>>()?;

        Ok(Rc::new(Node::FuncCall(self.ident.to_string(), arg_nodes)))
    }
}

impl CompileExpr for ast::Expr<'_> {
    fn compile(&self, symbol_table: &mut SymbolTable) -> Result<Rc<Node>, String> {
        match self {
            ast::Expr::Int(e) => e.compile(symbol_table),
            ast::Expr::Var(e) => e.compile(symbol_table),
            ast::Expr::Binary(_) => todo!(),
            ast::Expr::Func(e) => e.compile(symbol_table),
        }
    }
}

impl CompileExpr for ast::FunctionDef<'_> {
    fn compile(&self, symbol_table: &mut SymbolTable) -> Result<Rc<Node>, String> {
        let mut function_symbol_table = symbol_table.extend();
        for (i, parameter) in self.param.iter().enumerate() {
            function_symbol_table.register(
                parameter.name,
                Rc::new(Node::Parameter(i.try_into().unwrap())),
            );
        }
        self.body.compile(&mut function_symbol_table)
    }
}

pub trait CompileStatement {
    fn compile(&self, symbol_table: &mut SymbolTable) -> Result<(), String>;
}

impl CompileStatement for ast::Stmt<'_> {
    fn compile(&self, outer_symbol_table: &mut SymbolTable) -> Result<(), String> {
        match self {
            ast::Stmt::Assign(x) => x.compile(outer_symbol_table),
        }
    }
}

impl CompileStatement for ast::AssignmentStmt<'_> {
    fn compile(&self, symbol_table: &mut SymbolTable) -> Result<(), String> {
        let value = self.rhs.compile(symbol_table)?;

        match &self.lhs_type {
            Some(_type) => {
                // this is a typed declaration

                // declarations should not have index
                assert!(self.lhs.index.is_none());

                if let ast::Type::Q(number_of_qubits) = _type {
                    symbol_table.register(
                        self.lhs.ident,
                        Rc::new(Node::Resize(value, *number_of_qubits)),
                    );
                } else {
                    // FIXME: Are not C and N really same?
                    symbol_table.register(self.lhs.ident, value);
                }
            }
            None => {
                symbol_table.register(self.lhs.ident, value);
            }
        }

        Ok(())
    }
}

impl CompileExpr for ast::Block<'_> {
    fn compile(&self, outer_symbol_table: &mut SymbolTable) -> Result<Rc<Node>, String> {
        let mut local_symbol_table = outer_symbol_table.extend();

        for stmt in &self.stmts {
            stmt.compile(&mut local_symbol_table)?;
        }

        self.result.compile(&mut local_symbol_table)
    }
}

impl CompileStatement for ast::Def<'_> {
    fn compile(&self, function_table: &mut SymbolTable) -> Result<(), String> {
        match &self {
            ast::Def::Func(func) => {
                // TODO: Can we not create a new unused symbol table everytime
                let compiled_function = func.compile(&mut SymbolTable::new())?;
                function_table.register(func.name, compiled_function);
            }
        }
        Ok(())
    }
}

mod tests {
    use super::{Node, SymbolTable};
    use num_bigint::BigInt;

    #[test]
    fn symbol_table() {
        use std::rc::Rc;

        let mut parent = SymbolTable::new();
        parent.register("abc", Rc::new(Node::Constant(BigInt::from(12))));

        let mut child = parent.extend();
        child.register("def", Rc::new(Node::Constant(BigInt::from(34))));

        assert_eq!(
            parent.lookup("abc"),
            Some(Rc::new(Node::Constant(BigInt::from(12))))
        );
        assert_eq!(parent.lookup("def"), None);

        assert_eq!(
            child.lookup("abc"),
            Some(Rc::new(Node::Constant(BigInt::from(12))))
        );
        assert_eq!(
            child.lookup("def"),
            Some(Rc::new(Node::Constant(BigInt::from(34))))
        );
    }
}
