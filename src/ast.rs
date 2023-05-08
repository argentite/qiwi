use std::rc::Rc;

use num_bigint::BigInt;

#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    C,        // Classical integer
    Q(usize), // Fixed sized quantum integer
    N,        // Variable sized quantum integer
}

impl Type {
    pub fn size(&self) -> Option<usize> {
        match self {
            Type::C => None,
            Type::Q(n) => Some(*n),
            Type::N => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Ident<'a> {
    pub name: &'a str,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TypedSymbol<'a> {
    pub name: &'a str,
    pub _type: Type,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TypedParameter<'a> {
    pub name: &'a str,
    pub _type: Type,
    pub persist: bool,
}

pub trait Expression: std::fmt::Debug + Eq {}

#[derive(Debug, PartialEq, Eq)]
pub struct IntExpr {
    pub value: BigInt,
}

impl Expression for IntExpr {}

#[derive(Debug, PartialEq, Eq)]
pub struct VarExpr<'a> {
    pub ident: &'a str,
    pub index: Option<Rc<Expr<'a>>>,
}

impl Expression for VarExpr<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub struct TupleExpr<'a> {
    pub elements: Vec<Expr<'a>>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TupleLhsExpr<'a> {
    pub elements: Vec<VarExpr<'a>>,
}

impl Expression for TupleExpr<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub enum LhsExpr<'a> {
    Single(VarExpr<'a>),
    Tuple(TupleLhsExpr<'a>),
}

impl Expression for LhsExpr<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub struct BinaryExpr<'a> {
    pub op: char,
    pub lhs: Box<Expr<'a>>,
    pub rhs: Box<Expr<'a>>,
}

impl Expression for BinaryExpr<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub struct FuncExpr<'a> {
    pub ident: &'a str,
    pub args: Vec<Expr<'a>>,
}

impl Expression for FuncExpr<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub enum Expr<'a> {
    Int(IntExpr),
    Var(VarExpr<'a>),
    Binary(BinaryExpr<'a>),
    Func(FuncExpr<'a>),
}

pub trait Statement: std::fmt::Debug + Eq {}

#[derive(Debug, PartialEq, Eq)]
pub struct AssignmentStmt<'a> {
    pub lhs: Box<LhsExpr<'a>>,
    pub rhs: Box<Expr<'a>>,
    pub lhs_type: Option<Type>,
}

impl Statement for AssignmentStmt<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub struct ForStmt<'a> {
    pub variable: VarExpr<'a>,
    pub count: IntExpr,
    pub body: Block<'a>,
}

impl Statement for ForStmt<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub enum Stmt<'a> {
    Assign(AssignmentStmt<'a>),
    For(ForStmt<'a>),
}

#[derive(Debug, PartialEq, Eq)]
pub struct BlockExpr<'a> {
    pub stmts: Vec<Stmt<'a>>,
    pub result: Expr<'a>,
}

impl Expression for BlockExpr<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub struct Block<'a> {
    pub stmts: Vec<Stmt<'a>>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionDef<'a> {
    pub name: &'a str,
    pub param: Vec<TypedParameter<'a>>,
    pub body: BlockExpr<'a>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Def<'a> {
    Func(Rc<FunctionDef<'a>>),
}
