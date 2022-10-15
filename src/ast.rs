use num_bigint::BigInt;

#[derive(Debug, PartialEq, Eq)]
pub enum Type {
    C,      // Classical integer
    Q(u32), // Fixed sized quantum integer
    N,      // Variable sized quantum integer
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
    pub index: Option<u32>,
}

impl Expression for VarExpr<'_> {}

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
    pub lhs: Box<VarExpr<'a>>,
    pub rhs: Box<Expr<'a>>,
    pub lhs_type: Option<Type>,
}

impl Statement for AssignmentStmt<'_> {}

#[derive(Debug, PartialEq, Eq)]
pub enum Stmt<'a> {
    Assign(AssignmentStmt<'a>),
}

#[derive(Debug, PartialEq, Eq)]
pub struct Block<'a> {
    pub(crate) stmts: Vec<Stmt<'a>>,
    pub(crate) result: Expr<'a>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FunctionDef<'a> {
    pub name: &'a str,
    pub param: Vec<TypedParameter<'a>>,
    pub body: Block<'a>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Def<'a> {
    Func(FunctionDef<'a>),
}
