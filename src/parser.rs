use std::rc::Rc;

use crate::ast;

use nom::{
    branch::alt,
    bytes::complete::{is_a, tag, take, take_while},
    character::complete::{alpha1, alphanumeric1, anychar, char, digit1, one_of},
    combinator::{eof, map, map_res, recognize},
    multi::{many0, many0_count, separated_list0},
    sequence::{delimited, pair, terminated},
    IResult,
};

use num_bigint::{BigInt, ParseBigIntError};

#[derive(Debug, PartialEq)]
pub enum QiwiError<I> {
    UnknownType,
    ParseInt,
    IndexError,
    GenericNom(I, nom::error::ErrorKind),
}

// TODO: better reporting of int parsing errors
impl<I> nom::error::FromExternalError<I, std::num::ParseIntError> for QiwiError<I> {
    fn from_external_error(
        _input: I,
        _kind: nom::error::ErrorKind,
        _e: std::num::ParseIntError,
    ) -> Self {
        QiwiError::ParseInt
    }
}

impl<I> nom::error::FromExternalError<I, ParseBigIntError> for QiwiError<I> {
    fn from_external_error(_input: I, _kind: nom::error::ErrorKind, _e: ParseBigIntError) -> Self {
        QiwiError::ParseInt
    }
}

impl<I> nom::error::ParseError<I> for QiwiError<I> {
    fn from_error_kind(input: I, kind: nom::error::ErrorKind) -> Self {
        QiwiError::GenericNom(input, kind)
    }
    fn append(_: I, _: nom::error::ErrorKind, other: Self) -> Self {
        other
    }
}

impl<I> From<nom::error::Error<I>> for QiwiError<I> {
    fn from(e: nom::error::Error<I>) -> Self {
        QiwiError::GenericNom(e.input, e.code)
    }
}

fn comment(input: &str) -> IResult<&str, (), QiwiError<&str>> {
    let (input, _) = tag("//")(input)?;
    let (input, _) = take_while(|c| c != '\n')(input)?;

    if let Ok((_, _)) = eof::<_, QiwiError<&str>>(input) {
        // this is the last line and no \n at the end
        Ok((input, ()))
    } else {
        let (input, _) = take(1usize)(input)?;
        Ok((input, ()))
    }
}

fn space(input: &str) -> IResult<&str, (), QiwiError<&str>> {
    let (input, _) = is_a(" \t\r\n")(input)?;
    Ok((input, ()))
}

// This function supports comments inside space as well
fn spacing(input: &str) -> IResult<&str, (), QiwiError<&str>> {
    let (input, _) = many0_count(alt((comment, space)))(input)?;
    Ok((input, ()))
}

pub fn const_int(input: &str) -> IResult<&str, BigInt, QiwiError<&str>> {
    use std::str::FromStr;
    map_res(digit1, BigInt::from_str)(input)
}

pub fn symbol(input: &str) -> IResult<&str, &str, QiwiError<&str>> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ))(input)
}

pub fn _type(input: &str) -> IResult<&str, ast::Type, QiwiError<&str>> {
    let (input, typename) = anychar(input)?;

    if typename == 'C' {
        Ok((input, ast::Type::C))
    } else if typename == 'N' {
        Ok((input, ast::Type::N))
    } else if typename == 'Q' {
        use std::str::FromStr;
        let (input, len) = map_res(digit1, usize::from_str)(input)?;
        Ok((input, ast::Type::Q(len)))
    } else {
        Err(nom::Err::Error(QiwiError::UnknownType))
    }
}

pub fn typed_symbol(input: &str) -> IResult<&str, ast::TypedSymbol, QiwiError<&str>> {
    let (input, name) = recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ))(input)?;

    let (input, _) = spacing(input)?;
    let (input, _) = tag(":")(input)?;
    let (input, _) = spacing(input)?;
    let (input, _type) = _type(input)?;

    Ok((input, ast::TypedSymbol { name, _type }))
}

pub fn expression_int(input: &str) -> IResult<&str, ast::IntExpr, QiwiError<&str>> {
    let (input, value) = const_int(input)?;
    Ok((input, ast::IntExpr { value }))
}

// index is optional
pub fn expression_indexed_var(input: &str) -> IResult<&str, ast::VarExpr, QiwiError<&str>> {
    let (input, name) = symbol(input)?;
    match delimited(tag("["), expression, tag("]"))(input) {
        Ok((input, index)) => Ok((
            input,
            ast::VarExpr {
                ident: name,
                index: Some(Rc::new(index)),
            },
        )),
        Err(nom::Err::Error(QiwiError::ParseInt)) => Err(nom::Err::Error(QiwiError::IndexError)),
        Err(_) => Ok((
            input,
            ast::VarExpr {
                ident: name,
                index: None,
            },
        )),
    }
}

pub fn expression_paren(input: &str) -> IResult<&str, ast::Expr, QiwiError<&str>> {
    let (input, _) = tag("(")(input)?;
    let (input, _) = spacing(input)?;
    let (input, content) = expression(input)?;
    let (input, _) = spacing(input)?;
    let (input, _) = tag(")")(input)?;
    Ok((input, content))
}

pub fn expression_function_call(input: &str) -> IResult<&str, ast::FuncExpr, QiwiError<&str>> {
    let (input, ident) = symbol(input)?;
    let (input, _) = spacing(input)?;
    let (input, _) = tag("(")(input)?;
    let (input, args) = delimited(
        spacing,
        separated_list0(delimited(spacing, tag(","), spacing), expression),
        spacing,
    )(input)?;
    let (input, _) = tag(")")(input)?;

    Ok((input, ast::FuncExpr { ident, args }))
}

pub fn expression_unary(input: &str) -> IResult<&str, ast::Expr, QiwiError<&str>> {
    alt((
        map(expression_paren, |x| x),
        map(expression_function_call, ast::Expr::Func),
        map(expression_indexed_var, ast::Expr::Var),
        map(expression_int, ast::Expr::Int),
    ))(input)
}

const BINARY_OPERATORS: &str = "+-&|^";

pub fn expression(input: &str) -> IResult<&str, ast::Expr, QiwiError<&str>> {
    let (input, lhs) = expression_unary(input)?;
    let (input_after_space, _) = spacing(input)?;

    if let Ok((input, op)) =
        one_of::<_, _, (&str, nom::error::ErrorKind)>(BINARY_OPERATORS)(input_after_space)
    {
        // this is a binary expressions
        let (input, _) = spacing(input)?;

        let (input, rhs) = expression_unary(input)?;
        Ok((
            input,
            ast::Expr::Binary(ast::BinaryExpr {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            }),
        ))
    } else {
        Ok((input, lhs))
    }
}

pub fn expression_len(input: &str) -> IResult<&str, ast::LenExpr, QiwiError<&str>> {
    let (input, _) = tag("|")(input)?;
    let (input, _) = spacing(input)?;
    let (input, variable) = expression_indexed_var(input)?;
    let (input, _) = spacing(input)?;
    let (input, _) = tag("|")(input)?;
    Ok((input, ast::LenExpr { variable }))
}

pub fn const_expression(input: &str) -> IResult<&str, ast::ConstExpr, QiwiError<&str>> {
    alt((
        map(expression_len, ast::ConstExpr::Len),
        map(expression_int, ast::ConstExpr::Int),
        map(expression_indexed_var, ast::ConstExpr::Var),
    ))(input)
}

pub fn statement_assignment(input: &str) -> IResult<&str, ast::Stmt, QiwiError<&str>> {
    let (input, lhs, lhs_type) = if let Ok((input, lhs)) = typed_symbol(input) {
        // declaration & initialization
        (
            input,
            ast::LhsExpr::Single(ast::VarExpr {
                ident: lhs.name,
                index: None,
            }),
            Some(lhs._type),
        )
    } else if let Ok((input, lhs)) = expression_indexed_var(input) {
        // assignment
        (input, ast::LhsExpr::Single(lhs), None)
    } else {
        let (input, _) = tag("(")(input)?;
        let (input, elements) = delimited(
            spacing,
            separated_list0(
                delimited(spacing, tag(","), spacing),
                expression_indexed_var,
            ),
            spacing,
        )(input)?;
        let (input, _) = tag(")")(input)?;

        (
            input,
            ast::LhsExpr::Tuple(ast::TupleLhsExpr { elements }),
            None,
        )
    };

    let (input, _) = spacing(input)?;
    let (input, _) = char('=')(input)?;
    let (input, _) = spacing(input)?;

    let (input, rhs) = expression(input)?;

    Ok((
        input,
        ast::Stmt::Assign(ast::AssignmentStmt {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            lhs_type,
        }),
    ))
}

pub fn statement_for(input: &str) -> IResult<&str, ast::Stmt, QiwiError<&str>> {
    let (input, _) = tag("for")(input)?;
    let (input, _) = spacing(input)?;
    let (input, _) = tag("(")(input)?;
    let (input, variable) = delimited(spacing, symbol, spacing)(input)?;
    let (input, _) = tag(",")(input)?;
    let (input, value) = delimited(spacing, const_expression, spacing)(input)?;
    let (input, _) = tag(")")(input)?;
    let (input, _) = spacing(input)?;
    let (input, body) = block(input)?;

    Ok((
        input,
        ast::Stmt::For(ast::ForStmt {
            variable: ast::VarExpr {
                ident: variable,
                index: None,
            },
            count: value,
            body,
        }),
    ))
}

pub fn statement(input: &str) -> IResult<&str, ast::Stmt, QiwiError<&str>> {
    terminated(
        alt((statement_assignment, statement_for)),
        delimited(spacing, tag(";"), spacing),
    )(input)
}

pub fn block(input: &str) -> IResult<&str, ast::Block, QiwiError<&str>> {
    let (input, _) = tag("{")(input)?;
    let (input, _) = spacing(input)?;
    let (input, body) = many0(statement)(input)?;

    let (input, _) = tag("}")(input)?;

    Ok((input, ast::Block { stmts: body }))
}

pub fn expression_block(input: &str) -> IResult<&str, ast::BlockExpr, QiwiError<&str>> {
    let (input, _) = tag("{")(input)?;
    let (input, _) = spacing(input)?;
    let (input, body) = many0(statement)(input)?;

    // return value
    let (input, result) = delimited(spacing, expression, spacing)(input)?;

    let (input, _) = tag("}")(input)?;

    Ok((
        input,
        ast::BlockExpr {
            stmts: body,
            result,
        },
    ))
}

pub fn function_parameter(input: &str) -> IResult<&str, ast::TypedParameter, QiwiError<&str>> {
    let (input, name) = recognize(pair(
        alt((alpha1, tag("_"))),
        many0_count(alt((alphanumeric1, tag("_")))),
    ))(input)?;
    let (input, _) = spacing(input)?;
    let (input, _) = tag(":")(input)?;
    let (input, _) = spacing(input)?;

    if let Ok((input, _)) = tag::<_, _, QiwiError<&str>>("persist")(input) {
        let (input, _) = spacing(input)?;
        let (input, _type) = _type(input)?;
        Ok((
            input,
            ast::TypedParameter {
                name,
                _type,
                persist: true,
            },
        ))
    } else {
        let (input, _type) = _type(input)?;
        Ok((
            input,
            ast::TypedParameter {
                name,
                _type,
                persist: false,
            },
        ))
    }
}

pub fn function_def(input: &str) -> IResult<&str, ast::Def, QiwiError<&str>> {
    let (input, _) = tag("fn")(input)?;
    let (input, _) = spacing(input)?;
    let (input, name) = symbol(input)?;
    let (input, _) = spacing(input)?;
    let (input, _) = tag("(")(input)?;
    let (input, args) = delimited(
        spacing,
        separated_list0(delimited(spacing, tag(","), spacing), function_parameter),
        spacing,
    )(input)?;
    let (input, _) = tag(")")(input)?;
    let (input, _) = spacing(input)?;
    let (input, body) = expression_block(input)?;

    Ok((
        input,
        ast::Def::Func(Rc::new(ast::FunctionDef {
            name,
            param: args,
            body,
        })),
    ))
}

pub fn parse_source(input: &str) -> Result<Vec<ast::Def>, QiwiError<&str>> {
    let mut results = Vec::new();

    let mut rem_input = spacing(input).unwrap().0;
    while !rem_input.is_empty() {
        match function_def(rem_input) {
            Ok((input, def)) => {
                rem_input = spacing(input).unwrap().0;
                results.push(def);
            }
            Err(nom::Err::Error(e)) => {
                return Err(e);
            }
            _ => unreachable!(),
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use crate::ast;
    use num_bigint::BigInt;
    use std::{rc::Rc, str::FromStr};

    #[test]
    fn space() {
        assert_eq!(super::space(" "), Ok(("", ())));
        assert_eq!(super::space("  "), Ok(("", ())));
        assert_eq!(super::space("\n"), Ok(("", ())));
        assert_eq!(super::space("\t"), Ok(("", ())));
        assert_ne!(super::space("a"), Ok(("", ())));
    }

    #[test]
    fn comment() {
        assert_eq!(super::comment("// Hello, World!"), Ok(("", ())));
        assert_eq!(super::comment("// Hello, World!\n"), Ok(("", ())));
        assert_eq!(super::comment("// Hello, World!\nabc"), Ok(("abc", ())));

        assert_eq!(super::spacing("  // Hello, World!\n  "), Ok(("", ())));
    }

    #[test]
    fn integer() {
        assert_eq!(
            super::const_int("123456789"),
            Ok(("", BigInt::from_str("123456789").unwrap()))
        );
    }

    #[test]
    fn symbols() {
        for sym in ["abc", "Abc", "abc_def", "_abc", "abc123"] {
            assert_eq!(super::symbol(sym), Ok(("", sym)));
        }

        assert_eq!(super::symbol("abc "), Ok((" ", "abc")));

        assert!(super::symbol("1abc").is_err());
        assert!(super::symbol("123").is_err());
    }

    #[test]
    fn expression_unary() {
        assert_eq!(
            super::expression("abc"),
            Ok((
                "",
                ast::Expr::Var(ast::VarExpr {
                    ident: "abc",
                    index: None
                })
            ))
        );

        assert_eq!(
            super::expression("abc[42]"),
            Ok((
                "",
                ast::Expr::Var(ast::VarExpr {
                    ident: "abc",
                    index: Some(42)
                })
            ))
        );

        assert_eq!(
            super::expression("42"),
            Ok((
                "",
                ast::Expr::Int(ast::IntExpr {
                    value: BigInt::from(42)
                })
            ))
        );

        // extra space should not be parsed
        assert_eq!(
            super::expression("42 "),
            Ok((
                " ",
                ast::Expr::Int(ast::IntExpr {
                    value: BigInt::from(42)
                })
            ))
        );
    }

    #[test]
    fn expression_binary() {
        for s in ["abc-42", "abc - 42", "abc -42", "abc- 42"] {
            assert_eq!(
                super::expression(s),
                Ok((
                    "",
                    ast::Expr::Binary(ast::BinaryExpr {
                        op: '-',
                        lhs: Box::new(ast::Expr::Var(ast::VarExpr {
                            ident: "abc",
                            index: None
                        })),
                        rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from_str("42").unwrap()
                        })),
                    })
                ))
            );
        }
    }

    #[test]
    fn expression_parenthesis() {
        for s in ["(abc + 42)", "( abc + 42)", "(abc + 42 )", "( abc + 42 )"] {
            assert_eq!(
                super::expression(s),
                Ok((
                    "",
                    ast::Expr::Binary(ast::BinaryExpr {
                        op: '+',
                        lhs: Box::new(ast::Expr::Var(ast::VarExpr {
                            ident: "abc",
                            index: None
                        })),
                        rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from_str("42").unwrap()
                        })),
                    })
                ))
            );
        }
    }

    #[test]
    fn expression_function_call() {
        for s in ["abc()", "abc ()", "abc( )"] {
            assert_eq!(
                super::expression(s),
                Ok((
                    "",
                    ast::Expr::Func(ast::FuncExpr {
                        ident: "abc",
                        args: vec![],
                    })
                ))
            );
        }

        for s in ["abc(xyz)", "abc( xyz)", "abc(xyz )", "abc( xyz )"] {
            assert_eq!(
                super::expression(s),
                Ok((
                    "",
                    ast::Expr::Func(ast::FuncExpr {
                        ident: "abc",
                        args: vec![ast::Expr::Var(ast::VarExpr {
                            ident: "xyz",
                            index: None
                        })],
                    })
                ))
            );
        }

        for s in [
            "abc(xyz,42)",
            "abc(xyz, 42)",
            "abc(xyz ,42)",
            "abc(xyz , 42)",
        ] {
            assert_eq!(
                super::expression(s),
                Ok((
                    "",
                    ast::Expr::Func(ast::FuncExpr {
                        ident: "abc",
                        args: vec![
                            ast::Expr::Var(ast::VarExpr {
                                ident: "xyz",
                                index: None
                            }),
                            ast::Expr::Int(ast::IntExpr {
                                value: BigInt::from(42)
                            }),
                        ],
                    })
                ))
            );
        }
    }

    #[test]
    fn statement_assignment() {
        for s in ["x = 42", "x=42", "x =42", "x= 42"] {
            assert_eq!(
                super::statement_assignment(s),
                Ok((
                    "",
                    ast::Stmt::Assign(ast::AssignmentStmt {
                        lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                            ident: "x",
                            index: None
                        })),
                        rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from_str("42").unwrap()
                        })),
                        lhs_type: None,
                    })
                ))
            );
        }
        for s in ["x : Q6 = 42", "x: Q6 = 42", "x :Q6 = 42", "x : Q6= 42"] {
            assert_eq!(
                super::statement_assignment(s),
                Ok((
                    "",
                    ast::Stmt::Assign(ast::AssignmentStmt {
                        lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                            ident: "x",
                            index: None
                        })),
                        rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from_str("42").unwrap()
                        })),
                        lhs_type: Some(ast::Type::Q(6)),
                    })
                ))
            );
        }
        for s in ["(x,y) = 42", "(x, y) = 42", "( x ,y ) = 42", "(x , y)= 42"] {
            assert_eq!(
                super::statement_assignment(s),
                Ok((
                    "",
                    ast::Stmt::Assign(ast::AssignmentStmt {
                        lhs: Box::new(ast::LhsExpr::Tuple(ast::TupleLhsExpr {
                            elements: vec![
                                ast::VarExpr {
                                    ident: "x",
                                    index: None
                                },
                                ast::VarExpr {
                                    ident: "y",
                                    index: None
                                }
                            ]
                        })),
                        rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from_str("42").unwrap()
                        })),
                        lhs_type: None,
                    })
                ))
            );
        }
    }

    #[test]
    fn terminated_statement() {
        for s in ["x = 42;", "x = 42 ;", "x = 42 ; "] {
            assert_eq!(
                super::statement(s),
                Ok((
                    "",
                    ast::Stmt::Assign(ast::AssignmentStmt {
                        lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                            ident: "x",
                            index: None
                        })),
                        rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from(42)
                        })),
                        lhs_type: None,
                    }),
                ))
            );
        }
    }

    #[test]
    fn statement_for() {
        for s in ["for ( i , 42 ) { x = 42; }", "for(i,42){ x = 42; }"] {
            assert_eq!(
                super::statement_for(s),
                Ok((
                    "",
                    ast::Stmt::For(ast::ForStmt {
                        variable: ast::VarExpr {
                            ident: "i",
                            index: None
                        },
                        count: ast::IntExpr {
                            value: BigInt::from(42)
                        },
                        body: ast::Block {
                            stmts: vec![ast::Stmt::Assign(ast::AssignmentStmt {
                                lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                                    ident: "x",
                                    index: None
                                })),
                                rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                                    value: BigInt::from(42)
                                })),
                                lhs_type: None,
                            })]
                        }
                    })
                ))
            );
        }
    }

    #[test]
    fn block() {
        assert_eq!(
            super::expression_block("{ x }"),
            Ok((
                "",
                ast::BlockExpr {
                    stmts: vec![],
                    result: ast::Expr::Var(ast::VarExpr {
                        ident: "x",
                        index: None
                    }),
                }
            ))
        );

        for s in ["{x = 42; x}", "{ x = 42 ; x }"] {
            assert_eq!(
                super::expression_block(s),
                Ok((
                    "",
                    ast::BlockExpr {
                        stmts: vec![ast::Stmt::Assign(ast::AssignmentStmt {
                            lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                                ident: "x",
                                index: None
                            })),
                            rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                                value: BigInt::from_str("42").unwrap()
                            })),
                            lhs_type: None,
                        })],
                        result: ast::Expr::Var(ast::VarExpr {
                            ident: "x",
                            index: None
                        }),
                    }
                ))
            );
        }

        for s in ["{x = 42; }", "{ x = 42 ;}"] {
            assert_eq!(
                super::block(s),
                Ok((
                    "",
                    ast::Block {
                        stmts: vec![ast::Stmt::Assign(ast::AssignmentStmt {
                            lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                                ident: "x",
                                index: None
                            })),
                            rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                                value: BigInt::from_str("42").unwrap()
                            })),
                            lhs_type: None,
                        })],
                    }
                ))
            );
        }

        for s in ["{x = 42; y = x; y}", "{ x = 42 ; y = x; y }"] {
            assert_eq!(
                super::expression_block(s),
                Ok((
                    "",
                    ast::BlockExpr {
                        stmts: vec![
                            ast::Stmt::Assign(ast::AssignmentStmt {
                                lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                                    ident: "x",
                                    index: None
                                })),
                                rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                                    value: BigInt::from_str("42").unwrap()
                                })),
                                lhs_type: None,
                            }),
                            ast::Stmt::Assign(ast::AssignmentStmt {
                                lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                                    ident: "y",
                                    index: None
                                })),
                                rhs: Box::new(ast::Expr::Var(ast::VarExpr {
                                    ident: "x",
                                    index: None
                                })),
                                lhs_type: None,
                            })
                        ],
                        result: ast::Expr::Var(ast::VarExpr {
                            ident: "y",
                            index: None
                        }),
                    }
                ))
            );
        }

        for s in ["{x = 42; y = x;}", "{ x = 42 ; y = x; }"] {
            assert_eq!(
                super::block(s),
                Ok((
                    "",
                    ast::Block {
                        stmts: vec![
                            ast::Stmt::Assign(ast::AssignmentStmt {
                                lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                                    ident: "x",
                                    index: None
                                })),
                                rhs: Box::new(ast::Expr::Int(ast::IntExpr {
                                    value: BigInt::from_str("42").unwrap()
                                })),
                                lhs_type: None,
                            }),
                            ast::Stmt::Assign(ast::AssignmentStmt {
                                lhs: Box::new(ast::LhsExpr::Single(ast::VarExpr {
                                    ident: "y",
                                    index: None
                                })),
                                rhs: Box::new(ast::Expr::Var(ast::VarExpr {
                                    ident: "x",
                                    index: None
                                })),
                                lhs_type: None,
                            })
                        ],
                    }
                ))
            );
        }
    }

    #[test]
    fn typed_symbol() {
        assert_eq!(
            super::typed_symbol("abc: C"),
            Ok((
                "",
                ast::TypedSymbol {
                    name: "abc",
                    _type: ast::Type::C
                }
            ))
        );

        assert_eq!(
            super::typed_symbol("abc: N"),
            Ok((
                "",
                ast::TypedSymbol {
                    name: "abc",
                    _type: ast::Type::N
                }
            ))
        );

        assert_eq!(
            super::typed_symbol("abc: Q12"),
            Ok((
                "",
                ast::TypedSymbol {
                    name: "abc",
                    _type: ast::Type::Q(12)
                }
            ))
        );
    }

    #[test]
    fn function_definition() {
        assert_eq!(
            super::function_def("fn one() { 1 }"),
            Ok((
                "",
                ast::Def::Func(Rc::new(ast::FunctionDef {
                    name: "one",
                    param: vec![],
                    body: ast::BlockExpr {
                        stmts: vec![],
                        result: ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from(1)
                        })
                    }
                }))
            ))
        );
        assert_eq!(
            super::function_def("fn aes256(data: Q128, key: Q128) { 1 }"),
            Ok((
                "",
                ast::Def::Func(Rc::new(ast::FunctionDef {
                    name: "aes256",
                    param: vec![
                        ast::TypedParameter {
                            name: "data",
                            _type: ast::Type::Q(128),
                            persist: false
                        },
                        ast::TypedParameter {
                            name: "key",
                            _type: ast::Type::Q(128),
                            persist: false
                        }
                    ],
                    body: ast::BlockExpr {
                        stmts: vec![],
                        result: ast::Expr::Int(ast::IntExpr {
                            value: BigInt::from(1)
                        })
                    }
                }))
            ))
        );
        assert_eq!(
            super::function_def("fn add(a: N, b: persist N) { a + b }"),
            Ok((
                "",
                ast::Def::Func(Rc::new(ast::FunctionDef {
                    name: "add",
                    param: vec![
                        ast::TypedParameter {
                            name: "a",
                            _type: ast::Type::N,
                            persist: false
                        },
                        ast::TypedParameter {
                            name: "b",
                            _type: ast::Type::N,
                            persist: true
                        }
                    ],
                    body: ast::BlockExpr {
                        stmts: vec![],
                        result: ast::Expr::Binary(ast::BinaryExpr {
                            op: '+',
                            lhs: Box::new(ast::Expr::Var(ast::VarExpr {
                                ident: "a",
                                index: None
                            })),
                            rhs: Box::new(ast::Expr::Var(ast::VarExpr {
                                ident: "b",
                                index: None
                            }))
                        })
                    }
                }))
            ))
        );
    }
}
