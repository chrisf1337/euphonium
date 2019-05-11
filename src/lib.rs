pub mod ast;
pub mod lexer;

use lalrpop_util::lalrpop_mod;
lalrpop_mod!(pub parser);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;

    macro_rules! se {
        ( $ty:expr ) => {
            StrippedExpr { ty: $ty }
        };
    }

    macro_rules! sd {
        ( $ty:expr ) => {
            StrippedDecl { ty: $ty }
        };
    }

    #[test]
    fn test_tydecls() {
        let lexer = lexer::Lexer::new(include_str!("../test/test_tydecls.euph"));
        let mut errors = vec![];
        let parse_result = parser::ProgramParser::new().parse(&mut errors, lexer).map(|decls| decls.into_iter().map(Decl::strip).collect::<Vec<StrippedDecl>>());
        assert_eq!(errors, vec![]);
        assert_eq!(
            parse_result,
            Ok(vec![
                sd!(DeclType::Type(TypeDecl {
                    type_id: "a".to_owned(),
                    ty: Type::Type("int".to_owned())
                })),
                sd!(DeclType::Type(TypeDecl {
                    type_id: "b".to_owned(),
                    ty: Type::Array(
                        Box::new(Type::Record(vec![
                            TypeField {
                                id: "a".to_owned(),
                                ty: Type::Type("int".to_owned())
                            },
                            TypeField {
                                id: "b".to_owned(),
                                ty: Type::Type("int".to_owned())
                            }
                        ])),
                        3
                    )
                })),
            ])
        );
    }

    #[test]
    fn test_exprs() {
        let lexer = lexer::Lexer::new(include_str!("../test/test_exprs.euph"));
        let mut errors = vec![];
        let parse_result = parser::ProgramParser::new()
            .parse(&mut errors, lexer)
            .map(|decls| {
                decls
                    .into_iter()
                    .map(Decl::strip)
                    .collect::<Vec<StrippedDecl>>()
            });
        assert_eq!(errors, vec![]);
        assert_eq!(
            parse_result,
            Ok(vec![sd!(DeclType::Fn(FnDecl {
                id: "f".to_owned(),
                type_fields: vec![],
                return_type: None,
                body: se!(ExprType::Seq(
                    vec![
                        se!(ExprType::Arith(
                            Box::new(se!(ExprType::Number(1))),
                            ArithOp::Add,
                            Box::new(se!(ExprType::Arith(
                                Box::new(se!(ExprType::Number(2))),
                                ArithOp::Mul,
                                Box::new(se!(ExprType::Number(3)))
                            )))
                        )),
                        se!(ExprType::Bool(
                            Box::new(se!(ExprType::Not(Box::new(se!(ExprType::BoolLiteral(
                                true
                            )))))),
                            BoolOp::Or,
                            Box::new(se!(ExprType::Bool(
                                Box::new(se!(ExprType::BoolLiteral(true))),
                                BoolOp::And,
                                Box::new(se!(ExprType::BoolLiteral(false)))
                            )))
                        )),
                        se!(ExprType::Assign(Box::new(Assign {
                            lval: LVal::Field(
                                Box::new(LVal::Subscript(
                                    Box::new(LVal::Simple("a".to_owned())),
                                    se!(ExprType::LVal(Box::new(LVal::Simple("b".to_owned()))))
                                )),
                                "c".to_owned()
                            ),
                            expr: se!(ExprType::Number(1)),
                        }))),
                        se!(ExprType::FnCall(Box::new(FnCall {
                            id: "g".to_owned(),
                            args: vec![se!(ExprType::Number(1)), se!(ExprType::Number(2)),]
                        }))),
                        se!(ExprType::Record(Box::new(Record {
                            id: "record".to_owned(),
                            field_assigns: vec![
                                FieldAssign {
                                    id: "a".to_owned(),
                                    expr: se!(ExprType::Number(1)),
                                },
                                FieldAssign {
                                    id: "b".to_owned(),
                                    expr: se!(ExprType::Number(2)),
                                }
                            ]
                        }))),
                        se!(ExprType::If(Box::new(If {
                            cond: se!(ExprType::Compare(
                                Box::new(se!(ExprType::Number(1))),
                                CompareOp::Equal,
                                Box::new(se!(ExprType::Seq(vec![se!(ExprType::Number(1))], true)))
                            )),
                            then_expr: se!(ExprType::Seq(vec![se!(ExprType::Number(2))], true)),
                            else_expr: Some(se!(ExprType::If(Box::new(If {
                                cond: se!(ExprType::Number(3)),
                                then_expr: se!(ExprType::Seq(vec![se!(ExprType::Number(4))], true)),
                                else_expr: Some(se!(ExprType::Seq(
                                    vec![se!(ExprType::Unit)],
                                    true
                                ))),
                            }))))
                        }))),
                        se!(ExprType::If(Box::new(If {
                            cond: se!(ExprType::Seq(vec![se!(ExprType::BoolLiteral(true))], true)),
                            then_expr: se!(ExprType::Seq(vec![se!(ExprType::Unit)], true)),
                            else_expr: None,
                        }))),
                        se!(ExprType::Array(Box::new(Array {
                            initial_value: se!(ExprType::Number(0)),
                            len: se!(ExprType::Number(3)),
                        }))),
                        se!(ExprType::Range(
                            Box::new(se!(ExprType::Seq(vec![se!(ExprType::Number(1))], true))),
                            Box::new(se!(ExprType::Number(2)))
                        ))
                    ],
                    false
                ))
            }))])
        )
    }

    #[test]
    fn test_parser() {
        let lexer = lexer::Lexer::new(include_str!("../test/test1.euph"));
        let mut errors = vec![];
        let parse_result = dbg!(parser::ProgramParser::new()
            .parse(&mut errors, lexer))
            .map(|decls| {
                decls
                    .into_iter()
                    .map(Decl::strip)
                    .collect::<Vec<StrippedDecl>>()
            });
        assert_eq!(errors, vec![]);
        assert_eq!(
            parse_result,
            Ok(vec![
                sd!(DeclType::Fn(FnDecl {
                    id: "f".to_owned(),
                    type_fields: vec![],
                    return_type: None,
                    body: se!(ExprType::Seq(
                        vec![
                            se!(ExprType::Let(Box::new(Let {
                                pattern: Pattern::String("a".to_owned()),
                                mutable: false,
                                ty: Some(Type::Type("a".to_owned())),
                                expr: se!(ExprType::Arith(
                                    Box::new(se!(ExprType::Number(1))),
                                    ArithOp::Sub,
                                    Box::new(se!(ExprType::Neg(Box::new(se!(ExprType::Number(
                                        1
                                    ))))))
                                ))
                            }))),
                            se!(ExprType::Let(Box::new(Let {
                                pattern: Pattern::String("b".to_owned()),
                                mutable: false,
                                ty: None,
                                expr: se!(ExprType::String("".to_owned()))
                            }))),
                            se!(ExprType::Arith(
                                Box::new(se!(ExprType::LVal(Box::new(LVal::Simple(
                                    "a".to_owned()
                                ))))),
                                ArithOp::Add,
                                Box::new(se!(ExprType::LVal(Box::new(LVal::Simple(
                                    "b".to_owned()
                                )))))
                            )),
                        ],
                        true
                    ))
                })),
                sd!(DeclType::Fn(FnDecl {
                    id: "g".to_owned(),
                    type_fields: vec![TypeField {
                        id: "i".to_owned(),
                        ty: Type::Type("int".to_owned())
                    }],
                    return_type: Some(Type::Type("int".to_owned())),
                    body: se!(ExprType::Arith(
                        Box::new(se!(ExprType::Number(1))),
                        ArithOp::Div,
                        Box::new(se!(ExprType::Number(2)))
                    )),
                })),
            ])
        );
    }
}
