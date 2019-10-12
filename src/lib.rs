#![warn(clippy::all)]

#[macro_use]
mod utils;

pub mod ast;
mod fragment;
mod frame;
mod ir;
pub mod lexer;
mod log;
pub mod parser;
pub mod sourcemap;
pub mod tmp;
mod translate;
mod ty;
pub mod typecheck;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ast::*, utils::EMPTY_SOURCEMAP};

    #[test]
    fn tydecls() -> parser::Result<()> {
        let decls: Vec<_DeclType> =
            parser::parse_program(EMPTY_SOURCEMAP.1, include_str!("../test-euph-files/test_tydecls.euph"))?
                .into_iter()
                .map(|decl| decl.t.into())
                .collect();
        assert_eq!(
            decls,
            vec![
                _DeclType::Type(_TypeDecl {
                    type_id: "a".to_owned(),
                    ty: _TypeDeclType::Type("int".to_owned())
                }),
                _DeclType::Type(_TypeDecl {
                    type_id: "b".to_owned(),
                    ty: _TypeDeclType::Array(_Type::Type("a".to_owned()), 3)
                }),
                _DeclType::Type(_TypeDecl {
                    type_id: "Optionint".to_owned(),
                    ty: _TypeDeclType::Enum(vec![
                        _EnumCase {
                            id: "Some".to_owned(),
                            params: vec![_Type::Type("int".to_owned())]
                        },
                        _EnumCase {
                            id: "None".to_owned(),
                            params: vec![],
                        }
                    ])
                }),
                _DeclType::Type(_TypeDecl {
                    type_id: "f".to_owned(),
                    ty: _TypeDeclType::Fn(vec![_Type::Type("int".to_owned())], _Type::Unit),
                }),
                _DeclType::Type(_TypeDecl {
                    type_id: "g".to_owned(),
                    ty: _TypeDeclType::Fn(vec![], _Type::Type("int".to_owned())),
                }),
            ]
        );
        Ok(())
    }

    #[test]
    fn exprs() -> parser::Result<()> {
        let decls: Vec<_DeclType> =
            parser::parse_program(EMPTY_SOURCEMAP.1, include_str!("../test-euph-files/test_exprs.euph"))?
                .into_iter()
                .map(|decl| decl.t.into())
                .collect();
        assert_eq!(
            decls,
            vec![_DeclType::Fn(_FnDecl {
                id: "f".to_owned(),
                type_fields: vec![],
                return_type: None,
                body: _ExprType::Seq(
                    vec![
                        _ExprType::Arith(Box::new(_Arith {
                            l: _ExprType::Number(1),
                            op: ArithOp::Add,
                            r: _ExprType::Arith(Box::new(_Arith {
                                l: _ExprType::Number(2),
                                op: ArithOp::Mul,
                                r: _ExprType::Number(3)
                            }))
                        })),
                        _ExprType::Bool(Box::new(_Bool {
                            l: _ExprType::Not(Box::new(_ExprType::BoolLiteral(true))),
                            op: BoolOp::Or,
                            r: _ExprType::Bool(Box::new(_Bool {
                                l: _ExprType::BoolLiteral(true),
                                op: BoolOp::And,
                                r: _ExprType::BoolLiteral(false)
                            }))
                        })),
                        _ExprType::Assign(Box::new(_Assign {
                            lval: _LVal::Field(
                                Box::new(_LVal::Subscript(
                                    Box::new(_LVal::Simple("a".to_owned())),
                                    _ExprType::LVal(Box::new(_LVal::Simple("b".to_owned())))
                                )),
                                "c".to_owned()
                            ),
                            expr: _ExprType::Number(1),
                        })),
                        _ExprType::FnCall(Box::new(_FnCall {
                            id: "g".to_owned(),
                            args: vec![_ExprType::Number(1), _ExprType::Number(2)]
                        })),
                        _ExprType::Record(Box::new(_Record {
                            id: "record".to_owned(),
                            field_assigns: vec![
                                _FieldAssign {
                                    id: "a".to_owned(),
                                    expr: _ExprType::Number(1),
                                },
                                _FieldAssign {
                                    id: "b".to_owned(),
                                    expr: _ExprType::Number(2),
                                }
                            ]
                        })),
                        _ExprType::If(Box::new(_If {
                            cond: _ExprType::Compare(Box::new(_Compare {
                                l: _ExprType::Number(1),
                                op: CompareOp::Eq,
                                r: _ExprType::Seq(vec![_ExprType::Number(1)], true)
                            })),
                            then_expr: _ExprType::Seq(vec![_ExprType::Number(2)], true),
                            else_expr: Some(_ExprType::If(Box::new(_If {
                                cond: _ExprType::Number(3),
                                then_expr: _ExprType::Seq(vec![_ExprType::Number(4)], true),
                                else_expr: Some(_ExprType::Seq(vec![_ExprType::Unit], true)),
                            })))
                        })),
                        _ExprType::If(Box::new(_If {
                            cond: _ExprType::Seq(vec![_ExprType::BoolLiteral(true)], true),
                            then_expr: _ExprType::Seq(vec![_ExprType::Unit], true),
                            else_expr: None,
                        })),
                        _ExprType::Array(Box::new(_Array {
                            initial_value: _ExprType::Number(0),
                            len: _ExprType::Number(3),
                        })),
                        _ExprType::Range(Box::new(_Range {
                            lower: _ExprType::Seq(vec![_ExprType::Number(1)], true),
                            upper: _ExprType::Number(2)
                        })),
                        _ExprType::Closure(Box::new(_Closure {
                            type_fields: vec![],
                            body: _ExprType::Seq(vec![_ExprType::Number(1)], true),
                        })),
                        _ExprType::Let(Box::new(_Let {
                            pattern: Pattern::String("f".to_owned()),
                            immutable: true,
                            ty: Some(_Type::Fn(
                                vec![_Type::Type("int".to_owned())],
                                Box::new(_Type::Type("int".to_owned()))
                            )),
                            expr: _ExprType::Closure(Box::new(_Closure {
                                type_fields: vec![_TypeField {
                                    id: "a".to_owned(),
                                    ty: _Type::Type("int".to_owned())
                                }],
                                body: _ExprType::LVal(Box::new(_LVal::Simple("a".to_owned()))),
                            }))
                        })),
                        _ExprType::FnDecl(Box::new(_FnDecl {
                            id: "f".to_owned(),
                            type_fields: vec![],
                            return_type: None,
                            body: _ExprType::Number(1),
                        })),
                        _ExprType::Compare(Box::new(_Compare {
                            l: _ExprType::Compare(Box::new(_Compare {
                                l: _ExprType::LVal(Box::new(_LVal::Simple("a".to_owned()))),
                                op: CompareOp::Ge,
                                r: _ExprType::LVal(Box::new(_LVal::Simple("b".to_owned()))),
                            })),
                            op: CompareOp::Eq,
                            r: _ExprType::LVal(Box::new(_LVal::Simple("c".to_owned()))),
                        })),
                    ],
                    false
                )
            })]
        );
        Ok(())
    }

    #[test]
    fn parser() -> parser::Result<()> {
        let decls: Vec<_DeclType> =
            parser::parse_program(EMPTY_SOURCEMAP.1, include_str!("../test-euph-files/test1.euph"))?
                .into_iter()
                .map(|decl| decl.t.into())
                .collect();
        assert_eq!(
            decls,
            vec![
                _DeclType::Fn(_FnDecl {
                    id: "f".to_owned(),
                    type_fields: vec![],
                    return_type: None,
                    body: _ExprType::Seq(
                        vec![
                            _ExprType::Let(Box::new(_Let {
                                pattern: Pattern::String("a".to_owned()),
                                immutable: true,
                                ty: Some(_Type::Type("a".to_owned())),
                                expr: _ExprType::Arith(Box::new(_Arith {
                                    l: _ExprType::Number(1),
                                    op: ArithOp::Sub,
                                    r: _ExprType::Neg(Box::new(_ExprType::Number(1)))
                                }))
                            })),
                            _ExprType::Let(Box::new(_Let {
                                pattern: Pattern::String("b".to_owned()),
                                immutable: true,
                                ty: None,
                                expr: _ExprType::String("".to_owned())
                            })),
                            _ExprType::Arith(Box::new(_Arith {
                                l: _ExprType::LVal(Box::new(_LVal::Simple("a".to_owned()))),
                                op: ArithOp::Add,
                                r: _ExprType::LVal(Box::new(_LVal::Simple("b".to_owned())))
                            })),
                        ],
                        true
                    )
                }),
                _DeclType::Fn(_FnDecl {
                    id: "g".to_owned(),
                    type_fields: vec![_TypeField {
                        id: "i".to_owned(),
                        ty: _Type::Type("int".to_owned())
                    }],
                    return_type: Some(_Type::Type("int".to_owned())),
                    body: _ExprType::Arith(Box::new(_Arith {
                        l: _ExprType::Number(1),
                        op: ArithOp::Div,
                        r: _ExprType::Number(2)
                    })),
                }),
            ]
        );
        Ok(())
    }

    #[test]
    fn enum_exprs() -> parser::Result<()> {
        let decls: Vec<_DeclType> = parser::parse_program(
            EMPTY_SOURCEMAP.1,
            include_str!("../test-euph-files/test_enum_exprs.euph"),
        )?
        .into_iter()
        .map(|decl| decl.t.into())
        .collect();
        assert_eq!(
            decls,
            vec![_DeclType::Fn(_FnDecl {
                id: "f".to_owned(),
                type_fields: vec![],
                return_type: None,
                body: _ExprType::Seq(
                    vec![_ExprType::Enum(Box::new(_Enum {
                        enum_id: "A".to_owned(),
                        case_id: "B".to_owned(),
                        args: vec![
                            _ExprType::Number(1),
                            _ExprType::Record(Box::new(_Record {
                                id: "a".to_owned(),
                                field_assigns: vec![_FieldAssign {
                                    id: "a".to_owned(),
                                    expr: _ExprType::Number(1)
                                }]
                            }))
                        ],
                    }))],
                    false
                )
            }),]
        );
        Ok(())
    }

    #[test]
    fn parser_array_negative_index_err() {
        let decls = parser::parse_program(EMPTY_SOURCEMAP.1, "type a = [int; -3]");
        assert!(decls.is_err());
    }
}
