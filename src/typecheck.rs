use crate::ast::{self, Expr, ExprType, LVal, Let, Pattern, Record, Span, Spanned};
use codespan::{ByteIndex, ByteSpan};
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, TypecheckErr>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypecheckErrType {
    /// expected, actual
    TypeMismatch(Type, Type),
    UndefinedVar(String),
    UndefinedField(String),
    CannotSubscript,
    Source(Box<TypecheckErr>),
}

pub type TypecheckErr = Spanned<TypecheckErrType>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    /// first span: id, second span: type
    Record(HashMap<String, Type>),
    Array(Box<Type>, usize),
    Unit,
    Alias(String),
    Enum(Vec<EnumCase>),
    Fn(Vec<Type>, Box<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumCase {
    pub id: String,
    pub params: Vec<EnumParam>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnumParam {
    Simple(Type),
    Record(HashMap<String, Type>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeField {
    pub id: String,
    pub ty: Type,
}

impl From<ast::Type> for Type {
    fn from(ty: ast::Type) -> Self {
        match ty {
            ast::Type::Type(ty) => {
                if ty.t == "int" {
                    Type::Int
                } else if ty.t == "string" {
                    Type::String
                } else {
                    Type::Alias(ty.t)
                }
            }
            ast::Type::Enum(cases) => Type::Enum(cases.into_iter().map(|c| c.t.into()).collect()),
            ast::Type::Record(type_fields) => {
                let mut type_fields_hm = HashMap::new();
                for TypeField { id, ty } in type_fields.into_iter().map(|tf| tf.t.into()) {
                    type_fields_hm.insert(id, ty);
                }
                Type::Record(type_fields_hm)
            }
            ast::Type::Array(ty, len) => Type::Array(Box::new(ty.t.into()), len.t.into()),
        }
    }
}

impl From<ast::TypeField> for TypeField {
    fn from(type_field: ast::TypeField) -> Self {
        Self {
            id: type_field.id.t,
            ty: type_field.ty.t.into(),
        }
    }
}

impl From<ast::EnumParam> for EnumParam {
    fn from(param: ast::EnumParam) -> Self {
        match param {
            ast::EnumParam::Simple(s) => EnumParam::Simple(Type::Alias(s.t)),
            ast::EnumParam::Record(type_fields) => {
                let mut type_fields_hm = HashMap::new();
                for TypeField { id, ty } in type_fields.into_iter().map(|tf| tf.t.into()) {
                    type_fields_hm.insert(id, ty);
                }
                EnumParam::Record(type_fields_hm)
            }
        }
    }
}

impl From<ast::EnumCase> for EnumCase {
    fn from(case: ast::EnumCase) -> Self {
        Self {
            id: case.id.t,
            params: case.params.into_iter().map(|p| p.t.into()).collect(),
        }
    }
}

pub struct Env {
    pub vars: HashMap<String, Type>,
    pub types: HashMap<String, Type>,
    pub var_def_spans: HashMap<String, ByteSpan>,
    pub type_def_spans: HashMap<String, ByteSpan>,
}

impl Default for Env {
    fn default() -> Self {
        let mut types = HashMap::new();
        types.insert("int".to_owned(), Type::Int);
        types.insert("string".to_owned(), Type::String);
        let mut type_def_spans = HashMap::new();
        type_def_spans.insert(
            "int".to_owned(),
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        type_def_spans.insert(
            "string".to_owned(),
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        Env {
            vars: HashMap::new(),
            types,
            var_def_spans: HashMap::new(),
            type_def_spans,
        }
    }
}

impl Env {
    fn insert_var(&mut self, name: String, ty: Type, def_span: ByteSpan) {
        self.vars.insert(name.clone(), ty);
        self.var_def_spans.insert(name, def_span);
    }
}

macro_rules! assert_ty {
    ( $self:ident , $e:expr , $ty:expr ) => {{
        match $self.translate_expr($e) {
            Ok(ref ty) if ty == &$ty => Ok($ty),
            Ok(ref ty) => Err(TypecheckErr {
                t: TypecheckErrType::TypeMismatch($ty, ty.clone()),
                span: $e.span,
            }),
            Err(err) => Err(TypecheckErr {
                t: TypecheckErrType::Source(Box::new(err)),
                span: $e.span,
            }),
        }
    }};
}

impl Env {
    pub fn translate_expr(&self, expr: &Expr) -> Result<Type> {
        match &expr.t {
            ExprType::Seq(exprs, returns) => {
                for expr in &exprs[..exprs.len() - 1] {
                    let _ = self.translate_expr(expr)?;
                }
                if *returns {
                    Ok(self.translate_expr(expr)?)
                } else {
                    let _ = self.translate_expr(expr)?;
                    Ok(Type::Unit)
                }
            }
            ExprType::String(_) => Ok(Type::String),
            ExprType::Number(_) => Ok(Type::Int),
            ExprType::Neg(expr) => assert_ty!(self, expr, Type::Int),

            ExprType::Arith(l, _, r) => {
                assert_ty!(self, l, Type::Int)?;
                assert_ty!(self, r, Type::Int)?;
                Ok(Type::Int)
            }
            ExprType::Unit | ExprType::Continue | ExprType::Break => Ok(Type::Unit),
            ExprType::BoolLiteral(_) => Ok(Type::Bool),
            ExprType::Not(expr) => assert_ty!(self, expr, Type::Bool),
            ExprType::Bool(l, _, r) => {
                assert_ty!(self, l, Type::Bool)?;
                assert_ty!(self, r, Type::Bool)?;
                Ok(Type::Bool)
            }
            ExprType::LVal(lval) => self.translate_lval(lval),
            _ => unimplemented!(),
        }
    }

    pub fn translate_lval(&self, lval: &Spanned<LVal>) -> Result<Type> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(ty) = self.vars.get(var) {
                    Ok(ty.clone())
                } else {
                    Err(TypecheckErr {
                        t: TypecheckErrType::UndefinedVar(var.clone()),
                        span: lval.span,
                    })
                }
            }
            LVal::Field(var, field) => match self.translate_lval(var) {
                Ok(Type::Record(fields)) => {
                    if let Some(field_type) = fields.get(&field.t) {
                        Ok(field_type.clone())
                    } else {
                        Err(TypecheckErr {
                            t: TypecheckErrType::UndefinedField(field.t.clone()),
                            span: field.span,
                        })
                    }
                }
                Ok(_) => Err(TypecheckErr {
                    t: TypecheckErrType::UndefinedField(field.t.clone()),
                    span: field.span,
                }),
                Err(err) => Err(TypecheckErr {
                    t: TypecheckErrType::Source(Box::new(err)),
                    span: field.span,
                }),
            },
            LVal::Subscript(var, index) => match self.translate_lval(var) {
                Ok(Type::Array(ty, _)) => match self.translate_expr(index) {
                    Ok(Type::Int) => Ok((*ty).clone()),
                    Ok(ty) => Err(TypecheckErr {
                        t: TypecheckErrType::TypeMismatch(Type::Int, ty),
                        span: index.span,
                    }),
                    Err(err) => Err(TypecheckErr {
                        t: TypecheckErrType::Source(Box::new(err)),
                        span: index.span,
                    }),
                },
                Ok(_) => Err(TypecheckErr {
                    t: TypecheckErrType::CannotSubscript,
                    span: index.span,
                }),
                Err(err) => Err(TypecheckErr {
                    t: TypecheckErrType::Source(Box::new(err)),
                    span: index.span,
                }),
            },
        }
    }

    pub fn translate_let(&mut self, let_expr: &Spanned<Let>) -> Result<Type> {
        let Let {
            pattern, ty, expr, ..
        } = &let_expr.t;
        let expr_type = self.translate_expr(expr)?;
        if let Some(Spanned { t: ty, .. }) = ty {
            // Type annotation
            let ty = Type::from(ty.clone());
            if ty != expr_type {
                return Err(TypecheckErr::new(
                    TypecheckErrType::TypeMismatch(ty, expr_type),
                    let_expr.span,
                ));
            }
        }
        match &pattern.t {
            Pattern::String(var_name) => {
                self.insert_var(var_name.clone(), expr_type, let_expr.span)
            }
            _ => (),
        }
        Ok(Type::Unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BoolOp;
    use codespan::ByteOffset;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_typecheck_bool_expr() {
        let expr = expr!(ExprType::Bool(
            Box::new(expr!(ExprType::BoolLiteral(true))),
            zspan!(BoolOp::And),
            Box::new(expr!(ExprType::BoolLiteral(true))),
        ));
        let env = Env::default();
        assert_eq!(env.translate_expr(&expr), Ok(Type::Bool));
    }

    #[test]
    fn test_typecheck_bool_expr_source() {
        let expr = expr!(ExprType::Bool(
            Box::new(expr!(ExprType::Bool(
                Box::new(expr!(ExprType::BoolLiteral(true))),
                zspan!(BoolOp::And),
                Box::new(expr!(ExprType::Number(1)))
            ))),
            zspan!(BoolOp::And),
            Box::new(expr!(ExprType::BoolLiteral(true))),
        ));
        let env = Env::default();
        assert_eq!(
            env.translate_expr(&expr),
            Err(TypecheckErr {
                t: TypecheckErrType::Source(Box::new(TypecheckErr {
                    t: TypecheckErrType::TypeMismatch(Type::Bool, Type::Int),
                    span: span!(0, 0, ByteOffset(0))
                })),
                span: span!(0, 0, ByteOffset(0))
            })
        );
    }

    #[test]
    fn test_typecheck_record_field() {
        let mut env = Env::default();
        let record = {
            let mut hm = HashMap::new();
            hm.insert("f".to_owned(), Type::Int);
            hm
        };
        env.insert_var(
            "x".to_owned(),
            Type::Record(record),
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("f".to_owned())
        ));
        assert_eq!(env.translate_lval(&lval), Ok(Type::Int));
    }

    #[test]
    fn test_typecheck_record_field_err1() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            Type::Record(HashMap::new()),
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Err(TypecheckErr {
                t: TypecheckErrType::UndefinedField("g".to_owned()),
                span: span!(0, 0, ByteOffset(0)),
            })
        );
    }

    #[test]
    fn test_typecheck_record_field_err2() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            Type::Int,
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Err(TypecheckErr {
                t: TypecheckErrType::UndefinedField("g".to_owned()),
                span: span!(0, 0, ByteOffset(0))
            })
        );
    }

    #[test]
    fn test_typecheck_array_subscript() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            Type::Array(Box::new(Type::Int), 3),
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Subscript(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            expr!(ExprType::Number(0))
        ));
        assert_eq!(env.translate_lval(&lval), Ok(Type::Int));
    }

    #[test]
    fn test_typecheck_let() {
        let mut env = Env::default();
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            mutable: zspan!(false),
            ty: None,
            expr: expr!(ExprType::Number(0))
        });
        assert!(env.translate_let(&let_expr).is_ok());
        assert_eq!(env.vars["x"], Type::Int);
        assert!(env.var_def_spans.contains_key("x"));
    }
}
