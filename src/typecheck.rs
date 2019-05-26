use crate::ast::{self, Expr, ExprType, LVal, Let, Pattern, Record, Span, Spanned};
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
    Record(HashMap<String, Type>),
    Array(Box<Type>, usize),
    Unit,
    Alias(String),
    Enum(Vec<Spanned<EnumCase>>),
    Fn(Vec<Type>, Box<Type>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumCase {
    pub id: Spanned<String>,
    pub params: Vec<Spanned<EnumParam>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnumParam {
    Simple(Type),
    Record(Vec<Spanned<TypeField>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeField {
    pub id: Spanned<String>,
    pub ty: Spanned<Type>,
}

impl From<ast::Type> for Type {
    fn from(ty: ast::Type) -> Self {
        match ty {
            ast::Type::Type(ty) => {
                if ty == "int" {
                    Type::Int
                } else if ty == "string" {
                    Type::String
                } else {
                    Type::Alias(ty)
                }
            }
            ast::Type::Enum(cases) => {
                Type::Enum(cases.into_iter().map(|c| c.map(EnumCase::from)).collect())
            }
            ast::Type::Array(ty, len) => Type::Array(Box::new(ty.t.into()), len.t),
            _ => unimplemented!(),
        }
    }
}

impl From<ast::TypeField> for TypeField {
    fn from(type_field: ast::TypeField) -> Self {
        Self {
            id: type_field.id,
            ty: type_field.ty.map(Type::from),
        }
    }
}

impl From<ast::EnumParam> for EnumParam {
    fn from(param: ast::EnumParam) -> Self {
        match param {
            ast::EnumParam::Simple(s) => EnumParam::Simple(Type::Alias(s)),
            ast::EnumParam::Record(type_fields) => EnumParam::Record(
                type_fields
                    .into_iter()
                    .map(|tf| tf.map(TypeField::from))
                    .collect(),
            ),
        }
    }
}

impl From<ast::EnumCase> for EnumCase {
    fn from(case: ast::EnumCase) -> Self {
        Self {
            id: case.id,
            params: case
                .params
                .into_iter()
                .map(|p| p.map(EnumParam::from))
                .collect(),
        }
    }
}

pub struct Env {
    pub vars: HashMap<String, Type>,
    pub types: HashMap<String, Type>,
}

impl Default for Env {
    fn default() -> Self {
        let mut types = HashMap::new();
        types.insert("int".to_owned(), Type::Int);
        types.insert("string".to_owned(), Type::String);
        Env {
            vars: HashMap::new(),
            types,
        }
    }
}

impl Env {
    fn insert_var(&mut self, name: String, ty: Type) -> Option<Type> {
        self.vars.insert(name, ty)
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
                    Ok(Type::Int) => Ok(*ty.clone()),
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
            pattern,
            mutable,
            ty,
            expr,
        } = &let_expr.t;
        let expr_type = self.translate_expr(expr)?;
        if let Some(ty) = ty {}
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
        env.insert_var("x".to_owned(), Type::Record(record));
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("f".to_owned())
        ));
        assert_eq!(env.translate_lval(&lval), Ok(Type::Int));
    }

    #[test]
    fn test_typecheck_record_field_err1() {
        let mut env = Env::default();
        env.insert_var("x".to_owned(), Type::Record(HashMap::new()));
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
        env.insert_var("x".to_owned(), Type::Int);
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
        env.insert_var("x".to_owned(), Type::Array(Box::new(Type::Int), 3));
        let lval = zspan!(LVal::Subscript(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            expr!(ExprType::Number(0))
        ));
        assert_eq!(env.translate_lval(&lval), Ok(Type::Int));
    }
}
