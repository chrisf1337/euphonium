use crate::ast::{Expr, ExprType, LVal, Record, Spanned};
use std::collections::HashMap;

pub type Result<'a, T> = std::result::Result<T, TypecheckErr<'a>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypecheckErrType<'a> {
    /// expected, actual
    TypeMismatch(Type, Type),
    UndefinedVar(String),
    UndefinedField(String),
    CannotSubscript,
    Source(Box<TypecheckErr<'a>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypecheckErr<'a> {
    pub expr: &'a Expr,
    pub err_type: TypecheckErrType<'a>,
}

impl<'a> std::fmt::Display for TypecheckErr<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl<'a> TypecheckErr<'a> {
    pub fn new(expr: &'a Expr, err_type: TypecheckErrType<'a>) -> Self {
        TypecheckErr { expr, err_type }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Number,
    String,
    Bool,
    Record(HashMap<String, Type>),
    Array(Box<Type>, usize),
    Unit,
    Alias(Box<Type>),
    Sum(Vec<Type>),
    Fn(Vec<Type>, Box<Type>),
}

pub struct Env {
    pub vars: HashMap<String, Type>,
    pub types: HashMap<String, Type>,
}

impl Default for Env {
    fn default() -> Self {
        let mut types = HashMap::new();
        types.insert("int".to_owned(), Type::Number);
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
            Ok(ref ty) => Err(TypecheckErr::new(
                $e,
                TypecheckErrType::TypeMismatch($ty, ty.clone()),
            )),
            Err(err) => Err(TypecheckErr::new(
                $e,
                TypecheckErrType::Source(Box::new(err)),
            )),
        }
    }};
}

impl Env {
    pub fn translate_expr<'a>(&self, expr: &'a Expr) -> Result<'a, Type> {
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
            ExprType::Number(_) => Ok(Type::Number),
            ExprType::Neg(expr) => assert_ty!(self, expr, Type::Number),

            ExprType::Arith(l, _, r) => {
                assert_ty!(self, l, Type::Number)?;
                assert_ty!(self, r, Type::Number)?;
                Ok(Type::Number)
            }
            ExprType::Unit | ExprType::Continue | ExprType::Break => Ok(Type::Unit),
            ExprType::BoolLiteral(_) => Ok(Type::Bool),
            ExprType::Not(expr) => assert_ty!(self, expr, Type::Bool),
            ExprType::Bool(l, _, r) => {
                assert_ty!(self, l, Type::Bool)?;
                assert_ty!(self, r, Type::Bool)?;
                Ok(Type::Bool)
            }
            ExprType::LVal(lval) => self.translate_lval(expr, lval),
            _ => unimplemented!(),
        }
    }

    pub fn translate_lval<'a>(&self, expr: &'a Expr, lval: &'a Spanned<LVal>) -> Result<'a, Type> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(ty) = self.vars.get(var) {
                    Ok(ty.clone())
                } else {
                    Err(TypecheckErr::new(
                        expr,
                        TypecheckErrType::UndefinedVar(var.clone()),
                    ))
                }
            }
            LVal::Field(var, field) => match self.translate_lval(expr, var) {
                Ok(Type::Record(fields)) => {
                    if let Some(field_type) = fields.get(field) {
                        Ok(field_type.clone())
                    } else {
                        Err(TypecheckErr::new(
                            expr,
                            TypecheckErrType::UndefinedField(field.clone()),
                        ))
                    }
                }
                Ok(_) => Err(TypecheckErr::new(
                    expr,
                    TypecheckErrType::UndefinedField(field.clone()),
                )),
                Err(err) => Err(TypecheckErr::new(
                    expr,
                    TypecheckErrType::Source(Box::new(err)),
                )),
            },
            LVal::Subscript(var, index) => match self.translate_lval(expr, var) {
                Ok(Type::Array(ty, _)) => match self.translate_expr(index) {
                    Ok(Type::Number) => Ok(*ty.clone()),
                    Ok(ty) => Err(TypecheckErr::new(
                        index,
                        TypecheckErrType::TypeMismatch(Type::Number, ty),
                    )),
                    Err(err) => Err(TypecheckErr::new(
                        index,
                        TypecheckErrType::Source(Box::new(err)),
                    )),
                },
                Ok(_) => Err(TypecheckErr::new(expr, TypecheckErrType::CannotSubscript)),
                Err(err) => Err(TypecheckErr::new(
                    index,
                    TypecheckErrType::Source(Box::new(err)),
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::BoolOp;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_typecheck_bool_expr() {
        let expr = expr!(ExprType::Bool(
            Box::new(expr!(ExprType::BoolLiteral(true))),
            BoolOp::And,
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
                BoolOp::And,
                Box::new(expr!(ExprType::Number(1)))
            ))),
            BoolOp::And,
            Box::new(expr!(ExprType::BoolLiteral(true))),
        ));
        let env = Env::default();
        assert_eq!(
            env.translate_expr(&expr),
            Err(TypecheckErr {
                expr: &expr!(ExprType::Bool(
                    Box::new(expr!(ExprType::BoolLiteral(true))),
                    BoolOp::And,
                    Box::new(expr!(ExprType::Number(1)))
                )),
                err_type: TypecheckErrType::Source(Box::new(TypecheckErr {
                    expr: &expr!(ExprType::Number(1)),
                    err_type: TypecheckErrType::TypeMismatch(Type::Bool, Type::Number)
                }))
            })
        );
    }

    #[test]
    fn test_typecheck_record_field() {
        let mut env = Env::default();
        let record = {
            let mut hm = HashMap::new();
            hm.insert("f".to_owned(), Type::Number);
            hm
        };
        env.insert_var("x".to_owned(), Type::Record(record));
        let lval = span!(LVal::Field(
            Box::new(span!(LVal::Simple("x".to_owned()))),
            "f".to_owned()
        ));
        let expr = expr!(ExprType::LVal(Box::new(lval.clone())));
        assert_eq!(env.translate_lval(&expr, &lval), Ok(Type::Number));
    }

    #[test]
    fn test_typecheck_record_field_err1() {
        let mut env = Env::default();
        env.insert_var("x".to_owned(), Type::Record(HashMap::new()));
        let lval = span!(LVal::Field(
            Box::new(span!(LVal::Simple("x".to_owned()))),
            "g".to_owned()
        ));
        let expr = expr!(ExprType::LVal(Box::new(lval.clone())));
        assert_eq!(
            env.translate_lval(&expr, &lval),
            Err(TypecheckErr::new(
                &expr,
                TypecheckErrType::UndefinedField("g".to_owned())
            ))
        );
    }

    #[test]
    fn test_typecheck_record_field_err2() {
        let mut env = Env::default();
        env.insert_var("x".to_owned(), Type::Number);
        let lval = span!(LVal::Field(
            Box::new(span!(LVal::Simple("x".to_owned()))),
            "g".to_owned()
        ));
        let expr = expr!(ExprType::LVal(Box::new(lval.clone())));
        assert_eq!(
            env.translate_lval(&expr, &lval),
            Err(TypecheckErr::new(
                &expr,
                TypecheckErrType::UndefinedField("g".to_owned())
            ))
        );
    }

    #[test]
    fn test_typecheck_array_subscript() {
        let mut env = Env::default();
        env.insert_var("x".to_owned(), Type::Array(Box::new(Type::Number), 3));
        let lval = span!(LVal::Subscript(
            Box::new(span!(LVal::Simple("x".to_owned()))),
            expr!(ExprType::Number(0))
        ));
        let expr = expr!(ExprType::LVal(Box::new(lval.clone())));
        assert_eq!(env.translate_lval(&expr, &lval), Ok(Type::Number));
    }
}
