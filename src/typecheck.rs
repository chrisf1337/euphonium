use crate::ast::{self, Expr, ExprType, FnCall, LVal, Let, Pattern, Record, Spanned};
use codespan::{ByteIndex, ByteSpan};
use std::collections::HashMap;

pub type Result<T> = std::result::Result<T, Vec<TypecheckErr>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypecheckErrType {
    /// expected, actual
    TypeMismatch(Type, Type),
    // The reason we need a separate case for this instead of using TypeMismatch is because we would
    // need to translate the arguments passed to the non-function in order to determine the expected
    // function type.
    ArityMismatch(usize, usize),
    NotAFn,
    UndefinedVar(String),
    UndefinedFn(String),
    UndefinedField(String),
    CannotSubscript,
    Source(Vec<TypecheckErr>),
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

#[derive(Debug, Clone)]
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
            Ok(ref ty) => Err(vec![TypecheckErr {
                t: TypecheckErrType::TypeMismatch($ty, ty.clone()),
                span: $e.span,
            }]),
            Err(errs) => Err(vec![TypecheckErr {
                t: TypecheckErrType::Source(errs),
                span: $e.span,
            }]),
        }
    }};
}

impl Env {
    pub fn resolve<'a>(&'a self, ty: &'a Type) -> Option<&'a Type> {
        match ty {
            Type::Alias(alias) => self.vars.get(alias).map_or(None, |ty| self.resolve(ty)),
            _ => Some(ty),
        }
    }

    pub fn get_var<'a>(&'a self, var: &'a str) -> Option<&'a Type> {
        self.vars
            .get(var)
            .map_or(None, |var_ty| self.resolve(var_ty))
    }

    pub fn get_type<'a>(&'a self, ty: &'a str) -> Option<&'a Type> {
        self.types.get(ty).map_or(None, |ty| self.resolve(ty))
    }

    pub fn translate_expr(&mut self, expr: &Expr) -> Result<Type> {
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
            ExprType::Let(let_expr) => self.translate_let(let_expr),
            // ExprType::FnCall(fn_call)
            _ => unimplemented!(),
        }
    }

    pub fn translate_lval(&mut self, lval: &Spanned<LVal>) -> Result<Type> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(ty) = self.get_var(var) {
                    Ok(ty.clone())
                } else {
                    Err(vec![TypecheckErr {
                        t: TypecheckErrType::UndefinedVar(var.clone()),
                        span: lval.span,
                    }])
                }
            }
            LVal::Field(var, field) => match self.translate_lval(var) {
                Ok(Type::Record(fields)) => {
                    if let Some(field_type) = fields.get(&field.t) {
                        Ok(field_type.clone())
                    } else {
                        Err(vec![TypecheckErr {
                            t: TypecheckErrType::UndefinedField(field.t.clone()),
                            span: field.span,
                        }])
                    }
                }
                Ok(_) => Err(vec![TypecheckErr {
                    t: TypecheckErrType::UndefinedField(field.t.clone()),
                    span: field.span,
                }]),
                Err(errs) => Err(vec![TypecheckErr {
                    t: TypecheckErrType::Source(errs),
                    span: field.span,
                }]),
            },
            LVal::Subscript(var, index) => match self.translate_lval(var) {
                Ok(Type::Array(ty, _)) => match self.translate_expr(index) {
                    Ok(Type::Int) => Ok((*ty).clone()),
                    Ok(ty) => Err(vec![TypecheckErr {
                        t: TypecheckErrType::TypeMismatch(Type::Int, ty),
                        span: index.span,
                    }]),
                    Err(errs) => Err(vec![TypecheckErr {
                        t: TypecheckErrType::Source(errs),
                        span: index.span,
                    }]),
                },
                Ok(_) => Err(vec![TypecheckErr {
                    t: TypecheckErrType::CannotSubscript,
                    span: index.span,
                }]),
                Err(errs) => Err(vec![TypecheckErr {
                    t: TypecheckErrType::Source(errs),
                    span: index.span,
                }]),
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
                return Err(vec![TypecheckErr::new(
                    TypecheckErrType::TypeMismatch(ty, expr_type),
                    let_expr.span,
                )]);
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

    pub fn translate_fn_call(
        &self,
        Spanned {
            t: FnCall { id, args },
            span,
        }: &Spanned<FnCall>,
    ) -> Result<Type> {
        if let Some(fn_type) = self.get_var(&id.t) {
            match fn_type {
                Type::Fn(param_types, return_type) => {
                    if args.len() != param_types.len() {
                        return Err(vec![TypecheckErr::new(
                            TypecheckErrType::ArityMismatch(args.len(), param_types.len()),
                            *span,
                        )]);
                    }

                    for (Spanned { t: arg, span }, param_type) in
                        args.iter().zip(param_types.iter())
                    {}

                    Ok(*return_type.clone())
                }
                _ => Err(vec![TypecheckErr::new(TypecheckErrType::NotAFn, id.span)]),
            }
        } else {
            Err(vec![TypecheckErr::new(
                TypecheckErrType::UndefinedFn(id.t.clone()),
                *span,
            )])
        }
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
        let mut env = Env::default();
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
        let mut env = Env::default();
        assert_eq!(
            env.translate_expr(&expr),
            Err(vec![TypecheckErr {
                t: TypecheckErrType::Source(vec![TypecheckErr {
                    t: TypecheckErrType::TypeMismatch(Type::Bool, Type::Int),
                    span: span!(0, 0, ByteOffset(0))
                }]),
                span: span!(0, 0, ByteOffset(0))
            }])
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
            Err(vec![TypecheckErr {
                t: TypecheckErrType::UndefinedField("g".to_owned()),
                span: span!(0, 0, ByteOffset(0)),
            }])
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
            Err(vec![TypecheckErr {
                t: TypecheckErrType::UndefinedField("g".to_owned()),
                span: span!(0, 0, ByteOffset(0))
            }])
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
    fn test_typecheck_let_type_annotation() {
        let mut env = Env::default();
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            expr: expr!(ExprType::Number(0))
        });
        assert!(env.translate_let(&let_expr).is_ok());
        assert_eq!(env.vars["x"], Type::Int);
        assert!(env.var_def_spans.contains_key("x"));
    }

    #[test]
    fn test_typecheck_let_type_annotation_err() {
        let mut env = Env::default();
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("string".to_owned())))),
            expr: expr!(ExprType::Number(0))
        });
        assert!(dbg!(env.translate_let(&let_expr)).is_err());
    }

    #[test]
    fn test_typecheck_fn_call_undefined_err() {
        let env = Env::default();
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert!(dbg!(env.translate_fn_call(&fn_call_expr)).is_err());
    }

    #[test]
    fn test_typecheck_fn_call_not_fn_err() {
        let mut env = Env::default();
        env.insert_var(
            "f".to_owned(),
            Type::Int,
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert!(dbg!(env.translate_fn_call(&fn_call_expr)).is_err());
    }
}
