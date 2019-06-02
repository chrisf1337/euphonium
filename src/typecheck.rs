use crate::ast::{
    self, Decl, DeclType, Expr, ExprType, FnCall, FnDecl, LVal, Let, Pattern, Record, Spanned,
    TypeDecl,
};
use codespan::{ByteIndex, ByteSpan};
use codespan_reporting::{Diagnostic, Label};
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
    NotAFn(String),
    UndefinedVar(String),
    UndefinedFn(String),
    UndefinedField(String),
    UndefinedType(String),
    CannotSubscript,
    /// Type triggering the cycle, what it was aliased to
    TypeDeclCycle(String, Type),
}

impl TypecheckErrType {
    pub fn diagnostic(&self, env: &Env) -> Diagnostic {
        use TypecheckErrType::*;
        let msg = match self {
            TypeMismatch(expected, actual) => format!(
                "type mismatch\nexpected: {:?}\n  actual: {:?}",
                expected, actual
            ),
            ArityMismatch(expected, actual) => format!(
                "arity mismatch\nexpected: {:?}\n  actual: {:?}",
                expected, actual
            ),
            NotAFn(fun) => {
                let ty = env.get_var_type(fun).unwrap();
                format!("not a function: {} (has type {:?})", fun, ty)
            }
            UndefinedVar(var) => format!("undefined variable: {}", var),
            UndefinedFn(fun) => format!("undefined function: {}", fun),
            UndefinedField(field) => format!("undefined field: {}", field),
            UndefinedType(ty) => format!("undefined type: {:?}", ty),
            CannotSubscript => "cannot subscript".to_owned(),
            TypeDeclCycle(..) => "cycle in type declaration".to_owned(),
        };
        Diagnostic::new_error(&msg)
    }
}

pub type TypecheckErr = Spanned<TypecheckErrType>;

impl TypecheckErr {
    pub fn label(&self) -> Label {
        Label::new_primary(self.span)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    Record(HashMap<String, Type>),
    Array(Box<Type>, usize),
    Unit,
    Alias(String),
    Enum(Vec<EnumCase>),
    Fn(Vec<Type>, Box<Type>),
}

impl Type {
    fn alias(&self) -> Option<&str> {
        if let Type::Alias(alias) = self {
            Some(alias)
        } else {
            None
        }
    }
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
            ast::Type::Unit => Type::Unit,
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

/// `$ty` must have already been resolved.
macro_rules! assert_ty {
    ( $self:ident , $e:expr , $ty:expr ) => {{
        let expr_type = $self.translate_expr($e)?;
        let resolved_ty = $self.resolve($ty).unwrap();
        let resolved_expr_type = $self.resolve_type(&expr_type, $e.span)?;
        if resolved_ty != resolved_expr_type {
            Err(vec![TypecheckErr {
                t: TypecheckErrType::TypeMismatch($ty.clone(), expr_type.clone()),
                span: $e.span,
            }])
        } else {
            Ok(expr_type.clone())
        }
    }};
}

impl Env {
    fn insert_var(&mut self, name: String, ty: Type, def_span: ByteSpan) {
        self.vars.insert(name.clone(), ty);
        self.var_def_spans.insert(name, def_span);
    }

    fn insert_type(&mut self, name: String, ty: Type, def_span: ByteSpan) {
        self.types.insert(name.clone(), ty);
        self.type_def_spans.insert(name, def_span);
    }

    fn resolve<'a>(&'a self, ty: &'a Type) -> Option<&'a Type> {
        match ty {
            Type::Alias(alias) => self.get_type(alias),
            _ => Some(ty),
        }
    }

    fn resolve_ast_type(&self, Spanned { t: ty, span }: &Spanned<ast::Type>) -> Result<Type> {
        let ty = Type::from(ty.clone());
        self.resolve_type(&ty, *span).map(|ty| ty.clone())
    }

    fn resolve_type<'a>(&'a self, ty: &'a Type, def_span: ByteSpan) -> Result<&'a Type> {
        match ty {
            Type::Alias(alias) => {
                if let Some(resolved_type) = self.types.get(alias) {
                    let span = self.type_def_spans[alias];
                    self.resolve_type(resolved_type, span)
                } else {
                    Err(vec![TypecheckErr::new(
                        TypecheckErrType::UndefinedType(alias.clone()),
                        def_span,
                    )])
                }
            }
            _ => Ok(ty),
        }
    }

    fn get_var_type<'a>(&'a self, var: &'a str) -> Option<&'a Type> {
        self.vars
            .get(var)
            .map_or(None, |var_ty| self.resolve(var_ty))
    }

    fn get_type<'a>(&'a self, ty: &'a str) -> Option<&'a Type> {
        self.types.get(ty).map_or(None, |ty| self.resolve(ty))
    }

    fn check_for_type_decl_cycles(&self, ty: &str, path: Vec<&str>) -> Result<()> {
        if let Some(alias) = self.types.get(ty) {
            if let Some(alias_str) = alias.alias() {
                if path.contains(&alias_str) {
                    let span = self.type_def_spans[ty];
                    return Err(vec![TypecheckErr::new(
                        TypecheckErrType::TypeDeclCycle(ty.to_string(), alias.clone()),
                        span,
                    )]);
                }
                let mut path = path;
                path.push(ty);
                self.check_for_type_decl_cycles(alias_str, path)
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }

    /// Call after translating type decls and checking for cycles.
    fn validate_type(&self, ty: &Type, def_span: ByteSpan) -> Result<()> {
        match ty {
            Type::String | Type::Bool | Type::Unit | Type::Int => Ok(()),
            Type::Alias(_) => self.resolve_type(ty, def_span).map(|_| ()),
            Type::Record(record) => {
                let mut errors = vec![];
                for ty in record.values() {
                    match self.validate_type(ty, def_span) {
                        Ok(()) => (),
                        Err(errs) => errors.extend(errs),
                    }
                }
                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors)
                }
            }
            Type::Array(ty, _) => self.validate_type(ty, def_span),
            Type::Enum(cases) => {
                let mut errors = vec![];
                for case in cases {
                    for param in &case.params {
                        match param {
                            EnumParam::Simple(ty) => match self.validate_type(ty, def_span) {
                                Ok(()) => (),
                                Err(errs) => errors.extend(errs),
                            },
                            EnumParam::Record(fields) => {
                                for ty in fields.values() {
                                    match self.validate_type(ty, def_span) {
                                        Ok(()) => (),
                                        Err(errs) => errors.extend(errs),
                                    }
                                }
                            }
                        }
                    }
                }
                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors)
                }
            }
            Type::Fn(param_types, return_type) => {
                let mut errors = vec![];

                for param_type in param_types {
                    match self.validate_type(param_type, def_span) {
                        Ok(()) => (),
                        Err(errs) => errors.extend(errs),
                    }
                }

                match self.validate_type(return_type, def_span) {
                    Ok(()) => (),
                    Err(errs) => errors.extend(errs),
                }

                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors)
                }
            }
        }
    }

    fn check_for_invalid_types(&self) -> Result<()> {
        let mut errors = vec![];
        for (id, ty) in &self.types {
            match self.validate_type(ty, self.type_def_spans[id]) {
                Ok(()) => (),
                Err(errs) => errors.extend(errs),
            }
        }
        if !errors.is_empty() {
            Err(errors)
        } else {
            Ok(())
        }
    }

    pub fn translate_decls(&mut self, decls: &[Decl]) -> Result<()> {
        self.first_pass(&decls)?;
        self.second_pass(&decls)
    }

    fn first_pass(&mut self, decls: &[Decl]) -> Result<()> {
        let mut errors = vec![];
        let mut found_cycle = false;
        for decl in decls {
            match self.translate_decl_first_pass(decl) {
                Ok(()) => (),
                Err(errs) => {
                    for err in &errs {
                        if let TypecheckErr {
                            t: TypecheckErrType::TypeDeclCycle(..),
                            ..
                        } = err
                        {
                            found_cycle = true;
                        }
                    }
                    errors.extend(errs);
                }
            }
            if found_cycle {
                return Err(errors);
            }
        }

        match self.check_for_invalid_types() {
            Ok(()) => (),
            Err(errs) => errors.extend(errs),
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn second_pass(&mut self, decls: &[Decl]) -> Result<()> {
        let mut errors = vec![];
        for Decl { t: decl, .. } in decls {
            if let DeclType::Fn(fn_decl) = decl {
                match self.translate_fn_decl_body(fn_decl) {
                    Ok(()) => (),
                    Err(errs) => errors.extend(errs),
                }
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    pub fn translate_decl_first_pass(&mut self, decl: &Decl) -> Result<()> {
        match &decl.t {
            DeclType::Fn(fn_decl) => self.translate_fn_decl_sig(fn_decl),
            DeclType::Type(type_decl) => {
                self.translate_type_decl(type_decl)?;
                self.check_for_type_decl_cycles(&type_decl.id, vec![])
            }
            _ => unreachable!(),
        }
    }

    fn translate_fn_decl_sig(&mut self, fn_decl: &Spanned<FnDecl>) -> Result<()> {
        let mut param_types = vec![];
        for Spanned {
            t:
                ast::TypeField {
                    ty: Spanned { t: ty, .. },
                    ..
                },
            ..
        } in &fn_decl.type_fields
        {
            param_types.push(ty.clone().into());
        }
        let return_type = if let Some(Spanned { t: return_type, .. }) = &fn_decl.return_type {
            return_type.clone().into()
        } else {
            Type::Unit
        };
        self.vars.insert(
            fn_decl.id.t.clone(),
            Type::Fn(param_types, Box::new(return_type)),
        );
        self.var_def_spans
            .insert(fn_decl.id.t.clone(), fn_decl.span);
        Ok(())
    }

    fn translate_fn_decl_body(&mut self, fn_decl: &Spanned<FnDecl>) -> Result<()> {
        let mut new_env = self.clone();
        let mut all_errs = vec![];
        let mut param_types = vec![];
        for type_field in &fn_decl.type_fields {
            match self.resolve_ast_type(&type_field.ty) {
                Ok(ty) => {
                    new_env.insert_var(type_field.id.t.clone(), ty, type_field.span);
                    param_types.push(type_field.ty.t.clone().into());
                }
                Err(errs) => all_errs.extend(errs),
            }
        }
        let return_type = if let Some(ty) = &fn_decl.return_type {
            ty.t.clone().into()
        } else {
            Type::Unit
        };
        let body_type = new_env.translate_expr(&fn_decl.body)?;
        if self.resolve_type(&body_type, fn_decl.body.span)?
            != self.resolve_type(
                &return_type,
                fn_decl
                    .return_type
                    .as_ref()
                    .map_or_else(|| fn_decl.span, |ret_ty| ret_ty.span),
            )?
        {
            return Err(vec![TypecheckErr::new(
                TypecheckErrType::TypeMismatch(return_type, body_type),
                fn_decl.span,
            )]);
        }

        self.insert_var(
            fn_decl.id.t.clone(),
            Type::Fn(param_types, Box::new(return_type)),
            fn_decl.span,
        );

        Ok(())
    }

    fn translate_type_decl(&mut self, decl: &Spanned<TypeDecl>) -> Result<()> {
        self.insert_type(decl.id.t.clone(), decl.ty.t.clone().into(), decl.span);
        Ok(())
    }

    fn translate_expr(&mut self, expr: &Expr) -> Result<Type> {
        match &expr.t {
            ExprType::Seq(exprs, returns) => {
                for expr in &exprs[..exprs.len() - 1] {
                    let _ = self.translate_expr(expr)?;
                }
                let last_expr = exprs.last().unwrap();
                if *returns {
                    Ok(self.translate_expr(last_expr)?)
                } else {
                    let _ = self.translate_expr(last_expr)?;
                    Ok(Type::Unit)
                }
            }
            ExprType::String(_) => Ok(Type::String),
            ExprType::Number(_) => Ok(Type::Int),
            ExprType::Neg(expr) => assert_ty!(self, expr, &Type::Int),
            ExprType::Arith(l, _, r) => {
                assert_ty!(self, l, &Type::Int)?;
                assert_ty!(self, r, &Type::Int)?;
                Ok(Type::Int)
            }
            ExprType::Unit | ExprType::Continue | ExprType::Break => Ok(Type::Unit),
            ExprType::BoolLiteral(_) => Ok(Type::Bool),
            ExprType::Not(expr) => assert_ty!(self, expr, &Type::Bool),
            ExprType::Bool(l, _, r) => {
                assert_ty!(self, l, &Type::Bool)?;
                assert_ty!(self, r, &Type::Bool)?;
                Ok(Type::Bool)
            }
            ExprType::LVal(lval) => self.translate_lval(lval),
            ExprType::Let(let_expr) => self.translate_let(let_expr),
            ExprType::FnCall(fn_call) => self.translate_fn_call(fn_call),
            _ => unimplemented!(),
        }
    }

    fn translate_lval(&mut self, lval: &Spanned<LVal>) -> Result<Type> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(ty) = self.vars.get(var) {
                    Ok(ty.clone())
                } else {
                    Err(vec![TypecheckErr {
                        t: TypecheckErrType::UndefinedVar(var.clone()),
                        span: lval.span,
                    }])
                }
            }
            LVal::Field(var, field) => {
                let var_type = self.translate_lval(var)?;
                if let Some(Type::Record(fields)) = self.resolve(&var_type) {
                    if let Some(field_type) = fields.get(&field.t) {
                        Ok(field_type.clone())
                    } else {
                        Err(vec![TypecheckErr {
                            t: TypecheckErrType::UndefinedField(field.t.clone()),
                            span: field.span,
                        }])
                    }
                } else {
                    Err(vec![TypecheckErr {
                        t: TypecheckErrType::UndefinedField(field.t.clone()),
                        span: field.span,
                    }])
                }
            }
            LVal::Subscript(var, index) => {
                let var_type = self.translate_lval(var)?;
                let index_type = self.translate_expr(index)?;
                if let Some(Type::Array(ty, _)) = self.resolve(&var_type) {
                    if let Some(Type::Int) = self.resolve(&index_type) {
                        Ok(*(ty.clone()))
                    } else {
                        Err(vec![TypecheckErr::new(
                            TypecheckErrType::TypeMismatch(Type::Int, index_type),
                            index.span,
                        )])
                    }
                } else {
                    Err(vec![TypecheckErr::new(
                        TypecheckErrType::CannotSubscript,
                        index.span,
                    )])
                }
            }
        }
    }

    fn translate_let(&mut self, let_expr: &Spanned<Let>) -> Result<Type> {
        let Let {
            pattern, ty, expr, ..
        } = &let_expr.t;
        let expr_type = self.translate_expr(expr)?;
        if let Some(ty) = ty {
            // Type annotation
            let resolved_ty = self.resolve_ast_type(ty)?;
            let resolved_expr_ty = self.resolve_type(&expr_type, expr.span)?;
            if &resolved_ty != resolved_expr_ty {
                return Err(vec![TypecheckErr::new(
                    TypecheckErrType::TypeMismatch(ty.t.clone().into(), expr_type),
                    let_expr.span,
                )]);
            }
        }
        match &pattern.t {
            Pattern::String(var_name) => {
                // Prefer the annotated type if provided
                let ty = if let Some(ty) = ty {
                    ty.t.clone().into()
                } else {
                    expr_type
                };
                self.insert_var(var_name.clone(), ty, let_expr.span)
            }
            _ => (),
        }
        Ok(Type::Unit)
    }

    fn translate_fn_call(
        &mut self,
        Spanned {
            t: FnCall { id, args },
            span,
        }: &Spanned<FnCall>,
    ) -> Result<Type> {
        let fn_type = self.get_var_type(&id.t).map(|ty| ty.clone());
        if let Some(fn_type) = fn_type {
            match fn_type {
                Type::Fn(param_types, return_type) => {
                    if args.len() != param_types.len() {
                        return Err(vec![TypecheckErr::new(
                            TypecheckErrType::ArityMismatch(param_types.len(), args.len()),
                            *span,
                        )]);
                    }

                    let mut errs = vec![];
                    for (arg, param_type) in args.iter().zip(param_types.iter()) {
                        match self.translate_expr(arg) {
                            Ok(ref ty) => {
                                if self.resolve(ty) != self.resolve(param_type) {
                                    errs.push(TypecheckErr::new(
                                        TypecheckErrType::TypeMismatch(
                                            param_type.clone(),
                                            ty.clone(),
                                        ),
                                        arg.span,
                                    ));
                                }
                            }
                            Err(src_errs) => errs.extend(src_errs),
                        }
                    }

                    Ok(*return_type.clone())
                }
                _ => Err(vec![TypecheckErr::new(
                    TypecheckErrType::NotAFn(id.t.clone()),
                    id.span,
                )]),
            }
        } else {
            Err(vec![TypecheckErr::new(
                TypecheckErrType::UndefinedFn(id.t.clone()),
                *span,
            )])
        }
    }

    fn translate_record(
        &self,
        Spanned {
            t: Record { id, field_assigns },
            span,
        }: &Spanned<Record>,
    ) -> Result<Type> {
        if let Some(record_type) = self.types.get(&id.t) {
            for field_assign in field_assigns {}
            Ok(Type::Unit)
        } else {
            Err(vec![TypecheckErr::new(
                TypecheckErrType::UndefinedType(id.t.clone()),
                id.span,
            )])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{self, BoolOp, TypeField};
    use codespan::ByteOffset;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_resolve() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("b".to_owned()), zspan!());
        env.insert_type("b".to_owned(), Type::Alias("c".to_owned()), zspan!());
        env.insert_type("c".to_owned(), Type::Int, zspan!());
        env.insert_type("d".to_owned(), Type::Alias("e".to_owned()), zspan!());

        assert_eq!(env.resolve(&Type::Alias("a".to_owned())), Some(&Type::Int));
        assert_eq!(env.resolve(&Type::Int), Some(&Type::Int));
        assert_eq!(env.resolve(&Type::Alias("d".to_owned())), None);
    }

    #[test]
    fn test_resolve_type() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("b".to_owned()), zspan!());
        env.insert_type("b".to_owned(), Type::Alias("c".to_owned()), zspan!());
        env.insert_type("c".to_owned(), Type::Int, zspan!());
        env.insert_type("d".to_owned(), Type::Alias("e".to_owned()), zspan!());

        assert_eq!(
            env.resolve_ast_type(&zspan!(ast::Type::Type(zspan!("a".to_owned())))),
            Ok(Type::Int)
        );
        assert_eq!(
            env.resolve_ast_type(&zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            Ok(Type::Int)
        );
        assert_eq!(
            env.resolve_ast_type(&zspan!(ast::Type::Type(zspan!("d".to_owned())))),
            Err(vec![TypecheckErr::new(
                TypecheckErrType::UndefinedType("e".to_owned()),
                zspan!()
            )])
        );
    }

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
                t: TypecheckErrType::TypeMismatch(Type::Bool, Type::Int),
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
        assert_eq!(env.translate_let(&let_expr), Ok(Type::Unit));
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
        assert_eq!(
            env.translate_let(&let_expr).unwrap_err()[0].t,
            TypecheckErrType::TypeMismatch(Type::String, Type::Int)
        );
    }

    #[test]
    fn test_typecheck_fn_call_undefined_err() {
        let mut env = Env::default();
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert_eq!(
            env.translate_fn_call(&fn_call_expr).unwrap_err()[0].t,
            TypecheckErrType::UndefinedFn("f".to_owned())
        );
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
        assert_eq!(
            env.translate_fn_call(&fn_call_expr).unwrap_err()[0].t,
            TypecheckErrType::NotAFn("f".to_owned())
        );
    }

    #[test]
    fn test_typecheck_fn_call_arity_mismatch() {
        let mut env = Env::default();
        env.insert_var(
            "f".to_owned(),
            Type::Fn(vec![], Box::new(Type::Int)),
            zspan!(),
        );
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![expr!(ExprType::Number(0))],
        });
        assert_eq!(
            env.translate_fn_call(&fn_call_expr).unwrap_err()[0].t,
            TypecheckErrType::ArityMismatch(0, 1)
        );
    }

    #[test]
    fn test_typecheck_fn_call_arg_type_mismatch() {
        let mut env = Env::default();
        env.insert_var(
            "f".to_owned(),
            Type::Fn(vec![], Box::new(Type::Int)),
            zspan!(),
        );
    }

    #[test]
    fn test_typecheck_fn_call_returns_aliased_type() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("int".to_owned()), zspan!());
        env.insert_var(
            "f".to_owned(),
            Type::Fn(vec![], Box::new(Type::Alias("a".to_owned()))),
            zspan!(),
        );
        let fn_call = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![]
        });
        assert_eq!(
            env.translate_fn_call(&fn_call).unwrap(),
            Type::Alias("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_typedef() {
        let mut env = Env::default();
        let type_decl = zspan!(TypeDecl {
            id: zspan!("a".to_owned()),
            ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
        });
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::Wildcard),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("a".to_owned())))),
            expr: expr!(ExprType::Number(0)),
        });
        let _ = env.translate_type_decl(&type_decl);
        assert!(env.translate_let(&let_expr).is_ok());
    }

    #[test]
    fn test_typecheck_typedef_err() {
        let mut env = Env::default();
        let type_decl = zspan!(TypeDecl {
            id: zspan!("a".to_owned()),
            ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
        });
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::Wildcard),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("a".to_owned())))),
            expr: expr!(ExprType::String("".to_owned())),
        });
        let _ = env.translate_type_decl(&type_decl);
        assert_eq!(
            env.translate_let(&let_expr).unwrap_err()[0].t,
            TypecheckErrType::TypeMismatch(Type::Alias("a".to_owned()), Type::String)
        );
    }

    #[test]
    fn test_typecheck_expr_typedef() {
        let mut env = Env::default();
        let type_decl = zspan!(TypeDecl {
            id: zspan!("i".to_owned()),
            ty: zspan!(ast::Type::Type(zspan!("int".to_owned())))
        });
        let var_def = zspan!(Let {
            pattern: zspan!(Pattern::String("i".to_owned())),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("i".to_owned())))),
            expr: expr!(ExprType::Number(0))
        });
        let expr = expr!(ExprType::LVal(Box::new(zspan!(LVal::Simple(
            "i".to_owned()
        )))));
        env.translate_type_decl(&type_decl)
            .expect("translate type decl");
        env.translate_let(&var_def).expect("translate var def");
        assert_eq!(
            env.translate_expr(&expr).unwrap(),
            Type::Alias("i".to_owned())
        );
    }

    #[test]
    fn test_recursive_typedef() {
        let mut env = Env::default();
        let type_decl = zspan!(DeclType::Type(zspan!(TypeDecl {
            id: zspan!("i".to_owned()),
            ty: zspan!(ast::Type::Record(vec![zspan!(TypeField {
                id: zspan!("i".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("i".to_owned())))
            })]))
        })));
        env.translate_decl_first_pass(&type_decl)
            .expect("translate decl");
    }

    #[test]
    fn test_check_for_type_decl_cycles_err1() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("a".to_owned()), zspan!());
        assert_eq!(
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0].t,
            TypecheckErrType::TypeDeclCycle("a".to_owned(), Type::Alias("a".to_owned()))
        );
    }

    #[test]
    fn test_check_for_type_decl_cycles_err2() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("b".to_owned()), zspan!());
        env.insert_type("b".to_owned(), Type::Alias("c".to_owned()), zspan!());
        env.insert_type("c".to_owned(), Type::Alias("a".to_owned()), zspan!());
        assert_eq!(
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0].t,
            TypecheckErrType::TypeDeclCycle("c".to_owned(), Type::Alias("a".to_owned()))
        );
    }

    #[test]
    fn test_check_for_type_decl_cycles() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("b".to_owned()), zspan!());
        env.insert_type("b".to_owned(), Type::Alias("c".to_owned()), zspan!());
        assert_eq!(env.check_for_type_decl_cycles("a", vec![]), Ok(()));
    }

    #[test]
    fn test_translate_first_pass() {
        let mut env = Env::default();
        let result = env.first_pass(&vec![
            Decl::new(
                DeclType::Type(zspan!(TypeDecl {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("a".to_owned())))
                })),
                zspan!(),
            ),
            Decl::new(
                DeclType::Fn(zspan!(FnDecl {
                    id: zspan!("f".to_owned()),
                    type_fields: vec![],
                    return_type: None,
                    body: expr!(ExprType::Unit)
                })),
                zspan!(),
            ),
        ]);
        assert_eq!(
            result,
            Err(vec![TypecheckErr::new(
                TypecheckErrType::TypeDeclCycle("a".to_owned(), Type::Alias("a".to_owned())),
                zspan!()
            )])
        );
    }

    #[test]
    fn test_validate_type() {
        let mut env = Env::default();
        let mut record_fields = HashMap::new();
        record_fields.insert("f".to_owned(), Type::Alias("a".to_owned()));
        record_fields.insert("g".to_owned(), Type::Alias("b".to_owned()));
        env.insert_type("a".to_owned(), Type::Record(record_fields), zspan!());

        let errs = env.check_for_invalid_types().unwrap_err();
        assert_eq!(errs.len(), 1);
        // Recursive type def in records is allowed
        assert_eq!(errs[0].t, TypecheckErrType::UndefinedType("b".to_owned()));
    }
}
