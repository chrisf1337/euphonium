use crate::ast::{
    self, Assign, Decl, DeclType, Expr, ExprType, FieldAssign, FnCall, FnDecl, LVal, Let, Pattern,
    Record, Spanned, TypeDecl,
};
use codespan::{ByteIndex, ByteSpan};
use codespan_reporting::{Diagnostic, Label};
use std::{
    collections::{HashMap, HashSet},
    rc::Rc,
};

pub type Result<T> = std::result::Result<T, Vec<TypecheckErr>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypecheckErrType {
    /// expected, actual
    TypeMismatch(Rc<Type>, Rc<Type>),
    // The reason we need a separate case for this instead of using TypeMismatch is because we would
    // need to translate the arguments passed to the non-function in order to determine the expected
    // function type.
    ArityMismatch(usize, usize),
    NotAFn(String),
    NotARecord(String),
    UndefinedVar(String),
    UndefinedFn(String),
    UndefinedField(String),
    UndefinedType(String),
    CannotSubscript,
    /// Type triggering the cycle, what it was aliased to
    TypeDeclCycle(String, Rc<Type>),
    MissingFields(Vec<String>),
    InvalidFields(Vec<String>),
    IllegalLetExpr,
    DuplicateFn(String),
    DuplicateType(String),
    DuplicateField(String),
    MutatingImmutable(String),
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
                let ty = env.vars.get(fun).unwrap();
                format!("not a function: {} (has type {:?})", fun, ty)
            }
            NotARecord(ty) => {
                let actual_ty = env.types.get(ty).unwrap();
                format!("not a record: {} (has type {:?})", ty, actual_ty)
            }
            UndefinedVar(var) => format!("undefined variable: {}", var),
            UndefinedFn(fun) => format!("undefined function: {}", fun),
            UndefinedField(field) => format!("undefined field: {}", field),
            UndefinedType(ty) => format!("undefined type: {:?}", ty),
            CannotSubscript => "cannot subscript".to_owned(),
            TypeDeclCycle(..) => "cycle in type declaration".to_owned(),
            MissingFields(fields) => format!("missing fields: {}", fields.join(", ")),
            InvalidFields(fields) => format!("invalid fields: {}", fields.join(", ")),
            IllegalLetExpr => "a let expression cannot be used here".to_owned(),
            DuplicateFn(fun) => format!("duplicate function declaration for {}", fun),
            DuplicateType(ty) => format!("duplicate type declaration for {}", ty),
            DuplicateField(field) => format!("duplicate record field declaration for {}", field),
            MutatingImmutable(var) => format!("{} was declared as immutable", var),
        };
        Diagnostic::new_error(&msg)
    }
}

pub type TypecheckErr = Spanned<_TypecheckErr>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypecheckErr {
    pub ty: TypecheckErrType,
    pub source: Option<Spanned<String>>,
}

impl TypecheckErr {
    pub fn new_err(ty: TypecheckErrType, span: ByteSpan) -> Self {
        TypecheckErr::new(_TypecheckErr { ty, source: None }, span)
    }

    fn with_source(mut self, source: Spanned<String>) -> Self {
        self.source = Some(source);
        self
    }

    pub fn labels(&self) -> Vec<Label> {
        let mut labels = vec![Label::new_primary(self.span)];
        if let Some(source) = &self.t.source {
            labels.push(Label::new_secondary(source.span).with_message(&source.t));
        }
        labels
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    Record(HashMap<String, Rc<Type>>),
    Array(Rc<Type>, usize),
    Unit,
    Alias(String),
    Enum(Vec<EnumCase>),
    Fn(Vec<Rc<Type>>, Rc<Type>),
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
    Simple(Rc<Type>),
    Record(HashMap<String, Rc<Type>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeField {
    pub id: String,
    pub ty: Rc<Type>,
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
            ast::Type::Array(ty, len) => Type::Array(Rc::new(ty.t.into()), len.t.into()),
            ast::Type::Unit => Type::Unit,
        }
    }
}

impl From<ast::TypeField> for TypeField {
    fn from(type_field: ast::TypeField) -> Self {
        Self {
            id: type_field.id.t,
            ty: Rc::new(type_field.ty.t.into()),
        }
    }
}

impl From<ast::EnumParam> for EnumParam {
    fn from(param: ast::EnumParam) -> Self {
        match param {
            ast::EnumParam::Simple(s) => EnumParam::Simple(Rc::new(Type::Alias(s.t))),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarProperties {
    pub ty: Rc<Type>,
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct Env {
    pub vars: HashMap<String, VarProperties>,
    pub types: HashMap<String, Rc<Type>>,
    pub var_def_spans: HashMap<String, ByteSpan>,
    pub type_def_spans: HashMap<String, ByteSpan>,

    pub record_field_decl_spans: HashMap<String, HashMap<String, ByteSpan>>,
}

impl Default for Env {
    fn default() -> Self {
        let mut types = HashMap::new();
        types.insert("int".to_owned(), Rc::new(Type::Int));
        types.insert("string".to_owned(), Rc::new(Type::String));
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

            record_field_decl_spans: HashMap::new(),
        }
    }
}

/// `$ty` must have already been resolved.
macro_rules! assert_ty {
    ( $self:ident , $e:expr , $ty:expr ) => {{
        let expr_type = $self.translate_expr($e)?;
        let resolved_expr_type = $self.resolve_type(&expr_type, $e.span)?;
        if $ty != resolved_expr_type {
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeMismatch($ty.clone(), expr_type.clone()),
                $e.span,
            )])
        } else {
            Ok(expr_type.clone())
        }
    }};
}

impl Env {
    fn insert_var(&mut self, name: String, var_props: VarProperties, def_span: ByteSpan) {
        self.vars.insert(name.clone(), var_props);
        self.var_def_spans.insert(name, def_span);
    }

    fn insert_type(&mut self, name: String, ty: Type, def_span: ByteSpan) {
        self.types.insert(name.clone(), Rc::new(ty));
        self.type_def_spans.insert(name, def_span);
    }

    fn resolve_type(&self, ty: &Rc<Type>, def_span: ByteSpan) -> Result<Rc<Type>> {
        match ty.as_ref() {
            Type::Alias(alias) => {
                if let Some(resolved_type) = self.types.get(alias) {
                    let span = self.type_def_spans[alias];
                    self.resolve_type(resolved_type, span)
                } else {
                    Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::UndefinedType(alias.clone()),
                        def_span,
                    )])
                }
            }
            _ => Ok(ty.clone()),
        }
    }

    fn check_for_type_decl_cycles(&self, ty: &str, path: Vec<&str>) -> Result<()> {
        if let Some(alias) = self.types.get(ty) {
            if let Some(alias_str) = alias.alias() {
                if path.contains(&alias_str) {
                    let span = self.type_def_spans[ty];
                    return Err(vec![TypecheckErr::new_err(
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
    fn validate_type(&self, ty: &Rc<Type>, def_span: ByteSpan) -> Result<()> {
        match ty.as_ref() {
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
        for (id, var) in &self.vars {
            if let Type::Fn(param_types, return_type) = var.ty.as_ref() {
                for ty in param_types {
                    match self.validate_type(ty, self.var_def_spans[id]) {
                        Ok(()) => (),
                        Err(errs) => errors.extend(errs),
                    }
                }
                match self.validate_type(return_type, self.var_def_spans[id]) {
                    Ok(()) => (),
                    Err(errs) => errors.extend(errs),
                }
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
                        if let _TypecheckErr {
                            ty: TypecheckErrType::TypeDeclCycle(..),
                            ..
                        } = err.t
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

    fn second_pass(&self, decls: &[Decl]) -> Result<()> {
        let mut errors = vec![];
        for Decl { t: decl, .. } in decls {
            if let DeclType::Fn(fn_decl) = decl {
                let mut new_env = self.clone();
                match new_env.translate_fn_decl_body(fn_decl) {
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
            param_types.push(Rc::new(ty.clone().into()));
        }
        let return_type = if let Some(Spanned { t: return_type, .. }) = &fn_decl.return_type {
            return_type.clone().into()
        } else {
            Type::Unit
        };

        // Check if there already exists another function with the same name
        if self.vars.contains_key(&fn_decl.id.t) {
            let span = self.var_def_spans[&fn_decl.id.t];
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::DuplicateFn(fn_decl.id.t.clone()),
                fn_decl.id.span,
            )
            .with_source(Spanned::new(
                format!("{} was defined here", fn_decl.id.t.clone()),
                span,
            ))])
        } else {
            self.vars.insert(
                fn_decl.id.t.clone(),
                VarProperties {
                    ty: Rc::new(Type::Fn(param_types, Rc::new(return_type))),
                    mutable: false,
                },
            );
            self.var_def_spans
                .insert(fn_decl.id.t.clone(), fn_decl.span);
            Ok(())
        }
    }

    fn translate_fn_decl_body(&mut self, fn_decl: &Spanned<FnDecl>) -> Result<()> {
        let mut new_env = self.clone();
        let mut all_errs = vec![];
        let mut param_types = vec![];
        for type_field in &fn_decl.type_fields {
            let param_type = Rc::new(type_field.ty.t.clone().into());
            match self.resolve_type(&param_type, type_field.ty.span) {
                Ok(ty) => {
                    new_env.insert_var(
                        type_field.id.t.clone(),
                        VarProperties {
                            ty: ty.clone(),
                            mutable: true,
                        },
                        type_field.span,
                    );
                    param_types.push(param_type);
                }
                Err(errs) => all_errs.extend(errs),
            }
        }
        let return_type = if let Some(ty) = &fn_decl.return_type {
            Rc::new(ty.t.clone().into())
        } else {
            Rc::new(Type::Unit)
        };
        let body_type = new_env.translate_expr_mut(&fn_decl.body)?;
        if self.resolve_type(&body_type, fn_decl.body.span)?
            != self.resolve_type(
                &return_type,
                fn_decl
                    .return_type
                    .as_ref()
                    .map_or_else(|| fn_decl.span, |ret_ty| ret_ty.span),
            )?
        {
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeMismatch(return_type.clone(), body_type),
                fn_decl.span,
            )]);
        }

        self.insert_var(
            fn_decl.id.t.clone(),
            VarProperties {
                ty: Rc::new(Type::Fn(param_types, return_type)),
                mutable: false,
            },
            fn_decl.span,
        );

        Ok(())
    }

    fn translate_type_decl(&mut self, decl: &Spanned<TypeDecl>) -> Result<()> {
        let id = decl.id.t.clone();

        if self.types.contains_key(&id) {
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::DuplicateType(id.clone()),
                decl.span,
            )
            .with_source(Spanned::new(
                format!("{} was defined here", id.clone()),
                self.type_def_spans[&id],
            ))]);
        }

        let ty = decl.ty.t.clone().into();
        if let ast::Type::Record(ref record_fields) = decl.ty.t {
            let mut field_def_spans = HashMap::new();
            let mut errors = vec![];
            for Spanned {
                t: type_field,
                span,
            } in record_fields
            {
                let type_field_id = type_field.id.t.clone();
                if let Some(field_def_span) = field_def_spans.get(&type_field_id) {
                    errors.push(
                        TypecheckErr::new_err(
                            TypecheckErrType::DuplicateField(type_field_id.clone()),
                            *span,
                        )
                        .with_source(Spanned::new(
                            format!("{} was declared here", type_field_id),
                            *field_def_span,
                        )),
                    );
                    continue;
                }
                field_def_spans.insert(type_field_id, *span);
            }

            if !errors.is_empty() {
                return Err(errors);
            }

            self.record_field_decl_spans
                .insert(id.clone(), field_def_spans);
        }
        self.insert_type(id, ty, decl.span);

        Ok(())
    }

    fn translate_expr(&self, expr: &Expr) -> Result<Rc<Type>> {
        match &expr.t {
            ExprType::Seq(exprs, returns) => {
                // New scope
                let mut new_env = self.clone();
                for expr in &exprs[..exprs.len() - 1] {
                    let _ = new_env.translate_expr_mut(expr)?;
                }
                let last_expr = exprs.last().unwrap();
                if *returns {
                    Ok(new_env.translate_expr_mut(last_expr)?)
                } else {
                    let _ = new_env.translate_expr_mut(last_expr)?;
                    Ok(Rc::new(Type::Unit))
                }
            }
            ExprType::String(_) => Ok(Rc::new(Type::String)),
            ExprType::Number(_) => Ok(Rc::new(Type::Int)),
            ExprType::Neg(expr) => assert_ty!(self, expr, Rc::new(Type::Int)),
            ExprType::Arith(l, _, r) => {
                assert_ty!(self, l, Rc::new(Type::Int))?;
                assert_ty!(self, r, Rc::new(Type::Int))?;
                Ok(Rc::new(Type::Int))
            }
            ExprType::Unit | ExprType::Continue | ExprType::Break => Ok(Rc::new(Type::Unit)),
            ExprType::BoolLiteral(_) => Ok(Rc::new(Type::Bool)),
            ExprType::Not(expr) => assert_ty!(self, expr, Rc::new(Type::Bool)),
            ExprType::Bool(l, _, r) => {
                assert_ty!(self, l, Rc::new(Type::Bool))?;
                assert_ty!(self, r, Rc::new(Type::Bool))?;
                Ok(Rc::new(Type::Bool))
            }
            ExprType::LVal(lval) => Ok(self.translate_lval(lval)?.ty),
            ExprType::Let(_) => Err(vec![TypecheckErr::new_err(
                TypecheckErrType::IllegalLetExpr,
                expr.span,
            )]),
            ExprType::FnCall(fn_call) => self.translate_fn_call(fn_call),
            ExprType::Record(record) => self.translate_record(record),
            ExprType::Assign(assign) => self.translate_assign(assign),
            _ => unimplemented!(),
        }
    }

    fn translate_expr_mut(&mut self, expr: &Expr) -> Result<Rc<Type>> {
        match &expr.t {
            ExprType::Let(let_expr) => self.translate_let(let_expr),
            _ => self.translate_expr(expr),
        }
    }

    fn translate_lval(&self, lval: &Spanned<LVal>) -> Result<VarProperties> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(var_properties) = self.vars.get(var) {
                    Ok(var_properties.clone())
                } else {
                    Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::UndefinedVar(var.clone()),
                        lval.span,
                    )])
                }
            }
            LVal::Field(var, field) => {
                let var_properties = self.translate_lval(var)?;
                if let Type::Record(fields) =
                    self.resolve_type(&var_properties.ty, var.span)?.as_ref()
                {
                    if let Some(field_type) = fields.get(&field.t) {
                        // A field is mutable only if the record it belongs to is mutable
                        return Ok(VarProperties {
                            ty: field_type.clone(),
                            mutable: var_properties.mutable,
                        });
                    }
                }
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::UndefinedField(field.t.clone()),
                    field.span,
                )])
            }
            LVal::Subscript(var, index) => {
                let var_properties = self.translate_lval(var)?;
                let index_type = self.translate_expr(index)?;
                if let Type::Array(ty, _) =
                    self.resolve_type(&var_properties.ty, var.span)?.as_ref()
                {
                    if self.resolve_type(&index_type, index.span)? == Rc::new(Type::Int) {
                        Ok(VarProperties {
                            ty: ty.clone(),
                            mutable: var_properties.mutable,
                        })
                    } else {
                        Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), index_type),
                            index.span,
                        )])
                    }
                } else {
                    Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::CannotSubscript,
                        index.span,
                    )])
                }
            }
        }
    }

    fn translate_let(&mut self, let_expr: &Spanned<Let>) -> Result<Rc<Type>> {
        let Let {
            pattern,
            ty: ast_ty,
            mutable,
            expr,
        } = &let_expr.t;
        let expr_type = self.translate_expr_mut(expr)?;
        if let Some(ast_ty) = ast_ty {
            // Type annotation
            let ty = Rc::new(ast_ty.t.clone().into());
            let resolved_ty = self.resolve_type(&ty, ast_ty.span)?;
            let resolved_expr_ty = self.resolve_type(&expr_type, expr.span)?;
            if resolved_ty != resolved_expr_ty {
                return Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::TypeMismatch(ty, expr_type),
                    let_expr.span,
                )]);
            }
        }
        match &pattern.t {
            Pattern::String(var_name) => {
                // Prefer the annotated type if provided
                let ty = if let Some(ty) = ast_ty {
                    Rc::new(ty.t.clone().into())
                } else {
                    expr_type
                };
                self.insert_var(
                    var_name.clone(),
                    VarProperties {
                        ty,
                        mutable: mutable.t,
                    },
                    let_expr.span,
                )
            }
            _ => (),
        }
        Ok(Rc::new(Type::Unit))
    }

    fn translate_fn_call(
        &self,
        Spanned {
            t: FnCall { id, args },
            span,
        }: &Spanned<FnCall>,
    ) -> Result<Rc<Type>> {
        if let Some(fn_type) = self.vars.get(&id.t).map(|x| x.ty.clone()) {
            match self.resolve_type(&fn_type, id.span)?.as_ref() {
                Type::Fn(param_types, return_type) => {
                    if args.len() != param_types.len() {
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::ArityMismatch(param_types.len(), args.len()),
                            *span,
                        )]);
                    }

                    let mut errs = vec![];
                    for (arg, param_type) in args.iter().zip(param_types.iter()) {
                        match self.translate_expr(arg) {
                            Ok(ty) => {
                                // param_type should already be well-defined because we have already
                                // checked for invalid types
                                if self.resolve_type(&ty, arg.span)?
                                    != self.resolve_type(param_type, zspan!()).unwrap()
                                {
                                    errs.push(TypecheckErr::new_err(
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

                    Ok(return_type.clone())
                }
                _ => Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::NotAFn(id.t.clone()),
                    id.span,
                )]),
            }
        } else {
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedFn(id.t.clone()),
                *span,
            )])
        }
    }

    fn translate_record(
        &self,
        Spanned {
            t: Record {
                id: record_id,
                field_assigns,
            },
            span,
        }: &Spanned<Record>,
    ) -> Result<Rc<Type>> {
        if let Some(ty) = self.types.get(&record_id.t) {
            if let Type::Record(field_types) = ty.as_ref() {
                let mut field_assigns_hm = HashMap::new();
                for field_assign in field_assigns {
                    field_assigns_hm.insert(field_assign.id.t.clone(), field_assign.expr.clone());
                }

                let mut errors = vec![];

                let missing_fields: HashSet<&String> = field_types
                    .keys()
                    .collect::<HashSet<&String>>()
                    .difference(&field_assigns_hm.keys().collect())
                    .map(|&k| k)
                    .collect();
                if !missing_fields.is_empty() {
                    let mut missing_fields: Vec<String> =
                        missing_fields.into_iter().cloned().collect();
                    missing_fields.sort_unstable();
                    errors.push(TypecheckErr::new_err(
                        TypecheckErrType::MissingFields(missing_fields),
                        *span,
                    ));
                }

                let invalid_fields: HashSet<&String> = field_assigns_hm
                    .keys()
                    .collect::<HashSet<&String>>()
                    .difference(&field_types.keys().collect())
                    .map(|&k| k)
                    .collect();
                if !invalid_fields.is_empty() {
                    let mut invalid_fields: Vec<String> =
                        invalid_fields.into_iter().cloned().collect();
                    invalid_fields.sort_unstable();
                    errors.push(TypecheckErr::new_err(
                        TypecheckErrType::InvalidFields(invalid_fields),
                        *span,
                    ));
                }

                let mut checked_fields: HashMap<&String, ByteSpan> = HashMap::new();
                for Spanned {
                    t: FieldAssign { id, .. },
                    span,
                } in field_assigns
                {
                    if let Some(prev_def_span) = checked_fields.get(&id.t) {
                        errors.push(
                            TypecheckErr::new_err(
                                TypecheckErrType::DuplicateField(id.t.clone()),
                                *span,
                            )
                            .with_source(Spanned::new(
                                format!("{} was defined here", id.t.clone()),
                                *prev_def_span,
                            )),
                        );
                    }
                    checked_fields.insert(&id.t, *span);
                }

                if !errors.is_empty() {
                    return Err(errors);
                }

                let mut errors = vec![];
                for Spanned {
                    t: FieldAssign { id: field_id, expr },
                    span,
                } in field_assigns
                {
                    // This should never error because we already checked for invalid types
                    let expected_type = self
                        .resolve_type(&field_types[&field_id.t], zspan!())
                        .unwrap();
                    let ty = self.translate_expr(expr)?;
                    let actual_type = self.resolve_type(&ty, expr.span)?;
                    if expected_type != actual_type {
                        errors.push(TypecheckErr::new_err(
                            TypecheckErrType::TypeMismatch(
                                expected_type.clone(),
                                actual_type.clone(),
                            ),
                            *span,
                        ));
                    }
                }

                if !errors.is_empty() {
                    return Err(errors);
                }

                Ok(ty.clone())
            } else {
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::NotARecord(record_id.t.clone()),
                    *span,
                )])
            }
        } else {
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedType(record_id.t.clone()),
                record_id.span,
            )])
        }
    }

    fn translate_assign(&self, Spanned { t: assign, span }: &Spanned<Assign>) -> Result<Rc<Type>> {
        match &assign.lval.t {
            LVal::Simple(var) => {
                if let Some(var_properties) = self.vars.get(var) {
                    if !var_properties.mutable {
                        let def_span = self.var_def_spans[var];
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::MutatingImmutable(var.clone()),
                            assign.lval.span,
                        )
                        .with_source(Spanned::new(
                            format!("{} was defined here", var.clone()),
                            def_span,
                        ))]);
                    }

                    let resolved_expected_ty = self
                        .resolve_type(&var_properties.ty, assign.lval.span)?
                        .clone();
                    let resolved_actual_ty = self.translate_expr(&assign.expr)?;
                    if resolved_expected_ty != resolved_actual_ty {
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::TypeMismatch(
                                resolved_expected_ty,
                                resolved_actual_ty,
                            ),
                            *span,
                        )]);
                    }
                }
            }
            _ => unimplemented!(),
        }

        Ok(Rc::new(Type::Unit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{self, BoolOp, TypeField};
    use codespan::ByteOffset;
    use maplit::hashmap;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_resolve_type() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("b".to_owned()), zspan!());
        env.insert_type("b".to_owned(), Type::Alias("c".to_owned()), zspan!());
        env.insert_type("c".to_owned(), Type::Int, zspan!());
        env.insert_type("d".to_owned(), Type::Alias("e".to_owned()), zspan!());

        assert_eq!(
            env.resolve_type(&Rc::new(Type::Alias("a".to_owned())), zspan!()),
            Ok(Rc::new(Type::Int))
        );
        assert_eq!(
            env.resolve_type(&Rc::new(Type::Alias("d".to_owned())), zspan!())
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::UndefinedType("e".to_owned()),
        );
    }

    #[test]
    fn test_translate_bool_expr() {
        let expr = zspan!(ExprType::Bool(
            Box::new(zspan!(ExprType::BoolLiteral(true))),
            zspan!(BoolOp::And),
            Box::new(zspan!(ExprType::BoolLiteral(true))),
        ));
        let env = Env::default();
        assert_eq!(env.translate_expr(&expr), Ok(Rc::new(Type::Bool)));
    }

    #[test]
    fn test_translate_bool_expr_source() {
        let expr = zspan!(ExprType::Bool(
            Box::new(zspan!(ExprType::Bool(
                Box::new(zspan!(ExprType::BoolLiteral(true))),
                zspan!(BoolOp::And),
                Box::new(zspan!(ExprType::Number(1)))
            ))),
            zspan!(BoolOp::And),
            Box::new(zspan!(ExprType::BoolLiteral(true))),
        ));
        let env = Env::default();
        assert_eq!(
            env.translate_expr(&expr),
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeMismatch(Rc::new(Type::Bool), Rc::new(Type::Int)),
                span!(0, 0, ByteOffset(0))
            )])
        );
    }

    #[test]
    fn test_translate_record_field() {
        let mut env = Env::default();
        let record = hashmap! {
            "f".to_owned() => Rc::new(Type::Int)
        };
        env.insert_var(
            "x".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Record(record)),
                mutable: false,
            },
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("f".to_owned())
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Ok(VarProperties {
                ty: Rc::new(Type::Int),
                mutable: false
            })
        );
    }

    #[test]
    fn test_translate_record_field_err1() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Record(HashMap::new())),
                mutable: false,
            },
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedField("g".to_owned()),
                span!(0, 0, ByteOffset(0)),
            )])
        );
    }

    #[test]
    fn test_translate_record_field_err2() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Int),
                mutable: false,
            },
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedField("g".to_owned()),
                span!(0, 0, ByteOffset(0))
            )])
        );
    }

    #[test]
    fn test_translate_array_subscript() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Array(Rc::new(Type::Int), 3)),
                mutable: false,
            },
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Subscript(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!(ExprType::Number(0))
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Ok(VarProperties {
                ty: Rc::new(Type::Int),
                mutable: false
            })
        );
    }

    #[test]
    fn test_translate_let_type_annotation() {
        let mut env = Env::default();
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        assert_eq!(env.translate_let(&let_expr), Ok(Rc::new(Type::Unit)));
        assert_eq!(
            env.vars["x"],
            VarProperties {
                ty: Rc::new(Type::Int),
                mutable: false
            }
        );
        assert!(env.var_def_spans.contains_key("x"));
    }

    #[test]
    fn test_translate_let_type_annotation_err() {
        let mut env = Env::default();
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("string".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        assert_eq!(
            env.translate_let(&let_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::String), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_translate_fn_call_undefined_err() {
        let env = Env::default();
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert_eq!(
            env.translate_fn_call(&fn_call_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::UndefinedFn("f".to_owned())
        );
    }

    #[test]
    fn test_translate_fn_call_not_fn_err() {
        let mut env = Env::default();
        env.insert_var(
            "f".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Int),
                mutable: false,
            },
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert_eq!(
            env.translate_fn_call(&fn_call_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::NotAFn("f".to_owned())
        );
    }

    #[test]
    fn test_translate_fn_call_arity_mismatch() {
        let mut env = Env::default();
        env.insert_var(
            "f".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Int))),
                mutable: false,
            },
            zspan!(),
        );
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![zspan!(ExprType::Number(0))],
        });
        assert_eq!(
            env.translate_fn_call(&fn_call_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::ArityMismatch(0, 1)
        );
    }

    #[test]
    fn test_translate_fn_call_arg_type_mismatch() {
        let mut env = Env::default();
        env.insert_var(
            "f".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Int))),
                mutable: false,
            },
            zspan!(),
        );
    }

    #[test]
    fn test_translate_fn_call_returns_aliased_type() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("int".to_owned()), zspan!());
        env.insert_var(
            "f".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Alias("a".to_owned())))),
                mutable: false,
            },
            zspan!(),
        );
        let fn_call = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![]
        });
        assert_eq!(
            env.translate_fn_call(&fn_call).unwrap(),
            Rc::new(Type::Alias("a".to_owned()))
        );
    }

    #[test]
    fn test_translate_typedef() {
        let mut env = Env::default();
        let type_decl = zspan!(TypeDecl {
            id: zspan!("a".to_owned()),
            ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
        });
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::Wildcard),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("a".to_owned())))),
            expr: zspan!(ExprType::Number(0)),
        });
        let _ = env.translate_type_decl(&type_decl);
        assert!(env.translate_let(&let_expr).is_ok());
    }

    #[test]
    fn test_translate_typedef_err() {
        let mut env = Env::default();
        let type_decl = zspan!(TypeDecl {
            id: zspan!("a".to_owned()),
            ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
        });
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::Wildcard),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("a".to_owned())))),
            expr: zspan!(ExprType::String("".to_owned())),
        });
        let _ = env.translate_type_decl(&type_decl);
        assert_eq!(
            env.translate_let(&let_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(
                Rc::new(Type::Alias("a".to_owned())),
                Rc::new(Type::String)
            )
        );
    }

    #[test]
    fn test_translate_expr_typedef() {
        let mut env = Env::default();
        let type_decl = zspan!(TypeDecl {
            id: zspan!("i".to_owned()),
            ty: zspan!(ast::Type::Type(zspan!("int".to_owned())))
        });
        let var_def = zspan!(Let {
            pattern: zspan!(Pattern::String("i".to_owned())),
            mutable: zspan!(false),
            ty: Some(zspan!(ast::Type::Type(zspan!("i".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        let expr = zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple(
            "i".to_owned()
        )))));
        env.translate_type_decl(&type_decl)
            .expect("translate type decl");
        env.translate_let(&var_def).expect("translate var def");
        assert_eq!(
            env.translate_expr_mut(&expr).unwrap(),
            Rc::new(Type::Alias("i".to_owned()))
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
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeDeclCycle("a".to_owned(), Rc::new(Type::Alias("a".to_owned())))
        );
    }

    #[test]
    fn test_check_for_type_decl_cycles_err2() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("b".to_owned()), zspan!());
        env.insert_type("b".to_owned(), Type::Alias("c".to_owned()), zspan!());
        env.insert_type("c".to_owned(), Type::Alias("a".to_owned()), zspan!());
        assert_eq!(
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeDeclCycle("c".to_owned(), Rc::new(Type::Alias("a".to_owned())))
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
    fn test_duplicate_type_decl() {
        let mut env = Env::default();
        env.translate_type_decl(&zspan!(ast::TypeDecl {
            id: zspan!("a".to_owned()),
            ty: zspan!(ast::Type::Unit)
        }))
        .expect("translate type decl");

        assert_eq!(
            env.translate_type_decl(&zspan!(ast::TypeDecl {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Unit)
            }))
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::DuplicateType("a".to_owned())
        );
    }

    #[test]
    fn test_check_for_invalid_types_in_fn_sig() {
        let mut env = Env::default();
        env.insert_var(
            "f".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Alias("a".to_owned())))),
                mutable: false,
            },
            zspan!(),
        );
        assert_eq!(
            env.check_for_invalid_types().unwrap_err()[0].t.ty,
            TypecheckErrType::UndefinedType("a".to_owned())
        );
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
                    body: zspan!(ExprType::Unit)
                })),
                zspan!(),
            ),
        ]);
        assert_eq!(
            result,
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeDeclCycle(
                    "a".to_owned(),
                    Rc::new(Type::Alias("a".to_owned()))
                ),
                zspan!()
            )])
        );
    }

    #[test]
    fn test_validate_type() {
        let mut env = Env::default();
        let mut record_fields = HashMap::new();
        record_fields.insert("f".to_owned(), Rc::new(Type::Alias("a".to_owned())));
        record_fields.insert("g".to_owned(), Rc::new(Type::Alias("b".to_owned())));
        env.insert_type("a".to_owned(), Type::Record(record_fields), zspan!());

        let errs = env.check_for_invalid_types().unwrap_err();
        assert_eq!(errs.len(), 1);
        // Recursive type def in records is allowed
        assert_eq!(
            errs[0].t.ty,
            TypecheckErrType::UndefinedType("b".to_owned())
        );
    }

    #[test]
    fn test_translate_record_missing_fields() {
        let mut env = Env::default();
        let record_type = Type::Record(hashmap! {
            "a".to_owned() => Rc::new(Type::Int),
        });
        env.insert_type("r".to_owned(), record_type, zspan!());
        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![]
        });
        assert_eq!(
            env.translate_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrType::MissingFields(vec!["a".to_owned()])
        );
    }

    #[test]
    fn test_translate_record_invalid_fields() {
        let mut env = Env::default();
        let record_type = Type::Record(HashMap::new());
        env.insert_type("r".to_owned(), record_type, zspan!());
        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![zspan!(FieldAssign {
                id: zspan!("b".to_owned()),
                expr: zspan!(ExprType::Number(0))
            })]
        });
        assert_eq!(
            env.translate_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrType::InvalidFields(vec!["b".to_owned()])
        );
    }

    #[test]
    fn test_translate_record_duplicate_field() {
        let mut env = Env::default();
        let record_type = Type::Record(hashmap! {
            "a".to_owned() => Rc::new(Type::Int),
        });

        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans.insert(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => zspan!(),
            },
        );

        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![
                zspan!(FieldAssign {
                    id: zspan!("a".to_owned()),
                    expr: zspan!(ExprType::Number(0))
                }),
                zspan!(FieldAssign {
                    id: zspan!("a".to_owned()),
                    expr: zspan!(ExprType::Number(0))
                })
            ]
        });

        assert_eq!(
            env.translate_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrType::DuplicateField("a".to_owned())
        );
    }

    #[test]
    fn test_translate_record() {
        let mut env = Env::default();
        let record_type = Type::Record(hashmap! {
            "a".to_owned() => Rc::new(Type::Int),
            "b".to_owned() => Rc::new(Type::String),
        });
        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans.insert(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => zspan!(),
                "b".to_owned() => zspan!(),
            },
        );
        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![
                zspan!(FieldAssign {
                    id: zspan!("a".to_owned()),
                    expr: zspan!(ExprType::Number(0))
                }),
                zspan!(FieldAssign {
                    id: zspan!("b".to_owned()),
                    expr: zspan!(ExprType::String("b".to_owned()))
                }),
            ]
        });

        assert_eq!(env.translate_record(&record).unwrap(), Rc::new(record_type));
    }

    #[test]
    fn test_translate_fn_independent() {
        let mut env = Env::default();
        let fn_expr1 = zspan!(ExprType::Seq(
            vec![zspan!(ExprType::Let(Box::new(zspan!(Let {
                pattern: zspan!(Pattern::String("a".to_owned())),
                mutable: zspan!(false),
                ty: None,
                expr: zspan!(ExprType::Number(0))
            }))))],
            false
        ));
        let fn_expr2 = zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple(
            "a".to_owned()
        )))));

        let fn1 = zspan!(DeclType::Fn(zspan!(FnDecl {
            id: zspan!("f1".to_owned()),
            type_fields: vec![],
            return_type: None,
            body: fn_expr1,
        })));
        let fn2 = zspan!(DeclType::Fn(zspan!(FnDecl {
            id: zspan!("f2".to_owned()),
            type_fields: vec![],
            return_type: None,
            body: fn_expr2,
        })));

        assert_eq!(
            env.translate_decls(&[fn1, fn2]).unwrap_err()[0].t.ty,
            TypecheckErrType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn test_translate_seq_independent() {
        let env = Env::default();
        let seq_expr1 = zspan!(ExprType::Seq(
            vec![zspan!(ExprType::Let(Box::new(zspan!(Let {
                pattern: zspan!(Pattern::String("a".to_owned())),
                mutable: zspan!(false),
                ty: None,
                expr: zspan!(ExprType::Number(0))
            }))))],
            false
        ));
        let seq_expr2 = zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple(
            "a".to_owned()
        )))));

        env.translate_expr(&seq_expr1).expect("translate expr");
        assert_eq!(
            env.translate_expr(&seq_expr2).unwrap_err()[0].t.ty,
            TypecheckErrType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn test_illegal_let_expr() {
        let env = Env::default();
        let expr = zspan!(ExprType::Let(Box::new(zspan!(Let {
            pattern: zspan!(Pattern::Wildcard),
            mutable: zspan!(false),
            ty: None,
            expr: zspan!(ExprType::Let(Box::new(zspan!(Let {
                pattern: zspan!(Pattern::Wildcard),
                mutable: zspan!(false),
                ty: None,
                expr: zspan!(ExprType::Unit)
            }))))
        }))));

        assert_eq!(
            env.translate_expr(&expr).unwrap_err()[0].t.ty,
            TypecheckErrType::IllegalLetExpr
        );
    }

    #[test]
    fn test_assign_immut_err() {
        let mut env = Env::default();
        env.insert_var(
            "a".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Int),
                mutable: false,
            },
            zspan!(),
        );

        assert_eq!(
            env.translate_assign(&zspan!(Assign {
                lval: zspan!(LVal::Simple("a".to_owned())),
                expr: zspan!(ExprType::Number(0))
            }))
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::MutatingImmutable("a".to_owned())
        );
    }
}
