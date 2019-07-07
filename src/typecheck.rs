use crate::ast::{
    self, Arith, ArithOp, Array, Assign, Bool, Compare, Decl, DeclType, Enum, Expr, ExprType, FieldAssign, FnCall,
    FnDecl, For, If, LVal, Let, Pattern, Range, Record, Spanned, TypeDecl, TypeDeclType, While, Closure
};
use codespan::{ByteIndex, ByteSpan};
use codespan_reporting::{Diagnostic, Label};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
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
    NotARecord(Rc<Type>),
    NotAnArray(Rc<Type>),
    NotAnEnum(Rc<Type>),
    NotAnEnumCase(String),
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
    DuplicateEnumCase(String),
    DuplicateParam(String),
    MutatingImmutable(String),
    NonConstantArithExpr(ExprType),
    NegativeArrayLen(i64),
    IllegalNestedEnumDecl,
}

impl TypecheckErrType {
    pub fn diagnostic(&self, env: &Env) -> Diagnostic {
        use TypecheckErrType::*;
        let msg = match self {
            TypeMismatch(expected, actual) => {
                format!("type mismatch\nexpected: {:?}\n  actual: {:?}", expected, actual)
            }
            ArityMismatch(expected, actual) => {
                format!("arity mismatch\nexpected: {:?}\n  actual: {:?}", expected, actual)
            }
            NotAFn(fun) => {
                let ty = env.get_var(fun).unwrap();
                format!("not a function: {} (has type {:?})", fun, ty)
            }
            NotARecord(ty) => format!("not a record: (has type {:?})", ty.as_ref()),
            NotAnArray(ty) => format!("not an array: (has type {:?})", ty.as_ref()),
            NotAnEnum(ty) => format!("not an enum: (has type {:?})", ty.as_ref()),
            NotAnEnumCase(case_id) => format!("not an enum case: {}", case_id),
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
            DuplicateEnumCase(enum_case) => format!("duplicate enum case declaration for {}", enum_case),
            DuplicateParam(param) => format!("duplicate function param declaration for {}", param),
            MutatingImmutable(var) => format!("{} was declared as immutable", var),
            NonConstantArithExpr(expr) => format!("{:?} is not a constant arithmetic expression", expr),
            NegativeArrayLen(..) => "array length cannot be negative".to_owned(),
            IllegalNestedEnumDecl => "enum declarations cannot be nested".to_owned(),
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
    Record(String, HashMap<String, Rc<Type>>),
    Array(Rc<Type>, usize),
    Unit,
    Alias(String),
    Enum(String, HashMap<String, EnumCase>),
    Fn(Vec<Rc<Type>>, Rc<Type>),
    Iterator(Rc<Type>),
}

impl Type {
    fn alias(&self) -> Option<&str> {
        if let Type::Alias(alias) = self {
            Some(alias)
        } else {
            None
        }
    }

    fn from_type_decl(type_decl: TypeDecl) -> Self {
        match type_decl.ty.t {
            TypeDeclType::Type(ty) => {
                if ty.t == "int" {
                    Type::Int
                } else if ty.t == "string" {
                    Type::String
                } else {
                    Type::Alias(ty.t)
                }
            }
            TypeDeclType::Enum(cases) => Type::Enum(
                type_decl.id.t.clone(),
                cases
                    .into_iter()
                    .map(|c| (c.t.id.t.clone().into(), c.t.into()))
                    .collect(),
            ),
            TypeDeclType::Record(type_fields) => {
                let mut type_fields_hm = HashMap::new();
                for TypeField { id, ty } in type_fields.into_iter().map(|tf| tf.t.into()) {
                    type_fields_hm.insert(id, ty);
                }
                Type::Record(type_decl.id.t.clone(), type_fields_hm)
            }
            TypeDeclType::Array(ty, len) => Type::Array(Rc::new(ty.t.into()), len.t),
            TypeDeclType::Unit => Type::Unit,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumCase {
    pub id: String,
    pub params: Vec<Rc<Type>>,
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
            ast::Type::Array(ty, len) => Type::Array(Rc::new(ty.t.into()), len.t),
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

impl From<ast::EnumCase> for EnumCase {
    fn from(case: ast::EnumCase) -> Self {
        Self {
            id: case.id.t,
            params: case.params.into_iter().map(|p| Rc::new(p.t.into())).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarProperties {
    pub ty: Rc<Type>,
    pub immutable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LValProperties {
    pub ty: Rc<Type>,
    pub immutable: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Env<'a> {
    pub parent: Option<&'a Env<'a>>,
    pub vars: HashMap<String, VarProperties>,
    pub types: HashMap<String, Rc<Type>>,
    pub var_def_spans: HashMap<String, ByteSpan>,
    pub type_def_spans: HashMap<String, ByteSpan>,

    pub record_field_decl_spans: HashMap<String, HashMap<String, ByteSpan>>,
    pub fn_param_decl_spans: HashMap<String, Vec<ByteSpan>>,
    pub enum_case_param_decl_spans: HashMap<String, HashMap<String, Vec<ByteSpan>>>,
}

impl<'a> Default for Env<'a> {
    fn default() -> Self {
        let mut types = HashMap::new();
        types.insert("int".to_owned(), Rc::new(Type::Int));
        types.insert("string".to_owned(), Rc::new(Type::String));
        let mut type_def_spans = HashMap::new();
        type_def_spans.insert("int".to_owned(), ByteSpan::new(ByteIndex::none(), ByteIndex::none()));
        type_def_spans.insert("string".to_owned(), ByteSpan::new(ByteIndex::none(), ByteIndex::none()));
        Env {
            parent: None,
            vars: HashMap::new(),
            types,
            var_def_spans: HashMap::new(),
            type_def_spans,

            record_field_decl_spans: HashMap::new(),
            fn_param_decl_spans: HashMap::new(),
            enum_case_param_decl_spans: HashMap::new(),
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

impl<'a> Env<'a> {
    fn new() -> Self {
        Env {
            parent: None,
            vars: HashMap::new(),
            types: HashMap::new(),
            var_def_spans: HashMap::new(),
            type_def_spans: HashMap::new(),

            record_field_decl_spans: HashMap::new(),
            fn_param_decl_spans: HashMap::new(),
            enum_case_param_decl_spans: HashMap::new(),
        }
    }

    fn new_child(&'a self) -> Env<'a> {
        let mut child = Env::new();
        child.parent = Some(self);
        child
    }

    fn insert_var(&mut self, name: String, var_props: VarProperties, def_span: ByteSpan) {
        self.vars.insert(name.clone(), var_props);
        self.var_def_spans.insert(name, def_span);
    }

    fn insert_type(&mut self, name: String, ty: Type, def_span: ByteSpan) {
        self.types.insert(name.clone(), Rc::new(ty));
        self.type_def_spans.insert(name, def_span);
    }

    fn get_var(&self, name: &str) -> Option<&VarProperties> {
        if let Some(var) = self.vars.get(name) {
            Some(var)
        } else if let Some(parent) = self.parent {
            parent.get_var(name)
        } else {
            None
        }
    }

    fn get_type(&self, name: &str) -> Option<&Rc<Type>> {
        if let Some(ty) = self.types.get(name) {
            Some(ty)
        } else if let Some(parent) = self.parent {
            parent.get_type(name)
        } else {
            None
        }
    }

    fn get_var_def_span(&self, id: &str) -> Option<ByteSpan> {
        if let Some(span) = self.var_def_spans.get(id) {
            Some(*span)
        } else if let Some(parent) = self.parent {
            parent.get_var_def_span(id)
        } else {
            None
        }
    }

    fn get_type_def_span(&self, id: &str) -> Option<ByteSpan> {
        if let Some(span) = self.type_def_spans.get(id) {
            Some(*span)
        } else if let Some(parent) = self.parent {
            parent.get_type_def_span(id)
        } else {
            None
        }
    }

    fn get_record_field_decl_spans(&self, id: &str) -> Option<&HashMap<String, ByteSpan>> {
        if let Some(spans) = self.record_field_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.get_record_field_decl_spans(id)
        } else {
            None
        }
    }

    fn get_fn_param_decl_spans(&self, id: &str) -> Option<&[ByteSpan]> {
        if let Some(spans) = self.fn_param_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.get_fn_param_decl_spans(id)
        } else {
            None
        }
    }

    fn get_enum_case_param_decl_spans(&self, id: &str) -> Option<&HashMap<String, Vec<ByteSpan>>> {
        if let Some(spans) = self.enum_case_param_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.get_enum_case_param_decl_spans(id)
        } else {
            None
        }
    }

    fn contains_type(&self, name: &str) -> bool {
        if self.types.contains_key(name) {
            true
        } else if let Some(parent) = self.parent {
            parent.contains_type(name)
        } else {
            false
        }
    }

    /// Used to check if two types refer to the same "base type." Follows aliases and aliases in
    /// arrays.
    ///
    /// Does not follow aliases in records or enums because they can be recursive. This means that constructing `Type`s
    /// by hand instead of retrieving them from `types` in `Env` will result in failed equality
    /// checks, even if the aliases that they contain point to the same base type.
    fn resolve_type(&self, ty: &Rc<Type>, def_span: ByteSpan) -> Result<Rc<Type>> {
        match ty.as_ref() {
            Type::Alias(alias) => {
                if let Some(resolved_type) = self.get_type(alias) {
                    let span = self.get_type_def_span(alias).unwrap();
                    self.resolve_type(resolved_type, span)
                } else {
                    Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::UndefinedType(alias.clone()),
                        def_span,
                    )])
                }
            }
            Type::Array(elem_type, len) => Ok(Rc::new(Type::Array(self.resolve_type(elem_type, def_span)?, *len))),
            _ => Ok(ty.clone()),
        }
    }

    fn check_for_type_decl_cycles(&self, ty: &str, path: Vec<&str>) -> Result<()> {
        if let Some(alias) = self.types.get(ty) {
            if let Some(alias_str) = alias.alias() {
                if path.contains(&alias_str) {
                    let span = self.get_type_def_span(ty).unwrap();
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
            Type::String | Type::Bool | Type::Unit | Type::Int | Type::Iterator(_) => Ok(()),
            Type::Alias(_) => self.resolve_type(ty, def_span).map(|_| ()),
            Type::Record(_, record) => {
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
            Type::Enum(_, cases) => {
                let mut errors = vec![];
                for case in cases.values() {
                    for param in &case.params {
                        // Don't allow nested enum decls
                        if let Type::Enum(..) = param.as_ref() {
                            errors.push(TypecheckErr::new_err(TypecheckErrType::IllegalNestedEnumDecl, def_span));
                            continue;
                        }
                        match self.validate_type(param, def_span) {
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
            match self.validate_type(ty, self.get_type_def_span(id).unwrap()) {
                Ok(()) => (),
                Err(errs) => errors.extend(errs),
            }
        }
        for (id, var) in &self.vars {
            if let Type::Fn(param_types, return_type) = var.ty.as_ref() {
                for ty in param_types {
                    match self.validate_type(ty, self.get_var_def_span(id).unwrap()) {
                        Ok(()) => (),
                        Err(errs) => errors.extend(errs),
                    }
                }
                match self.validate_type(return_type, self.get_var_def_span(id).unwrap()) {
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

    fn check_for_duplicates<'b, I, T: 'b, Key: Hash + Eq, KeyFn, ErrGen>(
        iter: I,
        key_fn: KeyFn,
        err_gen: ErrGen,
    ) -> Result<HashMap<Key, ByteSpan>>
    where
        I: IntoIterator<Item = &'b Spanned<T>>,
        KeyFn: Fn(&'b Spanned<T>) -> Key,
        ErrGen: Fn(&'b Spanned<T>, ByteSpan) -> TypecheckErr,
    {
        // t -> def span
        let mut checked_elems: HashMap<Key, ByteSpan> = HashMap::new();
        let mut errors: Vec<TypecheckErr> = vec![];
        for elem in iter {
            let key = key_fn(elem);
            if let Some(prev_def_span) = checked_elems.get(&key) {
                errors.push(err_gen(elem, *prev_def_span));
            } else {
                checked_elems.insert(key, elem.span);
            }
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(checked_elems)
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
            _ => unreachable!(), // DeclType::Error
        }
    }

    fn translate_fn_decl_sig(&mut self, fn_decl: &Spanned<FnDecl>) -> Result<()> {
        // Check if there already exists another function with the same name
        if self.vars.contains_key(&fn_decl.id.t) {
            let span = self.get_var_def_span(&fn_decl.id.t).unwrap();
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::DuplicateFn(fn_decl.id.t.clone()),
                fn_decl.id.span,
            )
            .with_source(Spanned::new(
                format!("{} was defined here", fn_decl.id.t.clone()),
                span,
            ))]);
        }

        let param_decl_spans = Self::check_for_duplicates(
            &fn_decl.type_fields,
            |type_field| &type_field.id.t,
            |type_field, span| {
                TypecheckErr::new_err(
                    TypecheckErrType::DuplicateParam(type_field.id.t.clone()),
                    type_field.span,
                )
                .with_source(Spanned::new(
                    format!("{} was declared here", type_field.id.t.clone()),
                    span,
                ))
            },
        )?
        .values()
        .cloned()
        .collect();
        self.fn_param_decl_spans.insert(fn_decl.id.t.clone(), param_decl_spans);

        let param_types = fn_decl
            .type_fields
            .iter()
            .map(|type_field| Rc::new(type_field.ty.t.clone().into()))
            .collect();
        let return_type = if let Some(Spanned { t: return_type, .. }) = &fn_decl.return_type {
            return_type.clone().into()
        } else {
            Type::Unit
        };

        self.insert_var(
            fn_decl.id.t.clone(),
            VarProperties {
                ty: Rc::new(Type::Fn(param_types, Rc::new(return_type))),
                immutable: true,
            },
            fn_decl.span,
        );
        Ok(())
    }

    /// Creates a child env to typecheck the function body.
    fn translate_fn_decl_body(&self, fn_decl: &Spanned<FnDecl>) -> Result<()> {
        let mut new_env = self.new_child();

        let fn_type = new_env.get_var(&fn_decl.id.t).unwrap().ty.clone();
        if let Type::Fn(param_types, return_type) = fn_type.as_ref() {
            for ((param_id, span), param_type) in fn_decl
                .type_fields
                .iter()
                .map(|tf| (&tf.id.t, tf.span))
                .zip(param_types.iter())
            {
                new_env.insert_var(
                    param_id.clone(),
                    VarProperties {
                        ty: param_type.clone(),
                        immutable: false,
                    },
                    span,
                );
            }

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
        } else {
            unreachable!(format!("expected {} to be a function", fn_decl.id.t));
        }

        Ok(())
    }

    fn translate_type_decl(&mut self, decl: &Spanned<TypeDecl>) -> Result<()> {
        let id = decl.id.t.clone();

        if self.contains_type(&id) {
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::DuplicateType(id.clone()),
                decl.span,
            )
            .with_source(Spanned::new(
                format!("{} was defined here", id.clone()),
                self.get_type_def_span(&id).unwrap(),
            ))]);
        }

        let ty = Type::from_type_decl(decl.t.clone());
        match &decl.ty.t {
            TypeDeclType::Record(record_fields) => {
                let field_def_spans: HashMap<String, ByteSpan> = Self::check_for_duplicates(
                    record_fields,
                    |field| &field.id.t,
                    |field, span| {
                        let field_id = field.id.t.clone();
                        TypecheckErr::new_err(TypecheckErrType::DuplicateField(field_id.clone()), field.span)
                            .with_source(Spanned::new(format!("{} was declared here", field_id), span))
                    },
                )?
                .into_iter()
                .map(|(k, v)| (k.clone(), v))
                .collect();

                self.record_field_decl_spans.insert(id.clone(), field_def_spans);
            }
            TypeDeclType::Enum(cases) => {
                Self::check_for_duplicates(
                    cases,
                    |case| &case.id.t,
                    |case, span| {
                        let case_id = case.id.t.clone();
                        TypecheckErr::new_err(TypecheckErrType::DuplicateEnumCase(case_id.clone()), case.span)
                            .with_source(Spanned::new(format!("{} was declared here", case_id), span))
                    },
                )?;

                let decl_spans = self
                    .enum_case_param_decl_spans
                    .entry(decl.id.t.clone())
                    .or_insert(HashMap::new());
                for case in cases {
                    decl_spans.insert(case.id.t.clone(), case.params.iter().map(|param| param.span).collect());
                }
            }
            _ => (),
        }
        self.insert_type(id, ty, decl.span);

        Ok(())
    }

    fn translate_expr(&self, expr: &Expr) -> Result<Rc<Type>> {
        match &expr.t {
            ExprType::Seq(exprs, returns) => {
                // New scope
                let mut new_env = self.new_child();
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
            ExprType::Arith(arith) => self.translate_arith(arith),
            ExprType::Unit | ExprType::Continue | ExprType::Break => Ok(Rc::new(Type::Unit)),
            ExprType::BoolLiteral(_) => Ok(Rc::new(Type::Bool)),
            ExprType::Not(expr) => assert_ty!(self, expr, Rc::new(Type::Bool)),
            ExprType::Bool(bool_expr) => self.translate_bool(bool_expr),
            ExprType::LVal(lval) => Ok(self.translate_lval(lval)?.ty),
            ExprType::Let(_) => Err(vec![TypecheckErr::new_err(TypecheckErrType::IllegalLetExpr, expr.span)]),
            ExprType::FnCall(fn_call) => self.translate_fn_call(fn_call),
            ExprType::Record(record) => self.translate_record(record),
            ExprType::Assign(assign) => self.translate_assign(assign),
            ExprType::Array(array) => self.translate_array(array),
            ExprType::If(if_expr) => self.translate_if(if_expr),
            ExprType::Range(range) => self.translate_range(range),
            ExprType::For(for_expr) => self.translate_for(for_expr),
            ExprType::While(while_expr) => self.translate_while(while_expr),
            ExprType::Compare(compare) => self.translate_compare(compare),
            ExprType::Enum(enum_expr) => self.translate_enum(enum_expr),
            ExprType::Closure(closure) => self.translate_closure(closure),
        }
    }

    fn translate_expr_mut(&mut self, expr: &Expr) -> Result<Rc<Type>> {
        match &expr.t {
            ExprType::Let(let_expr) => self.translate_let(let_expr),
            _ => self.translate_expr(expr),
        }
    }

    fn translate_lval(&self, lval: &Spanned<LVal>) -> Result<LValProperties> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(var_properties) = self.get_var(var) {
                    Ok(LValProperties {
                        ty: var_properties.ty.clone(),
                        immutable: if var_properties.immutable {
                            Some(var.clone())
                        } else {
                            None
                        },
                    })
                } else {
                    Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::UndefinedVar(var.clone()),
                        lval.span,
                    )])
                }
            }
            LVal::Field(var, field) => {
                let lval_properties = self.translate_lval(var)?;
                if let Type::Record(_, fields) = self.resolve_type(&lval_properties.ty, var.span)?.as_ref() {
                    if let Some(field_type) = fields.get(&field.t) {
                        // A field is mutable only if the record it belongs to is mutable
                        return Ok(LValProperties {
                            ty: field_type.clone(),
                            immutable: lval_properties.immutable,
                        });
                    }
                }
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::UndefinedField(field.t.clone()),
                    field.span,
                )])
            }
            LVal::Subscript(var, index) => {
                let lval_properties = self.translate_lval(var)?;
                let index_type = self.translate_expr(index)?;
                if let Type::Array(ty, _) = self.resolve_type(&lval_properties.ty, var.span)?.as_ref() {
                    if self.resolve_type(&index_type, index.span)? == Rc::new(Type::Int) {
                        Ok(LValProperties {
                            ty: ty.clone(),
                            immutable: lval_properties.immutable,
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
            immutable,
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
        if let Pattern::String(var_name) = &pattern.t {
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
                    immutable: immutable.t,
                },
                let_expr.span,
            )
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
        if let Some(fn_type) = self.get_var(&id.t).map(|x| x.ty.clone()) {
            match self.resolve_type(&fn_type, id.span)?.as_ref() {
                Type::Fn(param_types, return_type) => {
                    if args.len() != param_types.len() {
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::ArityMismatch(param_types.len(), args.len()),
                            *span,
                        )]);
                    }

                    let mut errors = vec![];
                    for (index, (arg, param_type)) in args.iter().zip(param_types.iter()).enumerate() {
                        match self.translate_expr(arg) {
                            Ok(ty) => {
                                // param_type should already be well-defined because we have already
                                // checked for invalid types
                                if self.resolve_type(&ty, arg.span)? != self.resolve_type(param_type, zspan!()).unwrap()
                                {
                                    let mut err = TypecheckErr::new_err(
                                        TypecheckErrType::TypeMismatch(param_type.clone(), ty.clone()),
                                        arg.span,
                                    );
                                    if let Some(decl_spans) = self.get_fn_param_decl_spans(id) {
                                        err.source = Some(Spanned::new("declared here".to_owned(), decl_spans[index]));
                                    }
                                    errors.push(err);
                                }
                            }
                            Err(errs) => errors.extend(errs),
                        }
                    }

                    if !errors.is_empty() {
                        return Err(errors);
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
        if let Some(ty) = self.get_type(&record_id.t) {
            if let Type::Record(_, field_types) = ty.as_ref() {
                let mut field_assigns_hm = HashMap::new();
                for field_assign in field_assigns {
                    field_assigns_hm.insert(field_assign.id.t.clone(), field_assign.expr.clone());
                }

                let mut errors = vec![];

                let missing_fields: HashSet<&String> = field_types
                    .keys()
                    .collect::<HashSet<&String>>()
                    .difference(&field_assigns_hm.keys().collect())
                    .cloned()
                    .collect();
                if !missing_fields.is_empty() {
                    let mut missing_fields: Vec<String> = missing_fields.into_iter().cloned().collect();
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
                    .cloned()
                    .collect();
                if !invalid_fields.is_empty() {
                    let mut invalid_fields: Vec<String> = invalid_fields.into_iter().cloned().collect();
                    invalid_fields.sort_unstable();
                    errors.push(TypecheckErr::new_err(
                        TypecheckErrType::InvalidFields(invalid_fields),
                        *span,
                    ));
                }

                if !errors.is_empty() {
                    return Err(errors);
                }

                Self::check_for_duplicates(
                    field_assigns,
                    |field_assign| &field_assign.id.t,
                    |field_assign, span| {
                        TypecheckErr::new_err(
                            TypecheckErrType::DuplicateField(field_assign.id.t.clone()),
                            field_assign.span,
                        )
                        .with_source(Spanned::new(
                            format!("{} was defined here", field_assign.id.t.clone()),
                            span,
                        ))
                    },
                )?;

                let mut errors = vec![];
                for Spanned {
                    t: FieldAssign { id: field_id, expr },
                    span,
                } in field_assigns
                {
                    // This should never error because we already checked for invalid types
                    let expected_type = self.resolve_type(&field_types[&field_id.t], zspan!()).unwrap();
                    let ty = self.translate_expr(expr)?;
                    let actual_type = self.resolve_type(&ty, expr.span)?;
                    if expected_type != actual_type {
                        errors.push(
                            TypecheckErr::new_err(
                                TypecheckErrType::TypeMismatch(expected_type.clone(), actual_type.clone()),
                                *span,
                            )
                            .with_source(Spanned::new(
                                format!("{} was declared here", field_id.t),
                                self.get_record_field_decl_spans(&record_id.t).unwrap()[&field_id.t],
                            )),
                        );
                    }
                }

                if !errors.is_empty() {
                    return Err(errors);
                }

                Ok(ty.clone())
            } else {
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::NotARecord(ty.clone()),
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
                if let Some(var_properties) = self.get_var(var) {
                    if var_properties.immutable {
                        let def_span = self.get_var_def_span(var).unwrap();
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::MutatingImmutable(var.clone()),
                            assign.lval.span,
                        )
                        .with_source(Spanned::new(format!("{} was defined here", var.clone()), def_span))]);
                    }

                    let resolved_expected_ty = self.resolve_type(&var_properties.ty, assign.lval.span)?;
                    let resolved_actual_ty =
                        self.resolve_type(&self.translate_expr(&assign.expr)?, assign.expr.span)?;
                    if resolved_expected_ty != resolved_actual_ty {
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::TypeMismatch(resolved_expected_ty, resolved_actual_ty),
                            *span,
                        )]);
                    }
                }
            }
            LVal::Field(lval, field) => {
                let lval_properties = self.translate_lval(&lval)?;
                if let Some(ref base_var) = lval_properties.immutable {
                    return Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::MutatingImmutable(base_var.clone()),
                        *span,
                    )
                    .with_source(Spanned::new(
                        format!("{} was defined here", base_var),
                        self.get_var_def_span(base_var).unwrap(),
                    ))]);
                }
                let lval_type = self.resolve_type(&lval_properties.ty, lval.span)?;
                if let Type::Record(_, record_field_types) = lval_type.as_ref() {
                    if let Some(expected_ty) = record_field_types.get(&field.t) {
                        let resolved_expected_ty = self.resolve_type(expected_ty, assign.lval.span)?;
                        let resolved_actual_ty =
                            self.resolve_type(&self.translate_expr(&assign.expr)?, assign.expr.span)?;
                        if resolved_expected_ty != resolved_actual_ty {
                            return Err(vec![TypecheckErr::new_err(
                                TypecheckErrType::TypeMismatch(resolved_expected_ty, resolved_actual_ty),
                                *span,
                            )]);
                        }
                    } else {
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::UndefinedField(field.t.clone()),
                            field.span,
                        )]);
                    }
                } else {
                    return Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::NotARecord(lval_type),
                        lval.span,
                    )]);
                }
            }
            LVal::Subscript(lval, _) => {
                let lval_properties = self.translate_lval(&lval)?;
                if let Some(ref base_var) = lval_properties.immutable {
                    return Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::MutatingImmutable(base_var.clone()),
                        *span,
                    )
                    .with_source(Spanned::new(
                        format!("{} was defined here", base_var),
                        self.get_var_def_span(base_var).unwrap(),
                    ))]);
                }
                let lval_type = self.resolve_type(&lval_properties.ty, lval.span)?;
                if let Type::Array(elem_type, _) = lval_type.as_ref() {
                    let resolved_expected_ty = self.resolve_type(elem_type, assign.lval.span)?;
                    let resolved_actual_ty =
                        self.resolve_type(&self.translate_expr(&assign.expr)?, assign.expr.span)?;
                    if resolved_expected_ty != resolved_actual_ty {
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::TypeMismatch(resolved_expected_ty, resolved_actual_ty),
                            *span,
                        )]);
                    }
                } else {
                    return Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::NotAnArray(lval_type),
                        lval.span,
                    )]);
                }
            }
        }

        Ok(Rc::new(Type::Unit))
    }

    fn eval_arith_const_expr(Spanned { t: expr, span }: &Expr) -> Result<i64> {
        match expr {
            ExprType::Number(num) => Ok(*num as i64),
            ExprType::Neg(expr) => Ok(-Self::eval_arith_const_expr(expr)?),
            ExprType::Arith(arith_expr) => {
                let l = Self::eval_arith_const_expr(&arith_expr.l)?;
                let r = Self::eval_arith_const_expr(&arith_expr.r)?;
                match arith_expr.op.t {
                    ArithOp::Add => Ok(l + r),
                    ArithOp::Sub => Ok(l - r),
                    ArithOp::Mul => Ok(l * r),
                    ArithOp::Div => Ok(l / r),
                }
            }
            _ => Err(vec![TypecheckErr::new_err(
                TypecheckErrType::NonConstantArithExpr(expr.clone()),
                *span,
            )]),
        }
    }

    fn translate_array(&self, Spanned { t: array, .. }: &Spanned<Array>) -> Result<Rc<Type>> {
        let elem_type = self.translate_expr(&array.initial_value)?;
        let len = Self::eval_arith_const_expr(&array.len)?;
        if len < 0 {
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::NegativeArrayLen(len),
                array.len.span,
            )]);
        }
        Ok(Rc::new(Type::Array(elem_type, len as usize)))
    }

    fn translate_if(&self, Spanned { t: expr, span }: &Spanned<If>) -> Result<Rc<Type>> {
        assert_ty!(self, &expr.cond, Rc::new(Type::Bool))?;
        let then_expr_type = self.translate_expr(&expr.then_expr)?;
        if let Some(else_expr) = &expr.else_expr {
            let else_expr_type = self.translate_expr(else_expr)?;
            if self.resolve_type(&then_expr_type, expr.then_expr.span)?
                == self.resolve_type(&else_expr_type, else_expr.span)?
            {
                // Arbitrarily pick the then branch
                Ok(then_expr_type)
            } else {
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::TypeMismatch(then_expr_type.clone(), else_expr_type.clone()),
                    *span,
                )
                .with_source(Spanned::new(
                    format!(
                        "then branch has type {:?}, but else branch has type {:?}",
                        then_expr_type, else_expr_type
                    ),
                    *span,
                ))])
            }
        } else {
            if then_expr_type != Rc::new(Type::Unit) {
                return Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), then_expr_type),
                    expr.then_expr.span,
                )]);
            }
            Ok(Rc::new(Type::Unit))
        }
    }

    fn translate_range(&self, Spanned { t: expr, .. }: &Spanned<Range>) -> Result<Rc<Type>> {
        assert_ty!(self, &expr.start, Rc::new(Type::Int))?;
        assert_ty!(self, &expr.end, Rc::new(Type::Int))?;
        Ok(Rc::new(Type::Iterator(Rc::new(Type::Int))))
    }

    fn translate_for(&self, Spanned { t: expr, .. }: &Spanned<For>) -> Result<Rc<Type>> {
        assert_ty!(self, &expr.range, Rc::new(Type::Iterator(Rc::new(Type::Int))))?;
        let mut child_env = self.new_child();
        child_env.insert_var(
            expr.index.t.clone(),
            VarProperties {
                ty: Rc::new(Type::Int),
                immutable: false,
            },
            expr.index.span,
        );
        assert_ty!(child_env, &expr.body, Rc::new(Type::Unit))?;
        Ok(Rc::new(Type::Unit))
    }

    fn translate_while(&self, Spanned { t: expr, .. }: &Spanned<While>) -> Result<Rc<Type>> {
        assert_ty!(self, &expr.cond, Rc::new(Type::Bool))?;
        assert_ty!(self, &expr.body, Rc::new(Type::Unit))?;
        Ok(Rc::new(Type::Unit))
    }

    fn translate_arith(&self, Spanned { t: expr, .. }: &Spanned<Arith>) -> Result<Rc<Type>> {
        assert_ty!(self, &expr.l, Rc::new(Type::Int))?;
        assert_ty!(self, &expr.r, Rc::new(Type::Int))?;
        Ok(Rc::new(Type::Int))
    }

    fn translate_bool(&self, Spanned { t: expr, .. }: &Spanned<Bool>) -> Result<Rc<Type>> {
        assert_ty!(self, &expr.l, Rc::new(Type::Bool))?;
        assert_ty!(self, &expr.r, Rc::new(Type::Bool))?;
        Ok(Rc::new(Type::Bool))
    }

    fn translate_compare(&self, Spanned { t: expr, .. }: &Spanned<Compare>) -> Result<Rc<Type>> {
        assert_ty!(self, &expr.l, Rc::new(Type::Bool))?;
        assert_ty!(self, &expr.r, Rc::new(Type::Bool))?;
        Ok(Rc::new(Type::Bool))
    }

    fn translate_enum(&self, Spanned { t: expr, .. }: &Spanned<Enum>) -> Result<Rc<Type>> {
        if let Some(ty) = self.get_type(&expr.enum_id) {
            if let Type::Enum(_, enum_cases) = ty.as_ref() {
                if let Some(EnumCase { params, .. }) = enum_cases.get(&expr.case_id.t) {
                    if params.len() != expr.args.len() {
                        return Err(vec![TypecheckErr::new_err(
                            TypecheckErrType::ArityMismatch(params.len(), expr.args.len()),
                            expr.args.span,
                        )]);
                    }

                    let mut errors = vec![];
                    for (index, (arg, param_type)) in expr.args.iter().zip(params.iter()).enumerate() {
                        match self.translate_expr(arg) {
                            Ok(ty) => {
                                // param_type should already be well-defined because we have already
                                // checked for invalid types
                                if self.resolve_type(&ty, arg.span)? != self.resolve_type(param_type, zspan!()).unwrap()
                                {
                                    let mut err = TypecheckErr::new_err(
                                        TypecheckErrType::TypeMismatch(param_type.clone(), ty.clone()),
                                        arg.span,
                                    );
                                    if let Some(decl_spans) = self.get_enum_case_param_decl_spans(&expr.enum_id) {
                                        err.source = Some(Spanned::new(
                                            "declared here".to_owned(),
                                            decl_spans[&expr.case_id.t][index],
                                        ));
                                    }
                                    errors.push(err);
                                }
                            }
                            Err(errs) => errors.extend(errs),
                        }
                    }

                    if !errors.is_empty() {
                        return Err(errors);
                    }

                    Ok(ty.clone())
                } else {
                    Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::NotAnEnumCase(expr.case_id.t.clone()),
                        expr.case_id.span,
                    )])
                }
            } else {
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::NotAnEnum(ty.clone()),
                    expr.enum_id.span,
                )])
            }
        } else {
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedType(expr.enum_id.t.clone()),
                expr.enum_id.span,
            )])
        }
    }

    fn translate_closure(&self, Spanned { t: expr, span }: &Spanned<Closure>) -> Result<Rc<Type>> {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{self, BoolOp, CompareOp, TypeField};
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
    fn test_child_env() {
        let mut env = Env::default();
        let var_properties = VarProperties {
            ty: Rc::new(Type::Int),
            immutable: true,
        };
        env.insert_var("a".to_owned(), var_properties.clone(), zspan!());
        let mut child_env = env.new_child();
        child_env.insert_type("i".to_owned(), Type::Alias("int".to_owned()), zspan!());

        assert!(child_env.contains_type("int"));
        assert!(child_env.contains_type("i"));
        assert_eq!(
            child_env.resolve_type(child_env.get_type("i").unwrap(), zspan!()),
            Ok(Rc::new(Type::Int))
        );
        assert_eq!(child_env.get_var("a"), Some(&var_properties));
    }

    #[test]
    fn test_translate_bool_expr() {
        let expr = zspan!(ExprType::Bool(Box::new(zspan!(Bool {
            l: zspan!(ExprType::BoolLiteral(true)),
            op: zspan!(BoolOp::And),
            r: zspan!(ExprType::BoolLiteral(true)),
        }))));
        let env = Env::default();
        assert_eq!(env.translate_expr(&expr), Ok(Rc::new(Type::Bool)));
    }

    #[test]
    fn test_translate_bool_expr_source() {
        let expr = zspan!(ExprType::Bool(Box::new(zspan!(Bool {
            l: zspan!(ExprType::Bool(Box::new(zspan!(Bool {
                l: zspan!(ExprType::BoolLiteral(true)),
                op: zspan!(BoolOp::And),
                r: zspan!(ExprType::Number(1))
            })))),
            op: zspan!(BoolOp::And),
            r: zspan!(ExprType::BoolLiteral(true)),
        }))));
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
    fn test_translate_lval_record_field() {
        let mut env = Env::default();
        let record = hashmap! {
            "f".to_owned() => Rc::new(Type::Int)
        };
        env.insert_var(
            "x".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Record("r".to_owned(), record)),
                immutable: false,
            },
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("f".to_owned())
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Ok(LValProperties {
                ty: Rc::new(Type::Int),
                immutable: None,
            })
        );
    }

    #[test]
    fn test_translate_lval_record_field_err1() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Record("r".to_owned(), HashMap::new())),
                immutable: true,
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
    fn test_translate_lval_record_field_err2() {
        let mut env = Env::default();
        env.insert_var(
            "x".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Int),
                immutable: true,
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
                immutable: true,
            },
            ByteSpan::new(ByteIndex::none(), ByteIndex::none()),
        );
        let lval = zspan!(LVal::Subscript(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!(ExprType::Number(0))
        ));
        assert_eq!(
            env.translate_lval(&lval),
            Ok(LValProperties {
                ty: Rc::new(Type::Int),
                immutable: Some("x".to_owned())
            })
        );
    }

    #[test]
    fn test_translate_let_type_annotation() {
        let mut env = Env::default();
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            immutable: zspan!(true),
            ty: Some(zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        assert_eq!(env.translate_let(&let_expr), Ok(Rc::new(Type::Unit)));
        assert_eq!(
            env.vars["x"],
            VarProperties {
                ty: Rc::new(Type::Int),
                immutable: true
            }
        );
        assert!(env.var_def_spans.contains_key("x"));
    }

    #[test]
    fn test_translate_let_type_annotation_err() {
        let mut env = Env::default();
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            immutable: zspan!(true),
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
                immutable: true,
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
                immutable: true,
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
                ty: Rc::new(Type::Fn(vec![Rc::new(Type::Int)], Rc::new(Type::Int))),
                immutable: true,
            },
            zspan!(),
        );
        env.fn_param_decl_spans.insert("f".to_owned(), vec![zspan!()]);

        let fn_call = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![zspan!(ExprType::String("a".to_owned()))],
        });

        assert_eq!(
            env.translate_fn_call(&fn_call).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        )
    }

    #[test]
    fn test_translate_fn_call_returns_aliased_type() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("int".to_owned()), zspan!());
        env.insert_var(
            "f".to_owned(),
            VarProperties {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Alias("a".to_owned())))),
                immutable: true,
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
            ty: zspan!(ast::TypeDeclType::Type(zspan!("int".to_owned()))),
        });
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::Wildcard),
            immutable: zspan!(true),
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
            ty: zspan!(ast::TypeDeclType::Type(zspan!("int".to_owned()))),
        });
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::Wildcard),
            immutable: zspan!(true),
            ty: Some(zspan!(ast::Type::Type(zspan!("a".to_owned())))),
            expr: zspan!(ExprType::String("".to_owned())),
        });
        let _ = env.translate_type_decl(&type_decl);
        assert_eq!(
            env.translate_let(&let_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Alias("a".to_owned())), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_translate_expr_typedef() {
        let mut env = Env::default();
        let type_decl = zspan!(TypeDecl {
            id: zspan!("i".to_owned()),
            ty: zspan!(ast::TypeDeclType::Type(zspan!("int".to_owned())))
        });
        let var_def = zspan!(Let {
            pattern: zspan!(Pattern::String("i".to_owned())),
            immutable: zspan!(true),
            ty: Some(zspan!(ast::Type::Type(zspan!("i".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        let expr = zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))));
        env.translate_type_decl(&type_decl).expect("translate type decl");
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
            ty: zspan!(ast::TypeDeclType::Record(vec![zspan!(TypeField {
                id: zspan!("i".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("i".to_owned())))
            })]))
        })));
        env.translate_decl_first_pass(&type_decl).expect("translate decl");
    }

    #[test]
    fn test_check_for_type_decl_cycles_err1() {
        let mut env = Env::default();
        env.insert_type("a".to_owned(), Type::Alias("a".to_owned()), zspan!());
        assert_eq!(
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0].t.ty,
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
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0].t.ty,
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
            ty: zspan!(ast::TypeDeclType::Unit)
        }))
        .expect("translate type decl");

        assert_eq!(
            env.translate_type_decl(&zspan!(ast::TypeDecl {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::TypeDeclType::Unit)
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
                immutable: true,
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
        let result = env.first_pass(&[
            Decl::new(
                DeclType::Type(zspan!(TypeDecl {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::TypeDeclType::Type(zspan!("a".to_owned())))
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
                TypecheckErrType::TypeDeclCycle("a".to_owned(), Rc::new(Type::Alias("a".to_owned()))),
                zspan!()
            )])
        );
    }

    #[test]
    fn test_translate_fn_decl_duplicate() {
        let mut env = Env::default();
        let fn_decl1 = zspan!(FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![],
            return_type: None,
            body: zspan!(ExprType::Unit)
        });
        let fn_decl2 = zspan!(FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![],
            return_type: Some(zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            body: zspan!(ExprType::Number(0))
        });
        env.translate_fn_decl_sig(&fn_decl1)
            .expect("translate function signature");

        assert_eq!(
            env.translate_fn_decl_sig(&fn_decl2).unwrap_err()[0].t.ty,
            TypecheckErrType::DuplicateFn("f".to_owned())
        );
    }

    #[test]
    fn test_translate_fn_decl_duplicate_param() {
        let mut env = Env::default();
        let fn_decl = zspan!(FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![
                zspan!(TypeField {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
                }),
                zspan!(TypeField {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
                }),
            ],
            return_type: None,
            body: zspan!(ExprType::Unit)
        });

        assert_eq!(
            env.translate_fn_decl_sig(&fn_decl).unwrap_err()[0].t.ty,
            TypecheckErrType::DuplicateParam("a".to_owned())
        );
    }

    #[test]
    fn test_validate_type() {
        let mut env = Env::default();
        let mut record_fields = HashMap::new();
        record_fields.insert("f".to_owned(), Rc::new(Type::Alias("a".to_owned())));
        record_fields.insert("g".to_owned(), Rc::new(Type::Alias("b".to_owned())));
        env.insert_type("a".to_owned(), Type::Record("a".to_owned(), record_fields), zspan!());

        let errs = env.check_for_invalid_types().unwrap_err();
        assert_eq!(errs.len(), 1);
        // Recursive type def in records is allowed
        assert_eq!(errs[0].t.ty, TypecheckErrType::UndefinedType("b".to_owned()));
    }

    #[test]
    fn test_translate_record_missing_fields() {
        let mut env = Env::default();
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => Rc::new(Type::Int),
            },
        );
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
        let record_type = Type::Record("r".to_owned(), HashMap::new());
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
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => Rc::new(Type::Int),
            },
        );

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
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => Rc::new(Type::Int),
                "b".to_owned() => Rc::new(Type::String),
            },
        );
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
    fn test_translate_record_field_type_mismatch() {
        let mut env = Env::default();
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => Rc::new(Type::Int)
            },
        );
        env.insert_type("r".to_owned(), record_type, zspan!());
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });
        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![zspan!(FieldAssign {
                id: zspan!("a".to_owned()),
                expr: zspan!(ExprType::String("asdf".to_owned()))
            })]
        });

        assert_eq!(
            env.translate_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_translate_fn_independent() {
        let mut env = Env::default();
        let fn_expr1 = zspan!(ExprType::Seq(
            vec![zspan!(ExprType::Let(Box::new(zspan!(Let {
                pattern: zspan!(Pattern::String("a".to_owned())),
                immutable: zspan!(true),
                ty: None,
                expr: zspan!(ExprType::Number(0))
            }))))],
            false
        ));
        let fn_expr2 = zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("a".to_owned())))));

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
    fn test_translate_fn_body() {
        let mut env = Env::default();
        let fn_expr = zspan!(ExprType::Arith(Box::new(zspan!(Arith {
            l: zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("a".to_owned()))))),
            op: zspan!(ast::ArithOp::Add),
            r: zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("b".to_owned())))))
        }))));
        let fn_decl = zspan!(DeclType::Fn(zspan!(FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![
                zspan!(TypeField {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("int".to_owned())))
                }),
                zspan!(TypeField {
                    id: zspan!("b".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("int".to_owned())))
                })
            ],
            return_type: Some(zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            body: fn_expr,
        })));

        assert_eq!(env.translate_decls(&[fn_decl]), Ok(()));
    }

    #[test]
    fn test_translate_seq_independent() {
        let env = Env::default();
        let seq_expr1 = zspan!(ExprType::Seq(
            vec![zspan!(ExprType::Let(Box::new(zspan!(Let {
                pattern: zspan!(Pattern::String("a".to_owned())),
                immutable: zspan!(true),
                ty: None,
                expr: zspan!(ExprType::Number(0))
            }))))],
            false
        ));
        let seq_expr2 = zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("a".to_owned())))));

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
            immutable: zspan!(true),
            ty: None,
            expr: zspan!(ExprType::Let(Box::new(zspan!(Let {
                pattern: zspan!(Pattern::Wildcard),
                immutable: zspan!(true),
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
                immutable: true,
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

    #[test]
    fn test_assign_record_field_immut_err() {
        let mut env = Env::default();
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => Rc::new(Type::Int)
            },
        );
        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        env.insert_var(
            "r".to_owned(),
            VarProperties {
                ty: Rc::new(record_type),
                immutable: true,
            },
            zspan!(),
        );

        assert_eq!(
            env.translate_assign(&zspan!(Assign {
                lval: zspan!(LVal::Field(
                    Box::new(zspan!(LVal::Simple("r".to_owned()))),
                    zspan!("a".to_owned())
                )),
                expr: zspan!(ExprType::Number(0))
            }))
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::MutatingImmutable("r".to_owned())
        );
    }

    #[test]
    fn test_assign_record_field_type_mismatch() {
        let mut env = Env::default();
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => Rc::new(Type::Int)
            },
        );
        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        env.insert_var(
            "r".to_owned(),
            VarProperties {
                ty: Rc::new(record_type),
                immutable: false,
            },
            zspan!(),
        );

        assert_eq!(
            env.translate_assign(&zspan!(Assign {
                lval: zspan!(LVal::Field(
                    Box::new(zspan!(LVal::Simple("r".to_owned()))),
                    zspan!("a".to_owned())
                )),
                expr: zspan!(ExprType::Unit)
            }))
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_assign_record_field_type() {
        let mut env = Env::default();
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => Rc::new(Type::Int)
            },
        );
        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        env.insert_var(
            "r".to_owned(),
            VarProperties {
                ty: Rc::new(record_type),
                immutable: false,
            },
            zspan!(),
        );

        assert_eq!(
            env.translate_assign(&zspan!(Assign {
                lval: zspan!(LVal::Field(
                    Box::new(zspan!(LVal::Simple("r".to_owned()))),
                    zspan!("a".to_owned())
                )),
                expr: zspan!(ExprType::Number(0))
            })),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_assign_array_immut_err() {
        let mut env = Env::default();
        let array_type = Type::Array(Rc::new(Type::Int), 1);
        env.insert_var(
            "a".to_owned(),
            VarProperties {
                ty: Rc::new(array_type),
                immutable: true,
            },
            zspan!(),
        );

        assert_eq!(
            env.translate_assign(&zspan!(Assign {
                lval: zspan!(LVal::Subscript(
                    Box::new(zspan!(LVal::Simple("a".to_owned()))),
                    zspan!(ExprType::Number(0))
                )),
                expr: zspan!(ExprType::Number(0))
            }))
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::MutatingImmutable("a".to_owned())
        );
    }

    #[test]
    fn test_assign_array_type_mismatch() {
        let mut env = Env::default();
        let array_type = Type::Array(Rc::new(Type::Int), 1);
        env.insert_var(
            "a".to_owned(),
            VarProperties {
                ty: Rc::new(array_type),
                immutable: false,
            },
            zspan!(),
        );

        assert_eq!(
            env.translate_assign(&zspan!(Assign {
                lval: zspan!(LVal::Subscript(
                    Box::new(zspan!(LVal::Simple("a".to_owned()))),
                    zspan!(ExprType::Number(0))
                )),
                expr: zspan!(ExprType::String("s".to_owned()))
            }))
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_assign_array() {
        let mut env = Env::default();
        let array_type = Type::Array(Rc::new(Type::Int), 1);
        env.insert_var(
            "a".to_owned(),
            VarProperties {
                ty: Rc::new(array_type),
                immutable: false,
            },
            zspan!(),
        );

        assert_eq!(
            env.translate_assign(&zspan!(Assign {
                lval: zspan!(LVal::Subscript(
                    Box::new(zspan!(LVal::Simple("a".to_owned()))),
                    zspan!(ExprType::Number(0))
                )),
                expr: zspan!(ExprType::Number(0))
            })),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_translate_array() {
        let env = Env::default();
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Number(3))
        });

        assert_eq!(
            env.translate_array(&array_expr),
            Ok(Rc::new(Type::Array(Rc::new(Type::Int), 3)))
        );
    }

    #[test]
    fn test_translate_array_const_expr_len() {
        let env = Env::default();
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Arith(Box::new(zspan!(Arith {
                l: zspan!(ExprType::Number(1)),
                op: zspan!(ArithOp::Add),
                r: zspan!(ExprType::Number(2)),
            }))))
        });

        assert_eq!(
            env.translate_array(&array_expr),
            Ok(Rc::new(Type::Array(Rc::new(Type::Int), 3)))
        );
    }

    #[test]
    fn test_translate_array_negative_len_err() {
        let env = Env::default();
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Neg(Box::new(zspan!(ExprType::Number(3)))))
        });

        assert_eq!(
            env.translate_array(&array_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::NegativeArrayLen(-3)
        );
    }

    #[test]
    fn test_translate_array_non_constant_expr_err() {
        let env = Env::default();
        let fn_call = ExprType::FnCall(Box::new(zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![]
        })));
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(fn_call.clone())
        });

        assert_eq!(
            env.translate_array(&array_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::NonConstantArithExpr(fn_call)
        );
    }

    #[test]
    fn test_translate_if_then() {
        let env = Env::default();
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Unit),
            else_expr: None,
        });

        assert_eq!(env.translate_if(&if_expr), Ok(Rc::new(Type::Unit)));
    }

    #[test]
    fn test_translate_if_then_type_mismatch() {
        let env = Env::default();
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: None,
        });

        assert_eq!(
            env.translate_if(&if_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_translate_if_then_else() {
        let env = Env::default();
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: Some(zspan!(ExprType::Number(1))),
        });

        assert_eq!(env.translate_if(&if_expr), Ok(Rc::new(Type::Int)));
    }

    #[test]
    fn test_translate_if_then_else_type_mismatch() {
        let env = Env::default();
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: Some(zspan!(ExprType::String("s".to_owned()))),
        });

        assert_eq!(
            env.translate_if(&if_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_translate_range() {
        let env = Env::default();
        let range = zspan!(Range {
            start: zspan!(ExprType::Number(0)),
            end: zspan!(ExprType::Number(1)),
        });

        assert_eq!(
            env.translate_range(&range),
            Ok(Rc::new(Type::Iterator(Rc::new(Type::Int))))
        );
    }

    #[test]
    fn test_translate_range_type_mismatch() {
        let env = Env::default();
        let range = zspan!(Range {
            start: zspan!(ExprType::Number(0)),
            end: zspan!(ExprType::String("a".to_owned())),
        });

        assert_eq!(
            env.translate_range(&range).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_translate_for() {
        let env = Env::default();
        let for_expr = zspan!(For {
            index: zspan!("i".to_owned()),
            range: zspan!(ExprType::Range(Box::new(zspan!(Range {
                start: zspan!(ExprType::Number(0)),
                end: zspan!(ExprType::Number(1)),
            })))),
            body: zspan!(ExprType::Seq(
                vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))))],
                false
            ))
        });

        assert_eq!(env.translate_for(&for_expr), Ok(Rc::new(Type::Unit)));
    }

    #[test]
    fn test_translate_for_range_type_mismatch() {
        let env = Env::default();
        let for_expr = zspan!(For {
            index: zspan!("i".to_owned()),
            range: zspan!(ExprType::Number(0)),
            body: zspan!(ExprType::Seq(
                vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))))],
                false
            ))
        });

        assert_eq!(
            env.translate_for(&for_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Iterator(Rc::new(Type::Int))), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_translate_for_body_type_mismatch() {
        let env = Env::default();
        let for_expr = zspan!(For {
            index: zspan!("i".to_owned()),
            range: zspan!(ExprType::Range(Box::new(zspan!(Range {
                start: zspan!(ExprType::Number(0)),
                end: zspan!(ExprType::Number(1)),
            })))),
            body: zspan!(ExprType::Seq(
                vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))))],
                true
            ))
        });

        assert_eq!(
            env.translate_for(&for_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_translate_while() {
        let env = Env::default();
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::BoolLiteral(true)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Unit)], false))
        });

        assert_eq!(env.translate_while(&while_expr), Ok(Rc::new(Type::Unit)));
    }

    #[test]
    fn test_translate_while_cond_type_mismatch() {
        let env = Env::default();
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::Number(0)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Unit)], false))
        });
        assert_eq!(
            env.translate_while(&while_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Bool), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_translate_while_body_type_mismatch() {
        let env = Env::default();
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::BoolLiteral(true)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Number(0))], true))
        });

        assert_eq!(
            env.translate_while(&while_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_translate_compare() {
        let env = Env::default();
        let compare = zspan!(Compare {
            l: zspan!(ExprType::BoolLiteral(true)),
            op: zspan!(CompareOp::Equal),
            r: zspan!(ExprType::BoolLiteral(true)),
        });

        assert_eq!(env.translate_compare(&compare), Ok(Rc::new(Type::Bool)));
    }

    #[test]
    fn test_translate_compare_type_mismatch() {
        let env = Env::default();
        let compare = zspan!(Compare {
            l: zspan!(ExprType::Number(0)),
            op: zspan!(CompareOp::Equal),
            r: zspan!(ExprType::BoolLiteral(true)),
        });

        assert_eq!(
            env.translate_compare(&compare).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Bool), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_translate_enum_arity_mismatch() {
        let mut env = Env::default();
        env.insert_type(
            "e".to_owned(),
            Type::Enum(
                "e".to_owned(),
                hashmap! {
                    "c".to_owned() => EnumCase {
                        id: "c".to_owned(),
                        params: vec![Rc::new(Type::Int)]
                    }
                },
            ),
            zspan!(),
        );

        let enum_expr = zspan!(Enum {
            enum_id: zspan!("e".to_owned()),
            case_id: zspan!("c".to_owned()),
            args: zspan!(vec![])
        });

        assert_eq!(
            env.translate_enum(&enum_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::ArityMismatch(1, 0)
        );
    }

    #[test]
    fn test_translate_enum_type_mismatch() {
        let mut env = Env::default();
        env.insert_type(
            "e".to_owned(),
            Type::Enum(
                "e".to_owned(),
                hashmap! {
                    "c".to_owned() => EnumCase {
                        id: "c".to_owned(),
                        params: vec![Rc::new(Type::Int)]
                    }
                },
            ),
            zspan!(),
        );

        let enum_expr = zspan!(Enum {
            enum_id: zspan!("e".to_owned()),
            case_id: zspan!("c".to_owned()),
            args: zspan!(vec![zspan!(ExprType::String("a".to_owned()))])
        });

        assert_eq!(
            env.translate_enum(&enum_expr).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }
}
