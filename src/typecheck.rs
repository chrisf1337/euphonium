use crate::{
    ast::{
        Arith, ArithOp, Array, Assign, Bool, Closure, Compare, Decl, DeclType, Enum, Expr, ExprType, FieldAssign,
        FileSpan, FnCall, FnDecl, For, If, LVal, Let, Pattern, Range, Record, Spanned, TypeDecl, TypeDeclType, While,
    },
    fragment::{FnFragment, Fragment},
    frame::Frame,
    ir,
    log::TYPECHECK_LOG,
    tmp::{self, Label, TmpGenerator},
    translate::{self, Access, Level},
    ty::{EnumCase, RecordField, Type, TypeInfo, _RecordField, _Type},
};
use codespan::{FileId, Span};
use codespan_reporting;
use itertools::izip;
use maplit::hashmap;
use slog::warn;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
};

pub type Result<T> = std::result::Result<T, Vec<TypecheckError>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslateOutput<T> {
    pub t: T,
    pub expr: translate::Expr,
}

impl<T> TranslateOutput<T> {
    pub fn map<F, U>(self, f: F) -> TranslateOutput<U>
    where
        F: FnOnce(T) -> U,
    {
        TranslateOutput {
            t: f(self.t),
            expr: self.expr,
        }
    }

    pub fn map_expr<F>(self, f: F) -> TranslateOutput<T>
    where
        F: FnOnce(translate::Expr) -> translate::Expr,
    {
        TranslateOutput {
            t: self.t,
            expr: f(self.expr),
        }
    }

    pub fn unwrap(self) -> T {
        self.t
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypecheckErrorType {
    /// expected, actual
    TypeMismatch(TypeInfo, TypeInfo),
    // The reason we need a separate case for this instead of using TypeMismatch is because we would
    // need to typecheck the arguments passed to the non-function in order to determine the expected
    // function type.
    ArityMismatch(usize, usize),
    NotAFn(String),
    NotARecord(TypeInfo),
    NotAnArray(TypeInfo),
    NotAnEnum(TypeInfo),
    NotAnEnumCase(String),
    NotARangeLiteral,
    UndefinedVar(String),
    UndefinedFn(String),
    UndefinedField(String),
    UndefinedType(String),
    CannotSubscript,
    /// Type triggering the cycle, what it was aliased to
    TypeDeclCycle(String, Rc<_Type>),
    MissingFields(Vec<String>),
    InvalidFields(Vec<String>),
    IllegalLetExpr,
    IllegalFnDeclExpr,
    DuplicateFn(String),
    DuplicateType(String),
    DuplicateField(String),
    DuplicateEnumCase(String),
    DuplicateParam(String),
    MutatingImmutable(String),
    NonConstantArithExpr(ExprType),
    NegativeArrayLen(i64),
    IllegalNestedEnumDecl,
    IllegalBreakOrContinue,
}

pub type TypecheckError = Spanned<_TypecheckError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypecheckError {
    pub ty: TypecheckErrorType,
    pub secondary_messages: Vec<Spanned<String>>,
}

impl TypecheckError {
    fn new_err(ty: TypecheckErrorType, span: FileSpan) -> Self {
        TypecheckError::new(
            _TypecheckError {
                ty,
                secondary_messages: vec![],
            },
            span,
        )
    }

    fn with_secondary_messages(mut self, messages: impl IntoIterator<Item = Spanned<String>>) -> Self {
        self.secondary_messages = messages.into_iter().collect();
        self
    }

    pub fn diagnostic(&self, env: &Env) -> codespan_reporting::diagnostic::Diagnostic {
        use TypecheckErrorType::*;
        let msg = match &self.ty {
            TypeMismatch(expected, actual) => {
                format!("type mismatch\nexpected: {:?}\n  actual: {:?}", expected, actual)
            }
            ArityMismatch(expected, actual) => {
                format!("arity mismatch\nexpected: {:?}\n  actual: {:?}", expected, actual)
            }
            NotAFn(fun) => {
                let ty = env.var(fun).unwrap();
                format!("not a function: {} (has type {:?})", fun, ty)
            }
            NotARecord(ty) => format!("not a record (has type {:?})", ty.ty().as_ref()),
            NotAnArray(ty) => format!("not an array (has type {:?})", ty.ty().as_ref()),
            NotAnEnum(ty) => format!("not an enum (has type {:?})", ty.ty().as_ref()),
            NotAnEnumCase(case_id) => format!("not an enum case: {}", case_id),
            NotARangeLiteral => "not a range literal".to_owned(),
            UndefinedVar(var) => format!("undefined variable: {}", var),
            UndefinedFn(fun) => format!("undefined function: {}", fun),
            UndefinedField(field) => format!("undefined field: {}", field),
            UndefinedType(ty) => format!("undefined type: {:?}", ty),
            CannotSubscript => "cannot subscript".to_owned(),
            TypeDeclCycle(..) => "cycle in type declaration".to_owned(),
            MissingFields(fields) => format!("missing fields: {}", fields.join(", ")),
            InvalidFields(fields) => format!("invalid fields: {}", fields.join(", ")),
            IllegalLetExpr => "a let expression cannot be used here".to_owned(),
            IllegalFnDeclExpr => "a function cannot be declared here".to_owned(),
            DuplicateFn(fun) => format!("duplicate function declaration for {}", fun),
            DuplicateType(ty) => format!("duplicate type declaration for {}", ty),
            DuplicateField(field) => format!("duplicate record field declaration for {}", field),
            DuplicateEnumCase(enum_case) => format!("duplicate enum case declaration for {}", enum_case),
            DuplicateParam(param) => format!("duplicate function param declaration for {}", param),
            MutatingImmutable(var) => format!("{} was declared as immutable", var),
            NonConstantArithExpr(expr) => format!("{:?} is not a constant arithmetic expression", expr),
            NegativeArrayLen(..) => "array length cannot be negative".to_owned(),
            IllegalNestedEnumDecl => "enum declarations cannot be nested".to_owned(),
            IllegalBreakOrContinue => "break and continue can only be used inside for and while loops".to_owned(),
        };

        let primary_label = codespan_reporting::diagnostic::Label::new(self.span.file_id, self.span.span, &msg);
        let mut diagnostic = codespan_reporting::diagnostic::Diagnostic::new_error(&msg, primary_label);
        if !self.secondary_messages.is_empty() {
            diagnostic = diagnostic.with_secondary_labels(
                self.secondary_messages
                    .iter()
                    .map(|msg| codespan_reporting::diagnostic::Label::new(msg.span.file_id, msg.span.span, &msg.t))
                    .collect::<Vec<codespan_reporting::diagnostic::Label>>(),
            );
        }
        diagnostic
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EnvEntryType {
    Var(Access),
    Fn(Label),
}

impl EnvEntryType {
    fn fn_label(&self) -> Label {
        if let EnvEntryType::Fn(label) = self {
            label.clone()
        } else {
            panic!("env entry type is not Fn");
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EnvEntry {
    ty: TypeInfo,
    immutable: bool,
    entry_type: EnvEntryType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LValProperties {
    ty: TypeInfo,
    /// `None` if mutable, or `Some(var)` if immutable, with `var` being the root immutable variable.
    ///
    /// For example, if `a` is an immutable record with a field named `b`, then the `LValProperties`
    /// for `LVal::Field(a, b)` would have `Some("a")` for its `immutable` field.
    immutable: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Env<'a> {
    tmp_generator: TmpGenerator,
    pub file_id: FileId,
    parent: Option<&'a Env<'a>>,
    vars: HashMap<String, EnvEntry>,
    types: HashMap<String, TypeInfo>,
    pre_types: HashMap<String, Rc<_Type>>,
    var_def_spans: HashMap<String, FileSpan>,
    type_def_spans: HashMap<String, FileSpan>,

    record_field_decl_spans: HashMap<String, HashMap<String, FileSpan>>,
    fn_param_decl_spans: HashMap<String, Vec<FileSpan>>,
    enum_case_param_decl_spans: HashMap<String, HashMap<String, Vec<FileSpan>>>,

    pub(crate) levels: Rc<RefCell<HashMap<Label, Level>>>,
    pub(crate) fragments: Rc<RefCell<Vec<Fragment>>>,

    level_label: Label,
    break_label: Option<Label>,
    continue_label: Option<Label>,
}

impl<'a> Env<'a> {
    pub fn new(tmp_generator: TmpGenerator, file_id: FileId, level_label: Label) -> Self {
        let mut pre_types = HashMap::new();
        pre_types.insert("int".to_owned(), Rc::new(_Type::Int));
        pre_types.insert("string".to_owned(), Rc::new(_Type::String));
        let mut type_def_spans = HashMap::new();
        type_def_spans.insert("int".to_owned(), FileSpan::new(file_id, Span::initial()));
        type_def_spans.insert("string".to_owned(), FileSpan::new(file_id, Span::initial()));

        Env {
            tmp_generator,
            file_id,
            parent: None,
            vars: HashMap::new(),
            types: HashMap::new(),
            pre_types,
            var_def_spans: HashMap::new(),
            type_def_spans,

            record_field_decl_spans: HashMap::new(),
            fn_param_decl_spans: HashMap::new(),
            enum_case_param_decl_spans: HashMap::new(),
            levels: Rc::new(RefCell::new(hashmap! {
                Label::top() => Level::top(),
            })),
            fragments: Rc::new(RefCell::new(vec![])),

            level_label,
            break_label: None,
            continue_label: None,
        }
    }

    fn new_child(&'a self, level_label: Label) -> Env<'a> {
        Env {
            tmp_generator: self.tmp_generator.clone(),
            file_id: self.file_id,
            parent: Some(self),
            vars: HashMap::new(),
            types: HashMap::new(),
            pre_types: HashMap::new(),
            var_def_spans: HashMap::new(),
            type_def_spans: HashMap::new(),

            record_field_decl_spans: HashMap::new(),
            fn_param_decl_spans: HashMap::new(),
            enum_case_param_decl_spans: HashMap::new(),

            levels: self.levels.clone(),
            fragments: self.fragments.clone(),

            level_label: level_label.clone(),
            break_label: if level_label == self.level_label {
                self.break_label.clone()
            } else {
                None
            },
            continue_label: if level_label == self.level_label {
                self.continue_label.clone()
            } else {
                None
            },
        }
    }

    fn insert_var(&mut self, name: impl Into<String>, entry: EnvEntry, def_span: FileSpan) {
        let name = name.into();
        self.vars.insert(name.clone(), entry);
        self.var_def_spans.insert(name, def_span);
    }

    fn insert_pre_type(&mut self, name: impl Into<String>, ty: _Type, def_span: FileSpan) {
        let name = name.into();
        self.pre_types.insert(name.clone(), Rc::new(ty));
        self.type_def_spans.insert(name, def_span);
    }

    #[cfg(test)]
    fn insert_type(&mut self, name: impl Into<String>, ty: TypeInfo, def_span: FileSpan) {
        let name = name.into();
        self.types.insert(name.clone(), ty);
        self.type_def_spans.insert(name, def_span);
    }

    fn var(&self, name: &str) -> Option<&EnvEntry> {
        if let Some(var) = self.vars.get(name) {
            Some(var)
        } else if let Some(parent) = self.parent {
            parent.var(name)
        } else {
            None
        }
    }

    fn pre_type(&self, name: &str) -> Option<&Rc<_Type>> {
        if let Some(ty) = self.pre_types.get(name) {
            Some(ty)
        } else if let Some(parent) = self.parent {
            parent.pre_type(name)
        } else {
            None
        }
    }

    fn r#type(&self, name: &str) -> Option<&TypeInfo> {
        if let Some(ty) = self.types.get(name) {
            Some(ty)
        } else if let Some(parent) = self.parent {
            parent.r#type(name)
        } else {
            None
        }
    }

    fn var_def_span(&self, id: &str) -> Option<FileSpan> {
        if let Some(span) = self.var_def_spans.get(id) {
            Some(*span)
        } else if let Some(parent) = self.parent {
            parent.var_def_span(id)
        } else {
            None
        }
    }

    fn type_def_span(&self, id: &str) -> Option<FileSpan> {
        if let Some(span) = self.type_def_spans.get(id) {
            Some(*span)
        } else if let Some(parent) = self.parent {
            parent.type_def_span(id)
        } else {
            None
        }
    }

    fn record_field_decl_spans(&self, id: &str) -> Option<&HashMap<String, FileSpan>> {
        if let Some(spans) = self.record_field_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.record_field_decl_spans(id)
        } else {
            None
        }
    }

    fn fn_param_decl_spans(&self, id: &str) -> Option<&[FileSpan]> {
        if let Some(spans) = self.fn_param_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.fn_param_decl_spans(id)
        } else {
            None
        }
    }

    fn enum_case_param_decl_spans(&self, id: &str) -> Option<&HashMap<String, Vec<FileSpan>>> {
        if let Some(spans) = self.enum_case_param_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.enum_case_param_decl_spans(id)
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
    fn resolve_pre_type(&self, ty: &Rc<_Type>, def_span: FileSpan) -> Result<Rc<_Type>> {
        match ty.as_ref() {
            _Type::Alias(alias) => {
                if let Some(resolved_type) = self.pre_type(alias) {
                    let span = self.type_def_span(alias).unwrap();
                    self.resolve_pre_type(resolved_type, span)
                } else {
                    Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::UndefinedType(alias.clone()),
                        def_span,
                    )])
                }
            }
            _Type::Array(elem_type, len) => {
                Ok(Rc::new(_Type::Array(self.resolve_pre_type(elem_type, def_span)?, *len)))
            }
            _ => Ok(ty.clone()),
        }
    }

    fn resolve_type(&self, ty: &TypeInfo, def_span: FileSpan) -> Result<TypeInfo> {
        match ty.ty().as_ref() {
            Type::Alias(alias) => {
                if let Some(resolved_type) = self.r#type(alias) {
                    let span = self.type_def_span(alias).unwrap();
                    self.resolve_type(resolved_type, span)
                } else {
                    Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::UndefinedType(alias.clone()),
                        def_span,
                    )])
                }
            }
            Type::Array(elem_type, len) => Ok(TypeInfo::new(
                Rc::new(Type::Array(self.resolve_type(elem_type, def_span)?, *len)),
                ty.size(),
            )),
            _ => Ok(ty.clone()),
        }
    }

    /// `ty` must have already been resolved.
    fn assert_ty(&self, expr: &Expr, ty: &TypeInfo) -> Result<TranslateOutput<TypeInfo>> {
        let translate_output = self.typecheck_expr(expr)?;
        let expr_type = &translate_output.t;
        let resolved_expr_type = self.resolve_type(expr_type, expr.span)?;
        if ty != &resolved_expr_type {
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::TypeMismatch(ty.clone(), expr_type.clone()),
                expr.span,
            )])
        } else {
            Ok(translate_output)
        }
    }

    fn check_for_type_decl_cycles(&self, ty: &str, path: Vec<&str>) -> Result<()> {
        if let Some(alias) = self.pre_types.get(ty) {
            if let Some(alias_str) = alias.alias() {
                if path.contains(&alias_str) {
                    let span = self.type_def_span(ty).unwrap();
                    return Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::TypeDeclCycle(ty.to_string(), alias.clone()),
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
    fn validate_pre_type(&self, ty: &Rc<_Type>, def_span: FileSpan) -> Result<()> {
        match ty.as_ref() {
            _Type::String | _Type::Bool | _Type::Unit | _Type::Int | _Type::Iterator(_) => Ok(()),
            _Type::Alias(_) => self.resolve_pre_type(ty, def_span).map(|_| ()),
            _Type::Record(_, record) => {
                let mut errors = vec![];
                for (_, _RecordField { ty, .. }) in record {
                    match self.validate_pre_type(ty, def_span) {
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
            _Type::Array(ty, _) => self.validate_pre_type(ty, def_span),
            _Type::Enum(_, cases) => {
                let mut errors = vec![];
                for case in cases.values() {
                    for param in &case.params {
                        // Don't allow nested enum decls
                        if let _Type::Enum(..) = param.as_ref() {
                            errors.push(TypecheckError::new_err(
                                TypecheckErrorType::IllegalNestedEnumDecl,
                                def_span,
                            ));
                            continue;
                        }
                        match self.validate_pre_type(param, def_span) {
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
            _Type::Fn(param_types, return_type) => {
                let mut errors = vec![];

                for param_type in param_types {
                    match self.validate_pre_type(param_type, def_span) {
                        Ok(()) => (),
                        Err(errs) => errors.extend(errs),
                    }
                }

                match self.validate_pre_type(return_type, def_span) {
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

    /// Once we have typechecked all type decls, check if pre-types are all valid. This is
    /// essentially making sure that aliases resolve into actual types and enum decls are not nested.
    fn check_for_invalid_pre_types(&self) -> Result<()> {
        let mut errors = vec![];
        for (id, ty) in &self.pre_types {
            match self.validate_pre_type(ty, self.type_def_span(id).unwrap()) {
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

    fn check_for_duplicates<'b, I, T: 'b, Key: Hash + Eq, KeyFn, ErrGen>(
        iter: I,
        key_fn: KeyFn,
        err_gen: ErrGen,
    ) -> Result<HashMap<Key, FileSpan>>
    where
        I: IntoIterator<Item = &'b Spanned<T>>,
        KeyFn: Fn(&'b Spanned<T>) -> Key,
        ErrGen: Fn(&'b Spanned<T>, FileSpan) -> TypecheckError,
    {
        // t -> def span
        let mut checked_elems: HashMap<Key, FileSpan> = HashMap::new();
        let mut errors: Vec<TypecheckError> = vec![];
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

    pub fn typecheck_decls(&mut self, file_id: FileId, decls: &[Decl]) -> Result<()> {
        self.file_id = file_id;
        self.first_pass(decls)?;
        self.second_pass(decls)
    }

    fn size_of(&self, ty: &Rc<_Type>) -> usize {
        match ty.as_ref() {
            _Type::Int => std::mem::size_of::<i64>(),
            _Type::String => std::mem::size_of::<u64>(),
            _Type::Bool => std::mem::size_of::<bool>(),
            _Type::Record(_, fields) => fields.iter().map(|(_, f)| self.size_of(&f.ty)).sum(),
            _Type::Array(elem_type, length) => self.size_of(elem_type) * length,
            _Type::Unit => 0,
            _Type::Alias(_) => {
                let resolved_type = self
                    .resolve_pre_type(ty, FileSpan::new(self.file_id, Span::initial()))
                    .unwrap();
                self.size_of(&resolved_type)
            }
            _Type::Enum(_, cases) => cases
                .values()
                .map(|c| c.params.iter().map(|p| self.size_of(p)).sum::<usize>())
                .max()
                .unwrap_or(0),
            _Type::Fn(..) => std::mem::size_of::<u64>(),
            _Type::Iterator(ty) => self.size_of(ty) * 2,
        }
    }

    /// This must be called after checking for type decl cycles; otherwise, the compiler will
    /// infinitely loop trying to resolve types.
    fn convert_pre_type(&self, ty: &Rc<_Type>) -> TypeInfo {
        match ty.as_ref() {
            _Type::Int => TypeInfo::int(),
            _Type::String => TypeInfo::string(),
            _Type::Bool => TypeInfo::bool(),
            _Type::Record(id, fields) => {
                let mut record_fields = HashMap::new();
                let mut offset = 0;
                for (id, field) in fields {
                    let ty = self.convert_pre_type(&field.ty);
                    let size = ty.size();
                    record_fields.insert(id.clone(), RecordField { ty, offset });
                    offset += size;
                }
                TypeInfo::new(Rc::new(Type::Record(id.clone(), record_fields)), self.size_of(ty))
            }
            _Type::Array(elem_ty, len) => {
                let elem_ty = self.convert_pre_type(elem_ty);
                TypeInfo::new(Rc::new(Type::Array(elem_ty, *len)), self.size_of(ty))
            }
            _Type::Unit => TypeInfo::unit(),
            _Type::Alias(alias) => TypeInfo::new(Rc::new(Type::Alias(alias.clone())), self.size_of(ty)),
            _Type::Enum(id, cases) => TypeInfo::new(
                Rc::new(Type::Enum(
                    id.clone(),
                    cases
                        .clone()
                        .into_iter()
                        .map(|(id, case)| {
                            (
                                id,
                                EnumCase {
                                    id: case.id.clone(),
                                    params: case.params.iter().map(|p| self.convert_pre_type(p)).collect(),
                                },
                            )
                        })
                        .collect(),
                )),
                self.size_of(ty),
            ),
            _Type::Fn(param_types, return_type) => TypeInfo::new(
                Rc::new(Type::Fn(
                    param_types.iter().map(|ty| self.convert_pre_type(ty)).collect(),
                    self.convert_pre_type(return_type),
                )),
                self.size_of(ty),
            ),
            _Type::Iterator(ty) => TypeInfo::new(Rc::new(Type::Iterator(self.convert_pre_type(ty))), self.size_of(ty)),
        }
    }

    // Converts pre-types into actual TypeInfos.
    pub(crate) fn convert_pre_types(&mut self) {
        self.types = self
            .pre_types
            .iter()
            .map(|(id, pre_type)| (id.clone(), self.convert_pre_type(pre_type)))
            .collect();
    }

    fn validate_fn_decl_pre_types(&self, fn_decl: &FnDecl) -> Result<()> {
        let mut errors = vec![];
        let param_pre_types: Vec<Rc<_Type>> = fn_decl
            .type_fields
            .iter()
            .map(|type_field| Rc::new(type_field.t.clone().ty.t.into()))
            .collect();
        for (i, param_pre_type) in param_pre_types.iter().enumerate() {
            match self.validate_pre_type(param_pre_type, fn_decl.type_fields[i].ty.span) {
                Ok(_) => (),
                Err(errs) => errors.extend(errs),
            }
        }

        if let Some(return_type) = &fn_decl.return_type {
            let return_pre_type = Rc::new(return_type.t.clone().into());
            match self.validate_pre_type(&return_pre_type, return_type.span) {
                Ok(_) => (),
                Err(errs) => errors.extend(errs),
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    fn first_pass(&mut self, decls: &[Decl]) -> Result<()> {
        let mut errors = vec![];
        let mut found_cycle = false;
        for decl in decls {
            match self.typecheck_type_decl_and_check_cycles(decl) {
                Ok(()) => (),
                Err(errs) => {
                    for err in &errs {
                        if let _TypecheckError {
                            ty: TypecheckErrorType::TypeDeclCycle(..),
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

        match self.check_for_invalid_pre_types() {
            Ok(()) => (),
            Err(errs) => errors.extend(errs),
        }

        if !errors.is_empty() {
            return Err(errors);
        }

        self.convert_pre_types();

        for decl in decls {
            if let DeclType::Fn(fn_decl) = &decl.t {
                self.validate_fn_decl_pre_types(fn_decl)?;
                self.typecheck_fn_decl_sig(fn_decl)?;
            }
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
                match self.typecheck_fn_decl_body(fn_decl) {
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

    fn typecheck_type_decl_and_check_cycles(&mut self, decl: &Decl) -> Result<()> {
        self.level_label = Label::top();
        match &decl.t {
            DeclType::Type(type_decl) => {
                self.typecheck_type_decl(type_decl)?;
                self.check_for_type_decl_cycles(&type_decl.id, vec![])
            }
            DeclType::Fn(_) => Ok(()),
            _ => unreachable!(), // DeclType::Error
        }
    }

    fn typecheck_fn_decl_sig(&mut self, fn_decl: &Spanned<FnDecl>) -> Result<TypeInfo> {
        // Check if there already exists another function with the same name
        if self.vars.contains_key(&fn_decl.id.t) {
            let span = self.var_def_span(&fn_decl.id.t).unwrap();
            return Err(vec![TypecheckError::new_err(
                TypecheckErrorType::DuplicateFn(fn_decl.id.t.clone()),
                fn_decl.id.span,
            )
            .with_secondary_messages(vec![Spanned::new(
                format!("{} was defined here", fn_decl.id.t.clone()),
                span,
            )])]);
        }

        let param_decl_spans = Self::check_for_duplicates(
            &fn_decl.type_fields,
            |type_field| &type_field.id.t,
            |type_field, span| {
                TypecheckError::new_err(
                    TypecheckErrorType::DuplicateParam(type_field.id.t.clone()),
                    type_field.span,
                )
                .with_secondary_messages(vec![Spanned::new(
                    format!("{} was declared here", type_field.id.t.clone()),
                    span,
                )])
            },
        )?
        .values()
        .cloned()
        .collect();
        self.fn_param_decl_spans.insert(fn_decl.id.t.clone(), param_decl_spans);

        let param_types: Vec<TypeInfo> = fn_decl
            .type_fields
            .iter()
            .map(|type_field| self.convert_pre_type(&Rc::new(type_field.ty.t.clone().into())))
            .collect();
        let return_type = if let Some(Spanned { t: return_type, .. }) = &fn_decl.return_type {
            self.convert_pre_type(&Rc::new(return_type.clone().into()))
        } else {
            TypeInfo::unit()
        };

        let ty = TypeInfo::new(
            Rc::new(Type::Fn(param_types.clone(), return_type)),
            std::mem::size_of::<u64>(),
        );
        // FIXME: Don't assume all formals escape
        let formals: Vec<(usize, bool)> = param_types.iter().map(|param_type| (param_type.size(), true)).collect();
        let level = Level::new(
            &self.tmp_generator,
            Some(self.level_label.clone()),
            &fn_decl.id.t,
            &formals,
        );

        let label = level.label().clone();
        {
            let mut levels = self.levels.borrow_mut();
            levels.insert(label.clone(), level);
        }

        self.insert_var(
            fn_decl.id.t.clone(),
            EnvEntry {
                ty: ty.clone(),
                immutable: true,
                entry_type: EnvEntryType::Fn(label),
            },
            fn_decl.span,
        );
        Ok(ty)
    }

    /// Creates a child env to typecheck the function body.
    fn typecheck_fn_decl_body(&self, fn_decl: &Spanned<FnDecl>) -> Result<()> {
        let (fn_type, formals, label) = {
            let env_entry = self.var(&fn_decl.id.t).unwrap();
            let fn_type = env_entry.ty.clone();
            // A level should have already been created by typecheck_fn_decl_sig().
            let label = env_entry.entry_type.fn_label();
            let levels = self.levels.borrow();
            let formals = levels[&label].formals();

            (fn_type, formals, label)
        };
        let mut new_env = self.new_child(label);

        if let Type::Fn(param_types, return_type) = fn_type.ty().as_ref() {
            for ((param_id, span), param_type, formal) in izip!(
                fn_decl.type_fields.iter().map(|tf| (&tf.id.t, tf.span)),
                param_types,
                &formals
            ) {
                new_env.insert_var(
                    param_id.clone(),
                    EnvEntry {
                        ty: param_type.clone(),
                        immutable: false,
                        entry_type: EnvEntryType::Var(formal.clone()),
                    },
                    span,
                );
            }

            let body_type = new_env.typecheck_expr(&fn_decl.body)?.t;
            if self.resolve_type(&body_type, fn_decl.body.span)?
                != self.resolve_type(
                    &return_type,
                    fn_decl
                        .return_type
                        .as_ref()
                        .map_or_else(|| fn_decl.span, |ret_ty| ret_ty.span),
                )?
            {
                return Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::TypeMismatch(return_type.clone(), body_type),
                    fn_decl.span,
                )]);
            }
        } else {
            panic!(format!("expected {} to be a function", fn_decl.id.t));
        }

        Ok(())
    }

    /// Converts type decls into pre-types. `convert_pre_types()` needs to be called in order to size
    /// all types and convert them to "full" types.
    pub(crate) fn typecheck_type_decl(&mut self, decl: &Spanned<TypeDecl>) -> Result<()> {
        let id = decl.id.t.clone();

        if self.contains_type(&id) {
            return Err(vec![TypecheckError::new_err(
                TypecheckErrorType::DuplicateType(id.clone()),
                decl.span,
            )
            .with_secondary_messages(vec![Spanned::new(
                format!("{} was defined here", id.clone()),
                self.type_def_span(&id).unwrap(),
            )])]);
        }

        let ty = _Type::from_type_decl(decl.t.clone());
        match &decl.ty.t {
            TypeDeclType::Record(record_fields) => {
                let field_def_spans: HashMap<String, FileSpan> = Self::check_for_duplicates(
                    record_fields,
                    |field| &field.id.t,
                    |field, span| {
                        let field_id = field.id.t.clone();
                        TypecheckError::new_err(TypecheckErrorType::DuplicateField(field_id.clone()), field.span)
                            .with_secondary_messages(vec![Spanned::new(
                                format!("{} was declared here", field_id),
                                span,
                            )])
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
                        TypecheckError::new_err(TypecheckErrorType::DuplicateEnumCase(case_id.clone()), case.span)
                            .with_secondary_messages(vec![Spanned::new(format!("{} was declared here", case_id), span)])
                    },
                )?;

                let decl_spans = self
                    .enum_case_param_decl_spans
                    .entry(decl.id.t.clone())
                    .or_insert_with(HashMap::new);
                for case in cases {
                    decl_spans.insert(case.id.t.clone(), case.params.iter().map(|param| param.span).collect());
                }
            }
            _ => (),
        }
        self.insert_pre_type(id, ty, decl.span);

        Ok(())
    }

    pub(crate) fn typecheck_expr(&self, expr: &Expr) -> Result<TranslateOutput<TypeInfo>> {
        match &expr.t {
            ExprType::Seq(exprs, returns) => {
                let (mut new_env, return_type, trexpr) = {
                    // New scope
                    let mut new_env = self.new_child(self.level_label.clone());
                    let mut stmts = vec![];
                    for expr in &exprs[..exprs.len() - 1] {
                        let trexpr = new_env.typecheck_expr_mut(expr)?.expr;
                        // We don't emit any IR for function decls, so ignore them.
                        match &expr.t {
                            ExprType::FnDecl(_) => (),
                            _ => stmts.push(trexpr.unwrap_stmt(&self.tmp_generator)),
                        }
                    }
                    let last_expr = exprs.last().unwrap();
                    let TranslateOutput {
                        t: return_type,
                        expr: trexpr,
                    } = if *returns {
                        // Here, units are represented with ir::Expr::Const(0), the same as if we
                        // returned the result of a function decl, so we don't have to take that
                        // special case into account.
                        new_env.typecheck_expr_mut(last_expr)?
                    } else {
                        let TranslateOutput { expr: trexpr, .. } = new_env.typecheck_expr_mut(last_expr)?;
                        TranslateOutput {
                            t: TypeInfo::unit(),
                            expr: translate::Expr::Expr(ir::Expr::Seq(
                                Box::new(trexpr.unwrap_stmt(&self.tmp_generator)),
                                Box::new(ir::Expr::Const(0)),
                            )),
                        }
                    };

                    // Remove all non-fn decl exprs so that in the second pass they'll already be
                    // defined.
                    let mut fn_decls: HashMap<String, EnvEntry> = HashMap::new();
                    for (id, var) in &new_env.vars {
                        if let Type::Fn(..) = var.ty.ty().as_ref() {
                            fn_decls.insert(id.clone(), var.clone());
                        }
                    }

                    new_env.vars = fn_decls;
                    (
                        new_env,
                        return_type,
                        if stmts.is_empty() {
                            trexpr.unwrap_expr(&self.tmp_generator)
                        } else {
                            ir::Expr::Seq(
                                Box::new(ir::Stmt::seq(stmts)),
                                Box::new(trexpr.unwrap_expr(&self.tmp_generator)),
                            )
                        },
                    )
                };

                // The only possible place where inline fn decl exprs can be is inside seq exprs
                // since that's the only place where typecheck_expr_mut() is called. We need to
                // typecheck all the expressions again in order to make sure that variables are
                // defined before they get captured.
                for expr in exprs {
                    if let ExprType::FnDecl(fn_decl) = &expr.t {
                        new_env.typecheck_fn_decl_body(fn_decl)?;
                    } else {
                        new_env.typecheck_expr_mut(expr)?;
                    }
                }

                Ok(TranslateOutput {
                    t: return_type,
                    expr: translate::Expr::Expr(trexpr),
                })
            }
            ExprType::String(s) => {
                let (label, fragment) = Frame::string(&self.tmp_generator, s);
                let mut fragments = self.fragments.borrow_mut();
                fragments.push(fragment);
                Ok(TranslateOutput {
                    t: TypeInfo::string(),
                    expr: translate::Expr::Expr(label),
                })
            }
            ExprType::Number(n) => Ok(TranslateOutput {
                t: TypeInfo::int(),
                expr: translate::Expr::Expr(ir::Expr::Const(*n as i64)),
            }),
            ExprType::Neg(expr) => {
                let TranslateOutput { expr, .. } = self.assert_ty(expr, &TypeInfo::int())?;
                Ok(TranslateOutput {
                    t: TypeInfo::int(),
                    expr: translate::Expr::Expr(ir::Expr::BinOp(
                        Box::new(ir::Expr::Const(-1)),
                        ir::BinOp::Mul,
                        Box::new(expr.unwrap_expr(&self.tmp_generator)),
                    )),
                })
            }
            ExprType::Arith(arith) => self.typecheck_arith(arith),
            ExprType::Unit => Ok(TranslateOutput {
                t: TypeInfo::unit(),
                expr: translate::Expr::Expr(ir::Expr::Const(0)),
            }),
            ExprType::Continue => {
                if let Some(continue_label) = self.continue_label.as_ref() {
                    Ok(TranslateOutput {
                        t: TypeInfo::unit(),
                        expr: translate::Expr::Stmt(ir::Stmt::Jump(
                            ir::Expr::Label(continue_label.clone()),
                            vec![continue_label.clone()],
                        )),
                    })
                } else {
                    Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::IllegalBreakOrContinue,
                        expr.span,
                    )])
                }
            }
            ExprType::Break => {
                if let Some(break_label) = self.break_label.as_ref() {
                    Ok(TranslateOutput {
                        t: TypeInfo::unit(),
                        expr: translate::Expr::Stmt(ir::Stmt::Jump(
                            ir::Expr::Label(break_label.clone()),
                            vec![break_label.clone()],
                        )),
                    })
                } else {
                    Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::IllegalBreakOrContinue,
                        expr.span,
                    )])
                }
            }
            ExprType::BoolLiteral(b) => Ok(TranslateOutput {
                t: TypeInfo::bool(),
                expr: translate::Expr::Expr(ir::Expr::Const(if *b { 1 } else { 0 })),
            }),
            ExprType::Not(expr) => self.assert_ty(expr, &TypeInfo::bool()),
            ExprType::Bool(bool_expr) => self.typecheck_bool(bool_expr),
            ExprType::LVal(lval) => {
                let TranslateOutput {
                    t: lval_type,
                    expr: lval_trexpr,
                } = self.typecheck_lval(lval)?.map(|lval_properties| lval_properties.ty);
                let resolved_lval_type = self.resolve_type(&lval_type, lval.span)?;
                if resolved_lval_type.is_frame_allocated() {
                    // FIXME: Make this actually works for frame-allocated types. We need to make
                    // sure that assigning this to another variable results in a memcpy.
                    Ok(TranslateOutput {
                        t: lval_type,
                        expr: lval_trexpr,
                    })
                } else {
                    // We're using this value as an r-value, so dereference.
                    let size = resolved_lval_type.size();
                    Ok(TranslateOutput {
                        t: lval_type,
                        expr: translate::Expr::Expr(ir::Expr::Mem(
                            Box::new(lval_trexpr.unwrap_expr(&self.tmp_generator)),
                            size,
                        )),
                    })
                }
            }
            ExprType::Let(_) => Err(vec![TypecheckError::new_err(
                TypecheckErrorType::IllegalLetExpr,
                expr.span,
            )]),
            ExprType::FnCall(fn_call) => self.typecheck_fn_call(fn_call),
            ExprType::Record(record) => self.typecheck_record(record),
            ExprType::Assign(assign) => self.typecheck_assign(assign),
            ExprType::Array(array) => self.typecheck_array(array),
            ExprType::If(if_expr) => self.typecheck_if(if_expr),
            ExprType::Range(range) => self.typecheck_range(range),
            ExprType::For(for_expr) => self.typecheck_for(for_expr),
            ExprType::While(while_expr) => self.typecheck_while(while_expr),
            ExprType::Compare(compare) => self.typecheck_compare(compare),
            ExprType::Enum(enum_expr) => self.typecheck_enum(enum_expr),
            ExprType::Closure(closure) => self.typecheck_closure(closure),
            ExprType::FnDecl(_) => Err(vec![TypecheckError::new_err(
                TypecheckErrorType::IllegalFnDeclExpr,
                expr.span,
            )]),
        }
    }

    pub(crate) fn typecheck_expr_mut(&mut self, expr: &Expr) -> Result<TranslateOutput<TypeInfo>> {
        match &expr.t {
            ExprType::Let(let_expr) => self.typecheck_let(let_expr),
            ExprType::FnDecl(fn_decl) => self.typecheck_fn_decl_expr(fn_decl),
            _ => self.typecheck_expr(expr),
        }
    }

    fn typecheck_lval(&self, lval: &Spanned<LVal>) -> Result<TranslateOutput<LValProperties>> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(env_entry) = self.var(var) {
                    let trexpr = if let EnvEntryType::Var(access) = &env_entry.entry_type {
                        let levels = self.levels.borrow();
                        translate::Expr::Expr(Env::translate_simple_var(&levels, access, &self.level_label))
                    } else {
                        // Reaching here means that we're trying to access a function through an
                        // lval. This isn't supported properly right now, so we'll return a
                        // placeholder value here for now.
                        translate::Expr::Expr(ir::Expr::Const(0))
                    };
                    Ok(TranslateOutput {
                        t: LValProperties {
                            ty: env_entry.ty.clone(),
                            immutable: if env_entry.immutable { Some(var.clone()) } else { None },
                        },
                        expr: trexpr,
                    })
                } else {
                    Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::UndefinedVar(var.clone()),
                        lval.span,
                    )])
                }
            }
            LVal::Field(var, field) => {
                let TranslateOutput {
                    t: lval_properties,
                    expr: record_trexpr,
                } = self.typecheck_lval(var)?;
                if let Type::Record(_, fields) = self.resolve_type(&lval_properties.ty, var.span)?.ty().as_ref() {
                    if let Some(RecordField { ty: field_type, offset }) = fields.get(&field.t) {
                        // A field is mutable only if the record it belongs to is mutable
                        let trexpr = record_trexpr
                            .unwrap_expr(&self.tmp_generator)
                            .pointer_offset(-(*offset as i64));
                        return Ok(TranslateOutput {
                            t: LValProperties {
                                ty: field_type.clone(),
                                immutable: lval_properties.immutable,
                            },
                            expr: translate::Expr::Expr(trexpr),
                        });
                    }
                }
                Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::UndefinedField(field.t.clone()),
                    field.span,
                )])
            }
            LVal::Subscript(var, index) => {
                let TranslateOutput {
                    t: lval_properties,
                    expr: lval_trexpr,
                } = self.typecheck_lval(var)?;
                let TranslateOutput {
                    t: index_type,
                    expr: index_trexpr,
                } = self.typecheck_expr(index)?;
                if let Type::Array(element_ty, len) = self.resolve_type(&lval_properties.ty, var.span)?.ty().as_ref() {
                    if self.resolve_type(&index_type, index.span)? == TypeInfo::int() {
                        let index_tmp_trexpr = ir::Expr::Tmp(self.tmp_generator.new_tmp());
                        let true_label_1 = self.tmp_generator.new_label();
                        let true_label_2 = self.tmp_generator.new_label();
                        let false_label = self.tmp_generator.new_label();
                        let join_label = self.tmp_generator.new_label();
                        let bounds_check_stmt = vec![
                            ir::Stmt::Move(index_tmp_trexpr.clone(), index_trexpr.unwrap_expr(&self.tmp_generator)),
                            ir::Stmt::CJump(
                                index_tmp_trexpr.clone(),
                                ir::CompareOp::Ge,
                                ir::Expr::Const(0),
                                true_label_1.clone(),
                                false_label.clone(),
                            ),
                            ir::Stmt::Label(true_label_1),
                            ir::Stmt::CJump(
                                index_tmp_trexpr.clone(),
                                ir::CompareOp::Lt,
                                ir::Expr::Const(*len as i64),
                                true_label_2.clone(),
                                false_label.clone(),
                            ),
                            ir::Stmt::Label(true_label_2),
                            ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
                            ir::Stmt::Label(false_label),
                            ir::Stmt::Expr(ir::Expr::Call(
                                Box::new(ir::Expr::Label(Label("__panic".to_owned()))),
                                vec![],
                            )),
                            ir::Stmt::Label(join_label),
                        ];

                        let element_size = element_ty.size();

                        Ok(TranslateOutput {
                            t: LValProperties {
                                ty: element_ty.clone(),
                                immutable: lval_properties.immutable,
                            },
                            expr: translate::Expr::Expr(ir::Expr::Seq(
                                Box::new(ir::Stmt::seq(bounds_check_stmt)),
                                Box::new(
                                    lval_trexpr
                                        .unwrap_expr(&self.tmp_generator)
                                        .array_offset(index_tmp_trexpr, element_size),
                                ),
                            )),
                        })
                    } else {
                        Err(vec![TypecheckError::new_err(
                            TypecheckErrorType::TypeMismatch(TypeInfo::int(), index_type),
                            index.span,
                        )])
                    }
                } else {
                    Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::CannotSubscript,
                        index.span,
                    )])
                }
            }
        }
    }

    fn typecheck_let(&mut self, let_expr: &Spanned<Let>) -> Result<TranslateOutput<TypeInfo>> {
        let Let {
            pattern,
            ty: ast_ty,
            immutable,
            expr,
        } = &let_expr.t;
        let TranslateOutput {
            t: expr_type,
            expr: assigned_val_trexpr,
        } = self.typecheck_expr(expr)?;
        if let Some(ast_ty) = ast_ty {
            // Type annotation
            let ty = self.convert_pre_type(&Rc::new(ast_ty.t.clone().into()));
            let resolved_ty = self.resolve_type(&ty, ast_ty.span)?;
            let resolved_expr_ty = self.resolve_type(&expr_type, expr.span)?;
            if resolved_ty != resolved_expr_ty {
                return Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::TypeMismatch(ty, expr_type),
                    let_expr.span,
                )]);
            }
        }
        if let Pattern::String(var_name) = &pattern.t {
            // Prefer the annotated type if provided
            let ty = if let Some(ty) = ast_ty {
                self.convert_pre_type(&Rc::new(ty.t.clone().into()))
            } else {
                expr_type
            };
            let resolved_type = self.resolve_type(&ty, expr.span)?;
            // FIXME: Don't assume all locals escape
            // FIXME: For frame-allocated types (e.g. records, arrays, etc.), *two* locals will be
            // allocated: one in typecheck_* and one here.
            let local = {
                let mut levels = self.levels.borrow_mut();
                let level = levels.get_mut(&self.level_label).unwrap();
                level.alloc_local(&self.tmp_generator, resolved_type.size(), true)
            };

            self.insert_var(
                var_name.clone(),
                EnvEntry {
                    ty: ty.clone(),
                    immutable: immutable.t,
                    entry_type: EnvEntryType::Var(local.clone()),
                },
                let_expr.span,
            );
            let levels = self.levels.borrow();
            let assignee_trexpr = Env::translate_simple_var(&levels, &local, &self.level_label);

            let assign_expr = if resolved_type.is_frame_allocated() {
                if let ir::Expr::Seq(stmts, expr) = assigned_val_trexpr.unwrap_expr(&self.tmp_generator) {
                    let stmts = stmts.appending(translate::copy(assignee_trexpr, *expr, &ty));
                    translate::Expr::Stmt(stmts)
                } else {
                    panic!("frame-allocated value doesn't translate to ir::Expr::Seq");
                }
            } else {
                translate::Expr::Stmt(translate::copy(
                    assignee_trexpr,
                    assigned_val_trexpr.unwrap_expr(&self.tmp_generator),
                    &resolved_type,
                ))
            };

            Ok(TranslateOutput {
                t: TypeInfo::unit(),
                expr: assign_expr,
            })
        } else {
            Ok(TranslateOutput {
                t: TypeInfo::unit(),
                expr: translate::Expr::Stmt(ir::Stmt::Seq(
                    Box::new(assigned_val_trexpr.unwrap_stmt(&self.tmp_generator)),
                    Box::new(ir::Stmt::Expr(ir::Expr::Const(0))),
                )),
            })
        }
    }

    fn typecheck_fn_call(
        &self,
        Spanned {
            t: FnCall { id, args },
            span,
        }: &Spanned<FnCall>,
    ) -> Result<TranslateOutput<TypeInfo>> {
        if let Some(fn_type) = self.var(&id.t).map(|x| x.ty.clone()) {
            if let Type::Fn(param_types, return_type) = self.resolve_type(&fn_type, id.span)?.ty().as_ref() {
                if args.len() != param_types.len() {
                    return Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::ArityMismatch(param_types.len(), args.len()),
                        *span,
                    )]);
                }

                let mut errors = vec![];
                for (index, (arg, param_type)) in args.iter().zip(param_types.iter()).enumerate() {
                    match self.typecheck_expr(arg) {
                        Ok(TranslateOutput { t: ty, .. }) => {
                            // param_type should already be well-defined because we have already
                            // checked for invalid types
                            if self.resolve_type(&ty, arg.span)?
                                != self
                                    .resolve_type(param_type, FileSpan::new(self.file_id, Span::initial()))
                                    .unwrap()
                            {
                                let mut err = TypecheckError::new_err(
                                    TypecheckErrorType::TypeMismatch(param_type.clone(), ty.clone()),
                                    arg.span,
                                );
                                if let Some(decl_spans) = self.fn_param_decl_spans(id) {
                                    err.secondary_messages =
                                        vec![Spanned::new("declared here".to_owned(), decl_spans[index])];
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

                warn!(TYPECHECK_LOG, "unimplemented");
                Ok(TranslateOutput {
                    t: return_type.clone(),
                    expr: translate::Expr::Expr(ir::Expr::Const(0)),
                })
            } else {
                Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::NotAFn(id.t.clone()),
                    id.span,
                )])
            }
        } else {
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::UndefinedFn(id.t.clone()),
                *span,
            )])
        }
    }

    /// The returned `Expr` is a pointer (*not* a `Mem`) to a stack-allocated region of memory large
    /// enough for the record to fit (rounded up to the nearest multiple of the word size).
    fn typecheck_record(
        &self,
        Spanned {
            t: Record {
                id: record_id,
                field_assigns,
            },
            span,
            ..
        }: &Spanned<Record>,
    ) -> Result<TranslateOutput<TypeInfo>> {
        if let Some(ty) = self.r#type(&record_id.t) {
            if let Type::Record(_, field_types) = ty.ty().as_ref() {
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
                    errors.push(TypecheckError::new_err(
                        TypecheckErrorType::MissingFields(missing_fields),
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
                    errors.push(TypecheckError::new_err(
                        TypecheckErrorType::InvalidFields(invalid_fields),
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
                        TypecheckError::new_err(
                            TypecheckErrorType::DuplicateField(field_assign.id.t.clone()),
                            field_assign.span,
                        )
                        .with_secondary_messages(vec![Spanned::new(
                            format!("{} was defined here", field_assign.id.t.clone()),
                            span,
                        )])
                    },
                )?;

                let record_alloc_trexpr = {
                    let mut levels = self.levels.borrow_mut();
                    let level = levels.get_mut(&self.level_label).unwrap();
                    let access = level.alloc_local(&self.tmp_generator, ty.size(), true);
                    Env::translate_simple_var(&levels, &access, &self.level_label)
                };
                let mut assign_stmts = vec![];
                let mut errors = vec![];
                for Spanned {
                    t: FieldAssign { id: field_id, expr },
                    span,
                } in field_assigns
                {
                    // This should never error because we already checked for invalid types
                    let expected_type = self
                        .resolve_type(
                            &field_types[&field_id.t].ty,
                            FileSpan::new(self.file_id, Span::initial()),
                        )
                        .unwrap();
                    let TranslateOutput { t: ty, expr: trexpr } = self.typecheck_expr(expr)?;
                    let actual_type = self.resolve_type(&ty, expr.span)?;
                    if expected_type != actual_type {
                        errors.push(
                            TypecheckError::new_err(
                                TypecheckErrorType::TypeMismatch(expected_type.clone(), actual_type.clone()),
                                *span,
                            )
                            .with_secondary_messages(vec![Spanned::new(
                                format!("{} was declared here", field_id.t),
                                self.record_field_decl_spans(&record_id.t).unwrap()[&field_id.t],
                            )]),
                        );
                    }

                    let field_offset = field_types[&field_id.t].offset;
                    let field_trexpr = record_alloc_trexpr.pointer_offset(-(field_offset as i64));

                    assign_stmts.push(translate::copy(
                        field_trexpr,
                        trexpr.unwrap_expr(&self.tmp_generator),
                        &expected_type,
                    ));
                }

                if !errors.is_empty() {
                    return Err(errors);
                }

                Ok(TranslateOutput {
                    t: ty.clone(),
                    expr: translate::Expr::Expr(ir::Expr::Seq(
                        Box::new(ir::Stmt::seq(assign_stmts)),
                        Box::new(record_alloc_trexpr),
                    )),
                })
            } else {
                Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::NotARecord(ty.clone()),
                    *span,
                )])
            }
        } else {
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::UndefinedType(record_id.t.clone()),
                record_id.span,
            )])
        }
    }

    fn typecheck_assign(&self, Spanned { t: assign, span }: &Spanned<Assign>) -> Result<TranslateOutput<TypeInfo>> {
        let TranslateOutput {
            t: lval_properties,
            expr: assignee_trexpr,
        } = self.typecheck_lval(&assign.lval)?;
        // Make sure we don't mutate an immutable var.
        if let Some(root) = lval_properties.immutable.as_ref() {
            let def_span = self.var_def_span(root).unwrap();
            return Err(vec![TypecheckError::new_err(
                TypecheckErrorType::MutatingImmutable(root.clone()),
                assign.lval.span,
            )
            .with_secondary_messages(vec![Spanned::new(
                format!("{} was defined here", root.clone()),
                def_span,
            )])]);
        }

        let TranslateOutput {
            t: assigned_val_ty,
            expr: assigned_val_trexpr,
        } = self.typecheck_expr(&assign.expr)?;
        let resolved_actual_ty = self.resolve_type(&assigned_val_ty, assign.expr.span)?;
        let resolved_expected_ty = self.resolve_type(&lval_properties.ty, assign.lval.span)?;
        if resolved_expected_ty != resolved_actual_ty {
            return Err(vec![TypecheckError::new_err(
                TypecheckErrorType::TypeMismatch(resolved_expected_ty, resolved_actual_ty),
                *span,
            )]);
        }

        Ok(TranslateOutput {
            t: TypeInfo::unit(),
            expr: translate::Expr::Stmt(translate::copy(
                assignee_trexpr.unwrap_expr(&self.tmp_generator),
                assigned_val_trexpr.unwrap_expr(&self.tmp_generator),
                &resolved_expected_ty,
            )),
        })
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
                    ArithOp::Mod => Ok(l % r),
                }
            }
            _ => Err(vec![TypecheckError::new_err(
                TypecheckErrorType::NonConstantArithExpr(expr.clone()),
                *span,
            )]),
        }
    }

    fn typecheck_array(&self, Spanned { t: array, .. }: &Spanned<Array>) -> Result<TranslateOutput<TypeInfo>> {
        let TranslateOutput {
            t: elem_type,
            expr: init_val_trexpr,
        } = self.typecheck_expr(&array.initial_value)?;
        let elem_size = elem_type.size();
        let len = Self::eval_arith_const_expr(&array.len)?;
        if len < 0 {
            return Err(vec![TypecheckError::new_err(
                TypecheckErrorType::NegativeArrayLen(len),
                array.len.span,
            )]);
        }

        let array_alloc_trexpr = {
            let mut levels = self.levels.borrow_mut();
            let level = levels.get_mut(&self.level_label).unwrap();
            let access = level.alloc_local(&self.tmp_generator, len as usize * elem_size, true);
            Frame::expr(access.access, &ir::Expr::Tmp(tmp::FP.clone()))
        };
        let array_tmp_irexpr = ir::Expr::Tmp(self.tmp_generator.new_tmp());

        let mut assign_stmts = vec![ir::Stmt::Move(array_tmp_irexpr.clone(), array_alloc_trexpr)];
        for i in 0..len {
            let elem_irexpr = array_tmp_irexpr.array_offset(ir::Expr::Const(i as i64), elem_size);

            assign_stmts.push(translate::copy(
                elem_irexpr,
                init_val_trexpr.clone().unwrap_expr(&self.tmp_generator),
                &elem_type,
            ));
        }

        Ok(TranslateOutput {
            t: TypeInfo::new(Rc::new(Type::Array(elem_type, len as usize)), elem_size * len as usize),
            expr: translate::Expr::Expr(ir::Expr::Seq(
                Box::new(ir::Stmt::seq(assign_stmts)),
                Box::new(array_tmp_irexpr),
            )),
        })
    }

    fn typecheck_if(&self, Spanned { t: expr, span }: &Spanned<If>) -> Result<TranslateOutput<TypeInfo>> {
        let true_label = self.tmp_generator.new_label();
        let false_label = self.tmp_generator.new_label();
        let join_label = self.tmp_generator.new_label();
        let result = self.tmp_generator.new_tmp();
        let cond_gen = self.assert_ty(&expr.cond, &TypeInfo::bool())?.expr.unwrap_cond();

        let mut injected_labels = None;
        let mut then_expr_was_cond = false;
        let mut else_expr_was_cond = false;

        let TranslateOutput {
            t: then_expr_type,
            expr: translated_then_expr,
        } = self.typecheck_expr(&expr.then_expr)?;
        let then_instr = match translated_then_expr {
            translate::Expr::Stmt(stmt) => {
                // If the then expression is a statement, don't bother unwrapping into an Expr::Expr and just run it
                // directly.
                stmt
            }
            translate::Expr::Cond(cond_gen) => {
                let true_label = self.tmp_generator.new_label();
                let false_label = self.tmp_generator.new_label();
                let stmt = ir::Stmt::seq(vec![cond_gen(true_label.clone(), false_label.clone())]);
                injected_labels = Some((true_label, false_label));
                then_expr_was_cond = true;
                stmt
            }
            expr => ir::Stmt::Move(ir::Expr::Tmp(result), expr.unwrap_expr(&self.tmp_generator)),
        };

        let else_instr = if let Some(else_expr) = &expr.else_expr {
            let TranslateOutput {
                t: else_expr_type,
                expr: translated_else_expr,
            } = self.typecheck_expr(else_expr)?;
            if self.resolve_type(&then_expr_type, expr.then_expr.span)?
                != self.resolve_type(&else_expr_type, else_expr.span)?
            {
                return Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::TypeMismatch(then_expr_type.clone(), else_expr_type.clone()),
                    *span,
                )
                .with_secondary_messages(vec![Spanned::new(
                    format!(
                        "then branch has type {:?}, but else branch has type {:?}",
                        then_expr_type, else_expr_type
                    ),
                    *span,
                )])]);
            } else {
                Some(match translated_else_expr {
                    translate::Expr::Stmt(stmt) => {
                        // If the then expression is a statement, don't bother unwrapping into an Expr::Expr and just run it
                        // directly.
                        stmt
                    }
                    translate::Expr::Cond(cond_gen) => {
                        let (true_label, false_label) = injected_labels
                            .get_or_insert_with(|| (self.tmp_generator.new_label(), self.tmp_generator.new_label()));
                        let stmt = ir::Stmt::seq(vec![cond_gen(true_label.clone(), false_label.clone())]);
                        else_expr_was_cond = true;
                        stmt
                    }
                    expr => ir::Stmt::Move(ir::Expr::Tmp(result), expr.unwrap_expr(&self.tmp_generator)),
                })
            }
        } else {
            if then_expr_type != TypeInfo::unit() {
                return Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::TypeMismatch(TypeInfo::unit(), then_expr_type),
                    expr.then_expr.span,
                )]);
            }
            None
        };

        let cond_stmt = if else_instr.is_some() {
            cond_gen(true_label.clone(), false_label.clone())
        } else {
            // Jump directly to join_label if there's no else branch
            cond_gen(true_label.clone(), join_label.clone())
        };

        let mut seq = vec![cond_stmt, ir::Stmt::Label(true_label.clone()), then_instr];
        // then_instr will be a CJump if then_expr was Cond, so only insert a Jump if then_expr wasn't Cond
        if !then_expr_was_cond {
            seq.extend_from_slice(&[ir::Stmt::Jump(
                ir::Expr::Label(join_label.clone()),
                vec![join_label.clone()],
            )]);
        }
        if let Some(else_instr) = &else_instr {
            seq.extend(vec![ir::Stmt::Label(false_label), else_instr.clone()]);
            // Same logic here
            if !else_expr_was_cond {
                seq.extend_from_slice(&[ir::Stmt::Jump(
                    ir::Expr::Label(join_label.clone()),
                    vec![join_label.clone()],
                )]);
            }
        }

        if then_expr_was_cond || else_expr_was_cond {
            let (true_label, false_label) = injected_labels.unwrap();
            seq.extend_from_slice(&[
                ir::Stmt::Label(true_label),
                ir::Stmt::Move(ir::Expr::Tmp(result), ir::Expr::Const(1)),
                ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
                ir::Stmt::Label(false_label),
                ir::Stmt::Move(ir::Expr::Tmp(result), ir::Expr::Const(0)),
                ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
            ]);
        }
        seq.push(ir::Stmt::Label(join_label));

        Ok(TranslateOutput {
            t: if else_instr.is_some() {
                then_expr_type
            } else {
                TypeInfo::unit()
            },
            expr: translate::Expr::Expr(ir::Expr::Seq(
                Box::new(ir::Stmt::seq(seq)),
                Box::new(ir::Expr::Tmp(result)),
            )),
        })
    }

    fn typecheck_range(&self, Spanned { t: expr, .. }: &Spanned<Range>) -> Result<TranslateOutput<TypeInfo>> {
        self.assert_ty(&expr.lower, &TypeInfo::int())?;
        self.assert_ty(&expr.upper, &TypeInfo::int())?;
        Ok(TranslateOutput {
            t: TypeInfo::new(Rc::new(Type::Iterator(TypeInfo::int())), 2 * TypeInfo::int().size()),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    fn typecheck_for(&self, Spanned { t: expr, .. }: &Spanned<For>) -> Result<TranslateOutput<TypeInfo>> {
        self.assert_ty(
            &expr.range,
            &TypeInfo::new(Rc::new(Type::Iterator(TypeInfo::int())), 2 * TypeInfo::int().size()),
        )?;
        // Only support range literals in for loops for now
        if let ExprType::Range(range_expr) = &expr.range.t {
            let TranslateOutput { expr: lower_trexpr, .. } = self.typecheck_expr(&range_expr.lower)?;
            let lower_ir_expr = lower_trexpr.unwrap_expr(&self.tmp_generator);
            let TranslateOutput { expr: upper_trexpr, .. } = self.typecheck_expr(&range_expr.upper)?;
            let upper_ir_expr = upper_trexpr.unwrap_expr(&self.tmp_generator);

            let mut child_env = self.new_child(self.level_label.clone());
            let (index_local, end_local) = {
                let mut levels = child_env.levels.borrow_mut();
                let level = levels.get_mut(&self.level_label).unwrap();
                (
                    level.alloc_local(&self.tmp_generator, TypeInfo::int().size(), true),
                    level.alloc_local(&self.tmp_generator, TypeInfo::int().size(), true),
                )
            };

            let levels = self.levels.borrow();
            // This needs to be dereferenced since it's an int value.
            let index_expr = ir::Expr::Mem(
                Box::new(Env::translate_simple_var(&levels, &index_local, &self.level_label)),
                TypeInfo::int().size(),
            );
            // This gets captured by gen_stmt, so I need a duplicate clone to move into the closure.
            let index_expr_ = index_expr.clone();

            let pre_stmts = vec![
                ir::Stmt::Move(index_expr.clone(), lower_ir_expr.clone()),
                ir::Stmt::Move(
                    ir::Expr::Mem(
                        Box::new(Env::translate_simple_var(&levels, &end_local, &self.level_label)),
                        TypeInfo::int().size(),
                    ),
                    upper_ir_expr.clone(),
                ),
            ];
            let gen_stmt = Rc::new(move |true_label, false_label| {
                ir::Stmt::CJump(
                    index_expr_.clone(),
                    ir::CompareOp::Lt,
                    upper_ir_expr.clone(),
                    true_label,
                    false_label,
                )
            });
            let test_label = self.tmp_generator.new_label();
            let continue_label = self.tmp_generator.new_label();
            let done_label = self.tmp_generator.new_label();
            let cond_true_label = self.tmp_generator.new_label();

            child_env.continue_label = Some(continue_label.clone());
            child_env.break_label = Some(done_label.clone());
            child_env.insert_var(
                expr.index.t.clone(),
                EnvEntry {
                    ty: TypeInfo::int(),
                    immutable: false,
                    // FIXME
                    entry_type: EnvEntryType::Var(index_local),
                },
                expr.index.span,
            );

            let body = child_env
                .assert_ty(&expr.body, &TypeInfo::unit())?
                .expr
                .unwrap_stmt(&self.tmp_generator);

            let mut stmts = pre_stmts;
            stmts.extend_from_slice(&[
                ir::Stmt::Label(test_label.clone()),
                gen_stmt(cond_true_label.clone(), done_label.clone()),
                ir::Stmt::Label(cond_true_label),
                body,
                ir::Stmt::Label(continue_label.clone()),
                ir::Stmt::Move(
                    index_expr.clone(),
                    ir::Expr::BinOp(
                        Box::new(index_expr.clone()),
                        ir::BinOp::Add,
                        Box::new(ir::Expr::Const(1)),
                    ),
                ),
                ir::Stmt::Jump(ir::Expr::Label(test_label.clone()), vec![test_label]),
                ir::Stmt::Label(done_label),
            ]);

            Ok(TranslateOutput {
                t: TypeInfo::unit(),
                expr: translate::Expr::Stmt(ir::Stmt::seq(stmts)),
            })
        } else {
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::NotARangeLiteral,
                expr.range.span,
            )
            .with_secondary_messages(vec![Spanned::new(
                "only range literals are supported in for loops for now".to_owned(),
                expr.range.span,
            )])])
        }
    }

    fn typecheck_while(&self, Spanned { t: expr, .. }: &Spanned<While>) -> Result<TranslateOutput<TypeInfo>> {
        let test_label = self.tmp_generator.new_label();
        let done_label = self.tmp_generator.new_label();
        let cond_true_label = self.tmp_generator.new_label();
        let cond = self.assert_ty(&expr.cond, &TypeInfo::bool())?.expr;
        let gen_stmt = cond.unwrap_cond();

        let mut child_env = self.new_child(self.level_label.clone());
        child_env.continue_label = Some(test_label.clone());
        child_env.break_label = Some(done_label.clone());
        let body = child_env
            .assert_ty(&expr.body, &TypeInfo::unit())?
            .expr
            .unwrap_stmt(&self.tmp_generator);

        let stmts = ir::Stmt::seq(vec![
            ir::Stmt::Label(test_label.clone()),
            gen_stmt(cond_true_label.clone(), done_label.clone()),
            ir::Stmt::Label(cond_true_label),
            body,
            ir::Stmt::Jump(ir::Expr::Label(test_label.clone()), vec![test_label]),
            ir::Stmt::Label(done_label),
        ]);
        Ok(TranslateOutput {
            t: TypeInfo::unit(),
            expr: translate::Expr::Stmt(stmts),
        })
    }

    fn typecheck_arith(&self, Spanned { t: expr, .. }: &Spanned<Arith>) -> Result<TranslateOutput<TypeInfo>> {
        let TranslateOutput { expr: l_expr, .. } = self.assert_ty(&expr.l, &TypeInfo::int())?;
        let TranslateOutput { expr: r_expr, .. } = self.assert_ty(&expr.r, &TypeInfo::int())?;
        Ok(TranslateOutput {
            t: TypeInfo::int(),
            expr: translate::Expr::Expr(ir::Expr::BinOp(
                Box::new(l_expr.unwrap_expr(&self.tmp_generator)),
                expr.op.t.into(),
                Box::new(r_expr.unwrap_expr(&self.tmp_generator)),
            )),
        })
    }

    fn typecheck_bool(&self, Spanned { t: expr, span }: &Spanned<Bool>) -> Result<TranslateOutput<TypeInfo>> {
        // We should still typecheck each element individually to get more informative error messages.
        self.assert_ty(&expr.l, &TypeInfo::bool())?;
        self.assert_ty(&expr.r, &TypeInfo::bool())?;
        let if_expr = expr.clone().into_if();
        Ok(TranslateOutput {
            t: TypeInfo::bool(),
            expr: self.typecheck_if(&Spanned::new(if_expr, *span))?.expr,
        })
    }

    fn typecheck_compare(&self, Spanned { t: expr, span }: &Spanned<Compare>) -> Result<TranslateOutput<TypeInfo>> {
        let TranslateOutput { t: l_ty, expr: l_expr } = self.typecheck_expr(&expr.l)?;
        let TranslateOutput { t: r_ty, expr: r_expr } = self.typecheck_expr(&expr.r)?;
        let left_type = self.resolve_type(&l_ty, expr.l.span)?;
        let right_type = self.resolve_type(&r_ty, expr.r.span)?;
        if left_type != right_type {
            return Err(vec![TypecheckError::new_err(
                TypecheckErrorType::TypeMismatch(left_type.clone(), right_type.clone()),
                *span,
            )]);
        }

        let l_expr = l_expr.unwrap_expr(&self.tmp_generator);
        let r_expr = r_expr.unwrap_expr(&self.tmp_generator);

        let expr = if left_type == TypeInfo::string() {
            translate::Expr::Expr(Frame::external_call("__strcmp", vec![l_expr, r_expr]))
        } else {
            let op = expr.op.t.into();
            translate::Expr::Cond(Rc::new(move |true_label, false_label| {
                ir::Stmt::CJump(l_expr.clone(), op, r_expr.clone(), true_label, false_label)
            }))
        };

        Ok(TranslateOutput {
            t: TypeInfo::bool(),
            expr,
        })
    }

    fn typecheck_enum(&self, Spanned { t: expr, .. }: &Spanned<Enum>) -> Result<TranslateOutput<TypeInfo>> {
        if let Some(ty) = self.r#type(&expr.enum_id) {
            if let Type::Enum(_, enum_cases) = ty.ty().as_ref() {
                if let Some(EnumCase { params, .. }) = enum_cases.get(&expr.case_id.t) {
                    if params.len() != expr.args.len() {
                        return Err(vec![TypecheckError::new_err(
                            TypecheckErrorType::ArityMismatch(params.len(), expr.args.len()),
                            expr.args.span,
                        )]);
                    }

                    let mut errors = vec![];
                    for (index, (arg, param_type)) in expr.args.iter().zip(params.iter()).enumerate() {
                        match self.typecheck_expr(arg) {
                            Ok(TranslateOutput { t: ty, .. }) => {
                                // param_type should already be well-defined because we have already
                                // checked for invalid types
                                if self.resolve_type(&ty, arg.span)?
                                    != self
                                        .resolve_type(param_type, FileSpan::new(self.file_id, Span::initial()))
                                        .unwrap()
                                {
                                    let mut err = TypecheckError::new_err(
                                        TypecheckErrorType::TypeMismatch(param_type.clone(), ty.clone()),
                                        arg.span,
                                    );
                                    if let Some(decl_spans) = self.enum_case_param_decl_spans(&expr.enum_id) {
                                        err.secondary_messages = vec![Spanned::new(
                                            "declared here".to_owned(),
                                            decl_spans[&expr.case_id.t][index],
                                        )];
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

                    Ok(TranslateOutput {
                        t: ty.clone(),
                        expr: translate::Expr::Expr(ir::Expr::Const(0)),
                    })
                } else {
                    Err(vec![TypecheckError::new_err(
                        TypecheckErrorType::NotAnEnumCase(expr.case_id.t.clone()),
                        expr.case_id.span,
                    )])
                }
            } else {
                Err(vec![TypecheckError::new_err(
                    TypecheckErrorType::NotAnEnum(ty.clone()),
                    expr.enum_id.span,
                )])
            }
        } else {
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::UndefinedType(expr.enum_id.t.clone()),
                expr.enum_id.span,
            )])
        }
    }

    fn typecheck_closure(&self, Spanned { t: expr, .. }: &Spanned<Closure>) -> Result<TranslateOutput<TypeInfo>> {
        // FIXME: We need to create a new level for the closure?
        let child_env = self.new_child(self.level_label.clone());
        Self::check_for_duplicates(
            &expr.type_fields,
            |type_field| &type_field.id.t,
            |type_field, span| {
                TypecheckError::new_err(
                    TypecheckErrorType::DuplicateParam(type_field.id.t.clone()),
                    type_field.span,
                )
                .with_secondary_messages(vec![Spanned::new(
                    format!("{} was declared here", type_field.id.t.clone()),
                    span,
                )])
            },
        )?;

        let mut param_types = vec![];
        let mut errors = vec![];
        for type_field in &expr.type_fields {
            let ty = self.convert_pre_type(&Rc::new(type_field.ty.t.clone().into()));
            match self.resolve_type(&ty, type_field.ty.span) {
                Ok(_) => {
                    param_types.push(ty.clone());
                    // child_env.insert_var(
                    //     type_field.id.t.clone(),
                    //     EnvEntry {
                    //         ty,
                    //         immutable: false,
                    //         // FIXME
                    //         entry_type: EnvEntryType::Var()
                    //     },
                    //     type_field.span,
                    // );
                    unimplemented!("closures are not currently supported");
                }
                Err(errs) => errors.extend(errs),
            }
        }

        let return_type = child_env.typecheck_expr(&expr.body)?.t;
        if let Err(errs) = child_env.resolve_type(&return_type, expr.body.span) {
            errors.extend(errs);
        }

        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(TranslateOutput {
            t: TypeInfo::new(Rc::new(Type::Fn(param_types, return_type)), std::mem::size_of::<u64>()),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    fn typecheck_fn_decl_expr(&mut self, fn_decl: &Spanned<FnDecl>) -> Result<TranslateOutput<TypeInfo>> {
        // At this point, we have already typechecked all type decls, so we can validate the param
        // and return types.
        self.validate_fn_decl_pre_types(fn_decl)?;
        self.typecheck_fn_decl_sig(fn_decl)?;

        // Defer typechecking function body until all function declaration signatures have been
        // typechecked. This allows for mutually recursive functions.
        Ok(TranslateOutput {
            t: TypeInfo::unit(),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    /// Same as `Frame::expr()`, but follows statics links.
    pub fn translate_simple_var(levels: &HashMap<Label, Level>, access: &Access, level_label: &Label) -> ir::Expr {
        let def_level_label = &access.level_label;
        let def_level = &levels[def_level_label];
        let current_level = &levels[level_label];
        let static_link = Env::static_link(levels, current_level, def_level);
        Frame::expr(access.access, &static_link)
    }

    pub fn static_link(levels: &HashMap<Label, Level>, from_level: &Level, to_level: &Level) -> ir::Expr {
        if from_level == to_level {
            ir::Expr::Tmp(*tmp::FP)
        } else {
            let parent_label = from_level.parent_label.as_ref().unwrap();
            let parent_level = &levels[&parent_label];
            ir::Expr::Mem(
                Box::new(Env::static_link(levels, parent_level, to_level)),
                std::mem::size_of::<u64>(),
            )
        }
    }

    fn proc_entry_exit(&self, level: Level, body: ir::Stmt) {
        let mut fragments = self.fragments.borrow_mut();
        fragments.push(Fragment::Fn(FnFragment {
            label: level.label().clone(),
            body,
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{self, BoolOp, CompareOp, TypeField},
        frame,
        ty::_EnumCase,
        utils::EMPTY_SOURCEMAP,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn resolve_pre_type() {
        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, Label::top());
        env.insert_pre_type("a".to_owned(), _Type::Alias("b".to_owned()), zspan!());
        env.insert_pre_type("b".to_owned(), _Type::Alias("c".to_owned()), zspan!());
        env.insert_pre_type("c".to_owned(), _Type::Int, zspan!());
        env.insert_pre_type("d".to_owned(), _Type::Alias("e".to_owned()), zspan!());

        assert_eq!(
            env.resolve_pre_type(&Rc::new(_Type::Alias("a".to_owned())), zspan!()),
            Ok(Rc::new(_Type::Int))
        );
        assert_eq!(
            env.resolve_pre_type(&Rc::new(_Type::Alias("d".to_owned())), zspan!())
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::UndefinedType("e".to_owned()),
        );
    }

    #[test]
    fn child_env() {
        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, Label::top());
        env.convert_pre_types();

        let var_properties = EnvEntry {
            ty: TypeInfo::int(),
            immutable: true,
            entry_type: EnvEntryType::Var(Access {
                level_label: Label::top(),
                access: frame::Access {
                    ty: frame::AccessType::InFrame(-8),
                    size: std::mem::size_of::<i64>(),
                },
            }),
        };
        env.insert_var("a".to_owned(), var_properties.clone(), zspan!());

        let mut child_env = env.new_child(env.level_label.clone());
        child_env.insert_type(
            "i".to_owned(),
            TypeInfo::new(Rc::new(Type::Alias("int".to_owned())), std::mem::size_of::<i64>()),
            zspan!(),
        );

        assert!(child_env.contains_type("int"));
        assert!(child_env.contains_type("i"));
        assert_eq!(
            child_env.resolve_type(child_env.r#type("i").unwrap(), zspan!()),
            Ok(TypeInfo::int())
        );
        assert_eq!(child_env.var("a"), Some(&var_properties));
    }

    #[test]
    fn typecheck_bool_expr() {
        let level_label = Label::top();

        let expr = zspan!(ExprType::Bool(Box::new(zspan!(Bool {
            l: zspan!(ExprType::BoolLiteral(true)),
            op: zspan!(BoolOp::And),
            r: zspan!(ExprType::BoolLiteral(true)),
        }))));
        let env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, level_label);
        assert_eq!(
            env.typecheck_expr(&expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::bool())
        );
    }

    #[test]
    fn typecheck_bool_expr_source() {
        let level_label = Label::top();

        let expr = zspan!(ExprType::Bool(Box::new(zspan!(Bool {
            l: zspan!(ExprType::Bool(Box::new(zspan!(Bool {
                l: zspan!(ExprType::BoolLiteral(true)),
                op: zspan!(BoolOp::And),
                r: zspan!(ExprType::Number(1))
            })))),
            op: zspan!(BoolOp::And),
            r: zspan!(ExprType::BoolLiteral(true)),
        }))));
        let env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, level_label);
        assert_eq!(
            env.typecheck_expr(&expr),
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::TypeMismatch(TypeInfo::bool(), TypeInfo::int()),
                zspan!(),
            )])
        );
    }

    #[test]
    fn typecheck_lval_undefined_var() {
        let level_label = Label::top();

        let env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, level_label);
        let lval = zspan!(LVal::Simple("a".to_owned()));

        assert_eq!(
            env.typecheck_lval(&lval).unwrap_err()[0].t.ty,
            TypecheckErrorType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn typecheck_lval_record_field() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record = vec![(
            "f".to_owned(),
            _RecordField {
                ty: Rc::new(_Type::Int),
            },
        )];

        env.insert_pre_type("r".to_owned(), _Type::Record("r".to_owned(), record), zspan!());
        env.convert_pre_types();
        let ty = env.r#type("r").unwrap().clone();

        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: ty.clone(),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: ty.size(),
                    },
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("f".to_owned())
        ));
        assert_eq!(
            env.typecheck_lval(&lval).map(TranslateOutput::unwrap),
            Ok(LValProperties {
                ty: TypeInfo::int(),
                immutable: None,
            })
        );
    }

    #[test]
    fn typecheck_lval_record_field_err1() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_pre_type("r".to_owned(), _Type::Record("r".to_owned(), vec![]), zspan!());
        env.convert_pre_types();
        let ty = env.r#type("r").unwrap().clone();

        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: ty.clone(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: ty.size(),
                    },
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.typecheck_lval(&lval).map(TranslateOutput::unwrap),
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::UndefinedField("g".to_owned()),
                zspan!(),
            )])
        );
    }

    #[test]
    fn typecheck_lval_record_field_err2() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let ty = TypeInfo::int();

        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: ty.clone(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: ty.size(),
                    },
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.typecheck_lval(&lval).map(TranslateOutput::unwrap),
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::UndefinedField("g".to_owned()),
                zspan!()
            )])
        );
    }

    #[test]
    fn typecheck_array_subscript() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let ty = TypeInfo::new(Rc::new(Type::Array(TypeInfo::int(), 3)), TypeInfo::int().size() * 3);

        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: ty.clone(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: ty.size(),
                    },
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Subscript(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!(ExprType::Number(0))
        ));
        assert_eq!(
            env.typecheck_lval(&lval).map(TranslateOutput::unwrap),
            Ok(LValProperties {
                ty: TypeInfo::int(),
                immutable: Some("x".to_owned())
            })
        );
    }

    #[test]
    fn typecheck_let_type_annotation() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            immutable: zspan!(true),
            ty: Some(zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        assert_eq!(
            env.typecheck_let(&let_expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::unit())
        );
        assert_eq!(env.vars["x"].ty, TypeInfo::int());
        assert_eq!(env.vars["x"].immutable, true);
        assert!(env.var_def_spans.contains_key("x"));
    }

    #[test]
    fn typecheck_let_type_annotation_err() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            immutable: zspan!(true),
            ty: Some(zspan!(ast::Type::Type(zspan!("string".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        assert_eq!(
            env.typecheck_let(&let_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::string(), TypeInfo::int())
        );
    }

    #[test]
    fn typecheck_fn_call_undefined_err() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert_eq!(
            env.typecheck_fn_call(&fn_call_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::UndefinedFn("f".to_owned())
        );
    }

    #[test]
    fn typecheck_fn_call_not_fn_err() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: TypeInfo::int(),
                immutable: true,
                entry_type: EnvEntryType::Fn(label_f),
            },
            zspan!(),
        );
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert_eq!(
            env.typecheck_fn_call(&fn_call_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::NotAFn("f".to_owned())
        );
    }

    #[test]
    fn typecheck_fn_call_arity_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let ty = TypeInfo::new(Rc::new(Type::Fn(vec![], TypeInfo::int())), std::mem::size_of::<u64>());

        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty,
                immutable: true,
                entry_type: EnvEntryType::Fn(label_f),
            },
            zspan!(),
        );
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![zspan!(ExprType::Number(0))],
        });
        assert_eq!(
            env.typecheck_fn_call(&fn_call_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::ArityMismatch(0, 1)
        );
    }

    #[test]
    fn typecheck_fn_call_arg_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        env.insert_pre_type("f", _Type::Fn(vec![Rc::new(_Type::Int)], Rc::new(_Type::Int)), zspan!());
        env.convert_pre_types();
        let ty = env.r#type("f").unwrap().clone();

        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: ty.clone(),
                immutable: true,
                entry_type: EnvEntryType::Fn(label_f),
            },
            zspan!(),
        );
        env.fn_param_decl_spans.insert("f".to_owned(), vec![zspan!()]);

        let fn_call = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![zspan!(ExprType::String("a".to_owned()))],
        });

        assert_eq!(
            env.typecheck_fn_call(&fn_call).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::string())
        )
    }

    #[test]
    fn typecheck_fn_call_returns_aliased_type() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        env.insert_pre_type("a", _Type::Alias("int".to_owned()), zspan!());
        env.insert_pre_type("f", _Type::Fn(vec![], Rc::new(_Type::Alias("a".to_owned()))), zspan!());
        env.convert_pre_types();

        let ty = env.r#type("f").unwrap().clone();

        env.insert_var(
            "f",
            EnvEntry {
                ty: ty.clone(),
                immutable: true,
                entry_type: EnvEntryType::Fn(label_f),
            },
            zspan!(),
        );
        let fn_call = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![]
        });
        assert_eq!(
            env.typecheck_fn_call(&fn_call).unwrap().unwrap(),
            TypeInfo::new(Rc::new(Type::Alias("a".to_owned())), TypeInfo::int().size())
        );
    }

    #[test]
    fn typecheck_typedef() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
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
        let _ = env.typecheck_type_decl(&type_decl);
        env.convert_pre_types();

        assert!(env.typecheck_let(&let_expr).is_ok());
    }

    #[test]
    fn typecheck_typedef_err() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
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
        let _ = env.typecheck_type_decl(&type_decl);
        env.convert_pre_types();

        assert_eq!(
            env.typecheck_let(&let_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(
                TypeInfo::new(Rc::new(Type::Alias("a".to_owned())), TypeInfo::int().size()),
                TypeInfo::string()
            )
        );
    }

    #[test]
    fn typecheck_expr_typedef() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

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
        env.typecheck_type_decl(&type_decl).expect("typecheck type decl");
        env.convert_pre_types();

        env.typecheck_let(&var_def).expect("typecheck var def");
        assert_eq!(
            env.typecheck_expr_mut(&expr).unwrap().unwrap(),
            TypeInfo::new(Rc::new(Type::Alias("i".to_owned())), TypeInfo::int().size())
        );
    }

    #[test]
    fn recursive_typedef() {
        let tmp_generator = TmpGenerator::default();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, Label::top());
        let type_decl = zspan!(DeclType::Type(zspan!(TypeDecl {
            id: zspan!("i".to_owned()),
            ty: zspan!(ast::TypeDeclType::Record(vec![zspan!(TypeField {
                id: zspan!("i".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("i".to_owned())))
            })]))
        })));
        env.typecheck_type_decl_and_check_cycles(&type_decl)
            .expect("typecheck decl");
    }

    #[test]
    fn check_for_type_decl_cycles_err1() {
        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, Label::top());
        env.insert_pre_type("a", _Type::Alias("a".to_owned()), zspan!());

        assert_eq!(
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeDeclCycle("a".to_owned(), Rc::new(_Type::Alias("a".to_owned())))
        );
    }

    #[test]
    fn check_for_type_decl_cycles_err2() {
        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, Label::top());
        env.insert_pre_type("a", _Type::Alias("b".to_owned()), zspan!());
        env.insert_pre_type("b", _Type::Alias("c".to_owned()), zspan!());
        env.insert_pre_type("c", _Type::Alias("a".to_owned()), zspan!());

        assert_eq!(
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeDeclCycle("c".to_owned(), Rc::new(_Type::Alias("a".to_owned())))
        );
    }

    #[test]
    fn check_for_type_decl_cycles() {
        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, Label::top());
        env.insert_pre_type("a", _Type::Alias("b".to_owned()), zspan!());
        env.insert_pre_type("b", _Type::Alias("c".to_owned()), zspan!());

        assert_eq!(env.check_for_type_decl_cycles("a", vec![]), Ok(()));
    }

    #[test]
    fn duplicate_type_decl() {
        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, Label::top());
        env.typecheck_type_decl(&zspan!(ast::TypeDecl {
            id: zspan!("a".to_owned()),
            ty: zspan!(ast::TypeDeclType::Unit)
        }))
        .expect("typecheck type decl");
        env.convert_pre_types();

        assert_eq!(
            env.typecheck_type_decl(&zspan!(ast::TypeDecl {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::TypeDeclType::Unit)
            }))
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::DuplicateType("a".to_owned())
        );
    }

    #[test]
    fn check_for_invalid_types_in_fn_sig() {
        let tmp_generator = TmpGenerator::default();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, Label::top());
        let fn_decl = zspan!(ast::FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![],
            return_type: Some(zspan!(ast::Type::Type(zspan!("a".to_owned())))),
            body: zspan!(ast::ExprType::Unit)
        });

        assert_eq!(
            env.validate_fn_decl_pre_types(&fn_decl).unwrap_err()[0].t.ty,
            TypecheckErrorType::UndefinedType("a".to_owned())
        );
    }

    #[test]
    fn typecheck_first_pass() {
        let tmp_generator = TmpGenerator::default();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, Label::top());
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
            Err(vec![TypecheckError::new_err(
                TypecheckErrorType::TypeDeclCycle("a".to_owned(), Rc::new(_Type::Alias("a".to_owned()))),
                zspan!()
            )])
        );
    }

    #[test]
    fn typecheck_fn_decl_duplicate() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
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
        env.typecheck_fn_decl_sig(&fn_decl1)
            .expect("typecheck function signature");

        assert_eq!(
            env.typecheck_fn_decl_sig(&fn_decl2).unwrap_err()[0].t.ty,
            TypecheckErrorType::DuplicateFn("f".to_owned())
        );
    }

    #[test]
    fn typecheck_fn_decl_duplicate_param() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
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
            env.typecheck_fn_decl_sig(&fn_decl).unwrap_err()[0].t.ty,
            TypecheckErrorType::DuplicateParam("a".to_owned())
        );
    }

    #[test]
    fn typecheck_fn_decl() {
        let tmp_generator = TmpGenerator::default();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, Label::top());
        let fn_decl = zspan!(FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![zspan!(TypeField {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
            }),],
            return_type: None,
            body: zspan!(ExprType::Unit)
        });

        assert_eq!(
            env.typecheck_fn_decl_sig(&fn_decl),
            Ok(TypeInfo::new(
                Rc::new(Type::Fn(vec![TypeInfo::int()], TypeInfo::unit())),
                std::mem::size_of::<u64>()
            ))
        );
        env.typecheck_fn_decl_body(&fn_decl).expect("typecheck fn decl body");

        let label = env.vars["f"].entry_type.fn_label();
        let levels = env.levels.borrow();
        let level = &levels[&label];

        assert_eq!(level.parent_label, Some(Label::top()));
        assert_eq!(
            level.formals(),
            vec![Access {
                level_label: label,
                access: frame::Access {
                    ty: frame::AccessType::InFrame(-8),
                    size: std::mem::size_of::<u64>()
                }
            }]
        );
    }

    #[test]
    fn validate_pre_type() {
        let mut env = Env::new(TmpGenerator::default(), EMPTY_SOURCEMAP.1, Label::top());
        let mut record_fields = vec![];
        record_fields.push((
            "f".to_owned(),
            _RecordField {
                ty: Rc::new(_Type::Alias("a".to_owned())),
            },
        ));
        record_fields.push((
            "g".to_owned(),
            _RecordField {
                ty: Rc::new(_Type::Alias("b".to_owned())),
            },
        ));
        env.insert_pre_type("a".to_owned(), _Type::Record("a".to_owned(), record_fields), zspan!());

        let errs = env.check_for_invalid_pre_types().unwrap_err();
        assert_eq!(errs.len(), 1);
        // Recursive type def in records is allowed
        assert_eq!(errs[0].t.ty, TypecheckErrorType::UndefinedType("b".to_owned()));
    }

    #[test]
    fn typecheck_record_missing_fields() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let record_type = _Type::Record(
            "r".to_owned(),
            vec![(
                "a".to_owned(),
                _RecordField {
                    ty: Rc::new(_Type::Int),
                },
            )],
        );
        env.insert_pre_type("r".to_owned(), record_type, zspan!());
        env.convert_pre_types();

        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![]
        });
        assert_eq!(
            env.typecheck_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrorType::MissingFields(vec!["a".to_owned()])
        );
    }

    #[test]
    fn typecheck_record_invalid_fields() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let record_type = _Type::Record("r".to_owned(), vec![]);
        env.insert_pre_type("r", record_type, zspan!());
        env.convert_pre_types();

        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![zspan!(FieldAssign {
                id: zspan!("b".to_owned()),
                expr: zspan!(ExprType::Number(0))
            })]
        });
        assert_eq!(
            env.typecheck_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrorType::InvalidFields(vec!["b".to_owned()])
        );
    }

    #[test]
    fn typecheck_record_duplicate_field() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let record_type = _Type::Record(
            "r".to_owned(),
            vec![(
                "a".to_owned(),
                _RecordField {
                    ty: Rc::new(_Type::Int),
                },
            )],
        );

        env.insert_pre_type("r".to_owned(), record_type.clone(), zspan!());
        env.convert_pre_types();

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
            env.typecheck_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrorType::DuplicateField("a".to_owned())
        );
    }

    #[test]
    fn typecheck_record() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let record_type = _Type::Record(
            "r".to_owned(),
            vec![
                (
                    "a".to_owned(),
                    _RecordField {
                        ty: Rc::new(_Type::Int),
                    },
                ),
                (
                    "b".to_owned(),
                    _RecordField {
                        ty: Rc::new(_Type::String),
                    },
                ),
            ],
        );
        env.insert_pre_type("r".to_owned(), record_type.clone(), zspan!());
        env.convert_pre_types();

        let ty = env.r#type("r").unwrap().clone();

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

        assert_eq!(env.typecheck_record(&record).unwrap().unwrap(), ty);
    }

    #[test]
    fn typecheck_record_field_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let record_type = _Type::Record(
            "r".to_owned(),
            vec![(
                "a".to_owned(),
                _RecordField {
                    ty: Rc::new(_Type::Int),
                },
            )],
        );
        env.insert_pre_type("r", record_type, zspan!());
        env.convert_pre_types();

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
            env.typecheck_record(&record).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::string())
        );
    }

    #[test]
    fn typecheck_fn_independent() {
        let tmp_generator = TmpGenerator::default();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, Label::top());
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
            env.typecheck_decls(EMPTY_SOURCEMAP.1, &[fn1, fn2]).unwrap_err()[0].t.ty,
            TypecheckErrorType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn typecheck_fn_body() {
        let tmp_generator = TmpGenerator::default();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, Label::top());
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

        assert_eq!(env.typecheck_decls(EMPTY_SOURCEMAP.1, &[fn_decl]), Ok(()));
    }

    #[test]
    fn typecheck_seq_independent() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }
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

        env.typecheck_expr(&seq_expr1).expect("typecheck expr");
        assert_eq!(
            env.typecheck_expr(&seq_expr2).unwrap_err()[0].t.ty,
            TypecheckErrorType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn typecheck_seq_captures_value() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_var(
            "b".to_owned(),
            EnvEntry {
                ty: TypeInfo::string(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: TypeInfo::string().size(),
                    },
                }),
            },
            zspan!(),
        );
        let seq = zspan!(ExprType::Seq(
            vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("b".to_owned())))))],
            true
        ));

        assert_eq!(
            env.typecheck_expr(&seq).map(TranslateOutput::unwrap),
            Ok(TypeInfo::string())
        );
    }

    #[test]
    fn illegal_let_expr() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }
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
            env.typecheck_expr(&expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::IllegalLetExpr
        );
    }

    #[test]
    fn assign_immut_err() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_var(
            "a".to_owned(),
            EnvEntry {
                ty: TypeInfo::int(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: TypeInfo::int().size(),
                    },
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(&zspan!(Assign {
                lval: zspan!(LVal::Simple("a".to_owned())),
                expr: zspan!(ExprType::Number(0))
            }),)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::MutatingImmutable("a".to_owned())
        );
    }

    #[test]
    fn assign_record_field_immut_err() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record_type = _Type::Record(
            "r".to_owned(),
            vec![(
                "a".to_owned(),
                _RecordField {
                    ty: Rc::new(_Type::Int),
                },
            )],
        );
        env.insert_pre_type("r", record_type.clone(), zspan!());
        env.convert_pre_types();
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        let ty = env.r#type("r").unwrap().clone();

        env.insert_var(
            "r".to_owned(),
            EnvEntry {
                ty: ty.clone(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: ty.size(),
                    },
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(&zspan!(Assign {
                lval: zspan!(LVal::Field(
                    Box::new(zspan!(LVal::Simple("r".to_owned()))),
                    zspan!("a".to_owned())
                )),
                expr: zspan!(ExprType::Number(0))
            }),)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::MutatingImmutable("r".to_owned())
        );
    }

    #[test]
    fn assign_record_field_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record_type = _Type::Record(
            "r".to_owned(),
            vec![(
                "a".to_owned(),
                _RecordField {
                    ty: Rc::new(_Type::Int),
                },
            )],
        );
        env.insert_pre_type("r", record_type, zspan!());
        env.convert_pre_types();

        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        let ty = env.r#type("r").unwrap().clone();

        env.insert_var(
            "r".to_owned(),
            EnvEntry {
                ty: ty.clone(),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: ty.size(),
                    },
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(&zspan!(Assign {
                lval: zspan!(LVal::Field(
                    Box::new(zspan!(LVal::Simple("r".to_owned()))),
                    zspan!("a".to_owned())
                )),
                expr: zspan!(ExprType::Unit)
            }),)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::unit())
        );
    }

    #[test]
    fn assign_record_field_type() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record_type = _Type::Record(
            "r".to_owned(),
            vec![(
                "a".to_owned(),
                _RecordField {
                    ty: Rc::new(_Type::Int),
                },
            )],
        );
        env.insert_pre_type("r", record_type, zspan!());
        env.convert_pre_types();
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        let ty = env.r#type("r").unwrap().clone();

        env.insert_var(
            "r",
            EnvEntry {
                ty: ty.clone(),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: ty.size(),
                    },
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(&zspan!(Assign {
                lval: zspan!(LVal::Field(
                    Box::new(zspan!(LVal::Simple("r".to_owned()))),
                    zspan!("a".to_owned())
                )),
                expr: zspan!(ExprType::Number(0))
            }),)
                .map(TranslateOutput::unwrap),
            Ok(TypeInfo::unit())
        );
    }

    #[test]
    fn assign_array_immut_err() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let array_type = TypeInfo::new(Rc::new(Type::Array(TypeInfo::int(), 1)), TypeInfo::int().size());

        env.insert_var(
            "a",
            EnvEntry {
                ty: array_type.clone(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: array_type.size(),
                    },
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(&zspan!(Assign {
                lval: zspan!(LVal::Subscript(
                    Box::new(zspan!(LVal::Simple("a".to_owned()))),
                    zspan!(ExprType::Number(0))
                )),
                expr: zspan!(ExprType::Number(0))
            }),)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::MutatingImmutable("a".to_owned())
        );
    }

    #[test]
    fn assign_array_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let array_type = TypeInfo::new(Rc::new(Type::Array(TypeInfo::int(), 1)), TypeInfo::int().size());

        env.insert_var(
            "a".to_owned(),
            EnvEntry {
                ty: array_type.clone(),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: array_type.size(),
                    },
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(&zspan!(Assign {
                lval: zspan!(LVal::Subscript(
                    Box::new(zspan!(LVal::Simple("a".to_owned()))),
                    zspan!(ExprType::Number(0))
                )),
                expr: zspan!(ExprType::String("s".to_owned()))
            }),)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::string())
        );
    }

    #[test]
    fn assign_array() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }
        let array_type = TypeInfo::new(Rc::new(Type::Array(TypeInfo::int(), 1)), TypeInfo::int().size());

        env.insert_var(
            "a".to_owned(),
            EnvEntry {
                ty: array_type.clone(),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: array_type.size(),
                    },
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(&zspan!(Assign {
                lval: zspan!(LVal::Subscript(
                    Box::new(zspan!(LVal::Simple("a".to_owned()))),
                    zspan!(ExprType::Number(0))
                )),
                expr: zspan!(ExprType::Number(0))
            }),)
                .map(TranslateOutput::unwrap),
            Ok(TypeInfo::unit())
        );
    }

    #[test]
    fn translate_assign_to_fn() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let fn_type = TypeInfo::new(Rc::new(Type::Fn(vec![], TypeInfo::unit())), std::mem::size_of::<u64>());
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: fn_type,
                immutable: true,
                entry_type: EnvEntryType::Fn(Label("f".to_owned())),
            },
            zspan!(),
        );

        assert!(dbg!(env.typecheck_assign(&zspan!(Assign {
            lval: zspan!(LVal::Simple("f".to_owned())),
            expr: zspan!(ExprType::Number(0))
        }),))
        .is_err());
    }

    #[test]
    fn typecheck_array() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Number(3))
        });

        assert_eq!(
            env.typecheck_array(&array_expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::new(
                Rc::new(Type::Array(TypeInfo::int(), 3)),
                TypeInfo::int().size() * 3
            ))
        );
    }

    #[test]
    fn typecheck_array_const_expr_len() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Arith(Box::new(zspan!(Arith {
                l: zspan!(ExprType::Number(1)),
                op: zspan!(ArithOp::Add),
                r: zspan!(ExprType::Number(2)),
            }))))
        });

        assert_eq!(
            env.typecheck_array(&array_expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::new(
                Rc::new(Type::Array(TypeInfo::int(), 3)),
                TypeInfo::int().size() * 3
            ))
        );
    }

    #[test]
    fn typecheck_array_negative_len_err() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Neg(Box::new(zspan!(ExprType::Number(3)))))
        });

        assert_eq!(
            env.typecheck_array(&array_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::NegativeArrayLen(-3)
        );
    }

    #[test]
    fn typecheck_array_non_constant_expr_err() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let fn_call = ExprType::FnCall(Box::new(zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![]
        })));
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(fn_call.clone())
        });

        assert_eq!(
            env.typecheck_array(&array_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::NonConstantArithExpr(fn_call)
        );
    }

    #[test]
    fn typecheck_if_then() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Unit),
            else_expr: None,
        });

        assert_eq!(
            env.typecheck_if(&if_expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::unit())
        );
    }

    #[test]
    fn typecheck_if_then_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: None,
        });

        assert_eq!(
            env.typecheck_if(&if_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::unit(), TypeInfo::int())
        );
    }

    #[test]
    fn typecheck_if_then_else() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: Some(zspan!(ExprType::Number(1))),
        });

        assert_eq!(
            env.typecheck_if(&if_expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::int())
        );
    }

    #[test]
    fn typecheck_if_then_else_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: Some(zspan!(ExprType::String("s".to_owned()))),
        });

        assert_eq!(
            env.typecheck_if(&if_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::string())
        );
    }

    #[test]
    fn typecheck_range() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let range = zspan!(Range {
            lower: zspan!(ExprType::Number(0)),
            upper: zspan!(ExprType::Number(1)),
        });

        assert_eq!(
            env.typecheck_range(&range).map(TranslateOutput::unwrap),
            Ok(TypeInfo::new(
                Rc::new(Type::Iterator(TypeInfo::int())),
                TypeInfo::int().size() * 2
            ))
        );
    }

    #[test]
    fn typecheck_range_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let range = zspan!(Range {
            lower: zspan!(ExprType::Number(0)),
            upper: zspan!(ExprType::String("a".to_owned())),
        });

        assert_eq!(
            env.typecheck_range(&range).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::string())
        );
    }

    #[test]
    fn typecheck_for() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let for_expr = zspan!(For {
            index: zspan!("i".to_owned()),
            range: zspan!(ExprType::Range(Box::new(zspan!(Range {
                lower: zspan!(ExprType::Number(0)),
                upper: zspan!(ExprType::Number(1)),
            })))),
            body: zspan!(ExprType::Seq(
                vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))))],
                false
            ))
        });

        assert_eq!(
            env.typecheck_for(&for_expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::unit())
        );
    }

    #[test]
    fn typecheck_for_range_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let for_expr = zspan!(For {
            index: zspan!("i".to_owned()),
            range: zspan!(ExprType::Number(0)),
            body: zspan!(ExprType::Seq(
                vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))))],
                false
            ))
        });

        assert_eq!(
            env.typecheck_for(&for_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(
                TypeInfo::new(Rc::new(Type::Iterator(TypeInfo::int())), TypeInfo::int().size() * 2),
                TypeInfo::int()
            )
        );
    }

    #[test]
    fn typecheck_for_body_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let for_expr = zspan!(For {
            index: zspan!("i".to_owned()),
            range: zspan!(ExprType::Range(Box::new(zspan!(Range {
                lower: zspan!(ExprType::Number(0)),
                upper: zspan!(ExprType::Number(1)),
            })))),
            body: zspan!(ExprType::Seq(
                vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))))],
                true
            ))
        });

        assert_eq!(
            env.typecheck_for(&for_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::unit(), TypeInfo::int())
        );
    }

    #[test]
    fn typecheck_while() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::BoolLiteral(true)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Unit)], false))
        });

        assert_eq!(
            env.typecheck_while(&while_expr).map(TranslateOutput::unwrap),
            Ok(TypeInfo::unit())
        );
    }

    #[test]
    fn typecheck_while_cond_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::Number(0)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Unit)], false))
        });
        assert_eq!(
            env.typecheck_while(&while_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::bool(), TypeInfo::int())
        );
    }

    #[test]
    fn typecheck_while_body_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::BoolLiteral(true)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Number(0))], true))
        });

        assert_eq!(
            env.typecheck_while(&while_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::unit(), TypeInfo::int())
        );
    }

    #[test]
    fn typecheck_compare() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let compare = zspan!(Compare {
            l: zspan!(ExprType::BoolLiteral(true)),
            op: zspan!(CompareOp::Eq),
            r: zspan!(ExprType::BoolLiteral(true)),
        });

        assert_eq!(
            env.typecheck_compare(&compare).map(TranslateOutput::unwrap),
            Ok(TypeInfo::bool())
        );
    }

    #[test]
    fn typecheck_compare_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let compare = zspan!(Compare {
            l: zspan!(ExprType::Number(0)),
            op: zspan!(CompareOp::Eq),
            r: zspan!(ExprType::BoolLiteral(true)),
        });

        assert_eq!(
            env.typecheck_compare(&compare).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::bool())
        );
    }

    #[test]
    fn typecheck_enum_arity_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        env.insert_pre_type(
            "e",
            _Type::Enum(
                "e".to_owned(),
                hashmap! {
                    "c".to_owned() => _EnumCase {
                        id: "c".to_owned(),
                        params: vec![Rc::new(_Type::Int)]
                    }
                },
            ),
            zspan!(),
        );
        env.convert_pre_types();

        let enum_expr = zspan!(Enum {
            enum_id: zspan!("e".to_owned()),
            case_id: zspan!("c".to_owned()),
            args: zspan!(vec![])
        });

        assert_eq!(
            env.typecheck_enum(&enum_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::ArityMismatch(1, 0)
        );
    }

    #[test]
    fn typecheck_enum_type_mismatch() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        env.insert_pre_type(
            "e",
            _Type::Enum(
                "e".to_owned(),
                hashmap! {
                    "c".to_owned() => _EnumCase {
                        id: "c".to_owned(),
                        params: vec![Rc::new(_Type::Int)]
                    }
                },
            ),
            zspan!(),
        );
        env.convert_pre_types();

        let enum_expr = zspan!(Enum {
            enum_id: zspan!("e".to_owned()),
            case_id: zspan!("c".to_owned()),
            args: zspan!(vec![zspan!(ExprType::String("a".to_owned()))])
        });

        assert_eq!(
            env.typecheck_enum(&enum_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::TypeMismatch(TypeInfo::int(), TypeInfo::string())
        );
    }

    #[test]
    #[should_panic] // FIXME
    fn typecheck_closure() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let closure = zspan!(Closure {
            type_fields: vec![zspan!(TypeField {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
            })],
            body: zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("a".to_owned())))))
        });

        assert_eq!(
            env.typecheck_closure(&closure).map(TranslateOutput::unwrap),
            Ok(TypeInfo::new(
                Rc::new(Type::Fn(vec![TypeInfo::int()], TypeInfo::int())),
                std::mem::size_of::<u64>()
            ))
        );
    }

    #[test]
    #[should_panic] // FIXME
    fn typecheck_closure_captures_value() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }
        env.insert_var(
            "b".to_owned(),
            EnvEntry {
                ty: TypeInfo::string(),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access {
                        ty: frame::AccessType::InFrame(-8),
                        size: TypeInfo::string().size(),
                    },
                }),
            },
            zspan!(),
        );
        let closure = zspan!(Closure {
            type_fields: vec![zspan!(TypeField {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
            })],
            body: zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("b".to_owned())))))
        });

        assert_eq!(
            env.typecheck_closure(&closure).map(TranslateOutput::unwrap),
            Ok(TypeInfo::new(
                Rc::new(Type::Fn(vec![TypeInfo::int()], TypeInfo::string())),
                std::mem::size_of::<u64>()
            ))
        );
    }

    #[test]
    fn typecheck_closure_duplicate_param() {
        let tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label);
        let closure = zspan!(Closure {
            type_fields: vec![
                zspan!(TypeField {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
                }),
                zspan!(TypeField {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("string".to_owned()))),
                })
            ],
            body: zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("a".to_owned())))))
        });

        assert_eq!(
            env.typecheck_closure(&closure).unwrap_err()[0].t.ty,
            TypecheckErrorType::DuplicateParam("a".to_owned())
        );
    }

    #[test]
    fn typecheck_fn_decl_exprs_in_seq_recursive() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let fn_decl1 = zspan!(ExprType::FnDecl(Box::new(zspan!(FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![zspan!(TypeField {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("int".to_owned())))
            })],
            return_type: None,
            body: zspan!(ExprType::FnCall(Box::new(zspan!(FnCall {
                id: zspan!("g".to_owned()),
                args: vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("a".to_owned())))))]
            }))))
        }))));
        let let_expr = zspan!(ExprType::Let(Box::new(zspan!(Let {
            pattern: zspan!(Pattern::String("h".to_owned())),
            immutable: zspan!(true),
            ty: None,
            expr: zspan!(ExprType::Number(0)),
        }))));
        let fn_decl2 = zspan!(ExprType::FnDecl(Box::new(zspan!(FnDecl {
            id: zspan!("g".to_owned()),
            type_fields: vec![zspan!(TypeField {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("int".to_owned())))
            })],
            return_type: None,
            body: zspan!(ExprType::FnCall(Box::new(zspan!(FnCall {
                id: zspan!("f".to_owned()),
                args: vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("h".to_owned())))))]
            }))))
        }))));

        assert_eq!(
            env.typecheck_expr(&zspan!(ExprType::Seq(vec![fn_decl1, let_expr, fn_decl2], false)),)
                .map(TranslateOutput::unwrap),
            Ok(TypeInfo::unit())
        );
    }

    #[test]
    fn typecheck_fn_decl_exprs_in_seq_captures_correctly() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }
        let fn_decl1 = zspan!(ExprType::FnDecl(Box::new(zspan!(FnDecl {
            id: zspan!("f".to_owned()),
            type_fields: vec![zspan!(TypeField {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("int".to_owned())))
            })],
            return_type: None,
            body: zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("h".to_owned())))))
        }))));
        let let_expr = zspan!(ExprType::Let(Box::new(zspan!(Let {
            pattern: zspan!(Pattern::String("h".to_owned())),
            immutable: zspan!(true),
            ty: None,
            expr: zspan!(ExprType::Number(0)),
        }))));

        assert_eq!(
            env.typecheck_expr(&zspan!(ExprType::Seq(vec![fn_decl1, let_expr], false)),)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrorType::UndefinedVar("h".to_owned())
        );
    }

    #[test]
    fn add_static_link() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);

        assert_eq!(level.frame.formals.len(), 1);
        assert_eq!(
            level.frame.formals[0],
            frame::Access {
                ty: frame::AccessType::InFrame(0),
                size: std::mem::size_of::<u64>()
            }
        );
    }

    #[test]
    fn follows_static_links() {
        {
            let mut tmp_generator = TmpGenerator::default();
            let frame = Frame::new(&mut tmp_generator, "f", &[(std::mem::size_of::<u64>(), true)]);
            let label = frame.label.clone();
            let mut levels = hashmap! {
                Label::top() => Level::top(),
                label.clone() => Level {
                    parent_label: Some(Label::top()),
                    frame,
                },
            };
            let level = levels.get_mut(&label).unwrap();
            let local = level.alloc_local(&mut tmp_generator, TypeInfo::int().size(), true);

            assert_eq!(
                Env::translate_simple_var(&levels, &local, &label),
                ir::Expr::BinOp(
                    Box::new(ir::Expr::Tmp(*tmp::FP)),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                )
            );
        }

        {
            let mut tmp_generator = TmpGenerator::default();
            let frame_f = Frame::new(&mut tmp_generator, "f", &[(std::mem::size_of::<u64>(), true)]);
            let label_f = frame_f.label.clone();
            let frame_g = Frame::new(&mut tmp_generator, "g", &[(std::mem::size_of::<u64>(), true)]);
            let label_g = frame_g.label.clone();
            let mut levels = hashmap! {
                Label::top() => Level::top(),
                label_f.clone() => Level {
                    parent_label: Some(Label::top()),
                    frame: frame_f,
                },
                label_g.clone() => Level {
                    parent_label: Some(label_f.clone()),
                    frame: frame_g,
                }
            };
            let level = levels.get_mut(&label_f).unwrap();
            let local = level.alloc_local(&mut tmp_generator, TypeInfo::int().size(), true);

            assert_eq!(
                Env::translate_simple_var(&levels, &local, &label_g),
                ir::Expr::BinOp(
                    Box::new(ir::Expr::Mem(
                        Box::new(ir::Expr::Tmp(*tmp::FP)),
                        std::mem::size_of::<u64>()
                    )),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                )
            );
        }
    }

    #[test]
    fn illegal_break_continue_expr() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator, EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let assign_expr = zspan!(ast::ExprType::Let(Box::new(zspan!(ast::Let {
            pattern: zspan!(ast::Pattern::String("a".to_owned())),
            immutable: zspan!(true),
            ty: None,
            expr: zspan!(ast::ExprType::Continue),
        }))));
        assert_eq!(
            env.typecheck_expr_mut(&assign_expr).unwrap_err()[0].t.ty,
            TypecheckErrorType::IllegalBreakOrContinue
        );
    }
}
