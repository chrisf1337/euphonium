use crate::{
    ast::{
        self, Arith, ArithOp, Array, Assign, Bool, Closure, Compare, Decl, DeclType, Enum, Expr, ExprType, FieldAssign,
        FileSpan, FnCall, FnDecl, For, If, LVal, Let, Pattern, Range, Record, Spanned, TypeDecl, TypeDeclType, While,
    },
    frame::{self, Frame},
    ir,
    tmp::{self, Label, TmpGenerator},
    translate::{self, Access, Level},
};
use codespan::{FileId, Span};
use codespan_reporting;
use itertools::izip;
use maplit::hashmap;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
    str::FromStr,
};

pub type Result<T> = std::result::Result<T, Vec<TypecheckErr>>;

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
pub enum TypecheckErrType {
    /// expected, actual
    TypeMismatch(Rc<Type>, Rc<Type>),
    // The reason we need a separate case for this instead of using TypeMismatch is because we would
    // need to typecheck the arguments passed to the non-function in order to determine the expected
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
}

pub type TypecheckErr = Spanned<_TypecheckErr>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypecheckErr {
    pub ty: TypecheckErrType,
    pub source: Option<Spanned<String>>,
}

impl TypecheckErr {
    fn new_err(ty: TypecheckErrType, span: FileSpan) -> Self {
        TypecheckErr::new(_TypecheckErr { ty, source: None }, span)
    }

    fn with_source(mut self, source: Spanned<String>) -> Self {
        self.source = Some(source);
        self
    }

    pub fn diagnostic(&self, env: &Env) -> codespan_reporting::diagnostic::Diagnostic {
        use TypecheckErrType::*;
        let msg = match &self.ty {
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
        };

        let primary_label = codespan_reporting::diagnostic::Label::new(self.span.file_id, self.span.span, &msg);
        let mut diagnostic = codespan_reporting::diagnostic::Diagnostic::new_error(&msg, primary_label);
        if let Some(source) = self.t.source.as_ref() {
            diagnostic = diagnostic.with_secondary_labels(vec![codespan_reporting::diagnostic::Label::new(
                source.span.file_id,
                source.span.span,
                &source.t,
            )]);
        }
        diagnostic
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordField {
    ty: Rc<Type>,
    index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    Record(String, HashMap<String, RecordField>),
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
            TypeDeclType::Type(ty) => Type::from_str(&ty.t).unwrap(),
            TypeDeclType::Enum(cases) => Type::Enum(
                type_decl.id.t.clone(),
                cases.into_iter().map(|c| (c.t.id.t.clone(), c.t.into())).collect(),
            ),
            TypeDeclType::Record(type_fields) => {
                let mut type_fields_hm = HashMap::new();
                for (i, TypeField { id, ty }) in type_fields.into_iter().map(|tf| tf.t.into()).enumerate() {
                    type_fields_hm.insert(id, RecordField { ty, index: i });
                }
                Type::Record(type_decl.id.t.clone(), type_fields_hm)
            }
            TypeDeclType::Array(ty, len) => Type::Array(Rc::new(ty.t.into()), len.t),
            TypeDeclType::Fn(param_types, return_type) => Type::Fn(
                param_types
                    .into_iter()
                    .map(|param_type| Rc::new(param_type.t.into()))
                    .collect(),
                Rc::new(return_type.t.into()),
            ),
            TypeDeclType::Unit => Type::Unit,
        }
    }
}

impl FromStr for Type {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "int" => Ok(Type::Int),
            "string" => Ok(Type::String),
            "bool" => Ok(Type::Bool),
            _ => Ok(Type::Alias(s.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumCase {
    id: String,
    params: Vec<Rc<Type>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TypeField {
    id: String,
    ty: Rc<Type>,
}

impl From<ast::Type> for Type {
    fn from(ty: ast::Type) -> Self {
        match ty {
            ast::Type::Type(ty) => Type::from_str(&ty.t).unwrap(),
            ast::Type::Array(ty, len) => Type::Array(Rc::new(ty.t.into()), len.t),
            ast::Type::Fn(param_types, return_type) => Type::Fn(
                param_types
                    .into_iter()
                    .map(|param_type| Rc::new(param_type.t.into()))
                    .collect(),
                Rc::new(return_type.t.into()),
            ),
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
    ty: Rc<Type>,
    immutable: bool,
    entry_type: EnvEntryType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LValProperties {
    ty: Rc<Type>,
    /// `None` if mutable, or `Some(var)` if immutable, with `var` being the root immutable variable.
    ///
    /// For example, if `a` is an immutable record with a field named `b`, then the `LValProperties`
    /// for `LVal::Field(a, b)` would have `Some("a")` for its `immutable` field.
    immutable: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Env<'a> {
    pub current_file: FileId,
    parent: Option<&'a Env<'a>>,
    vars: HashMap<String, EnvEntry>,
    types: HashMap<String, Rc<Type>>,
    var_def_spans: HashMap<String, FileSpan>,
    type_def_spans: HashMap<String, FileSpan>,

    record_field_decl_spans: HashMap<String, HashMap<String, FileSpan>>,
    fn_param_decl_spans: HashMap<String, Vec<FileSpan>>,
    enum_case_param_decl_spans: HashMap<String, HashMap<String, Vec<FileSpan>>>,

    pub(crate) levels: Rc<RefCell<HashMap<Label, Level>>>,
}

impl<'a> Env<'a> {
    pub fn new(current_file: FileId) -> Self {
        let mut types = HashMap::new();
        types.insert("int".to_owned(), Rc::new(Type::Int));
        types.insert("string".to_owned(), Rc::new(Type::String));
        let mut type_def_spans = HashMap::new();
        type_def_spans.insert("int".to_owned(), FileSpan::new(current_file, Span::initial()));
        type_def_spans.insert("string".to_owned(), FileSpan::new(current_file, Span::initial()));

        Env {
            current_file,
            parent: None,
            vars: HashMap::new(),
            types,
            var_def_spans: HashMap::new(),
            type_def_spans,

            record_field_decl_spans: HashMap::new(),
            fn_param_decl_spans: HashMap::new(),
            enum_case_param_decl_spans: HashMap::new(),
            levels: Rc::new(RefCell::new(hashmap! {
                Label::top() => Level::top(),
            })),
        }
    }

    fn new_child(&'a self, current_file: FileId) -> Env<'a> {
        Env {
            current_file,
            parent: Some(self),
            vars: HashMap::new(),
            types: HashMap::new(),
            var_def_spans: HashMap::new(),
            type_def_spans: HashMap::new(),

            record_field_decl_spans: HashMap::new(),
            fn_param_decl_spans: HashMap::new(),
            enum_case_param_decl_spans: HashMap::new(),

            levels: self.levels.clone(),
        }
    }

    fn insert_var(&mut self, name: String, entry: EnvEntry, def_span: FileSpan) {
        self.vars.insert(name.clone(), entry);
        self.var_def_spans.insert(name, def_span);
    }

    fn insert_type(&mut self, name: String, ty: Type, def_span: FileSpan) {
        self.types.insert(name.clone(), Rc::new(ty));
        self.type_def_spans.insert(name, def_span);
    }

    fn get_var(&self, name: &str) -> Option<&EnvEntry> {
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

    fn get_var_def_span(&self, id: &str) -> Option<FileSpan> {
        if let Some(span) = self.var_def_spans.get(id) {
            Some(*span)
        } else if let Some(parent) = self.parent {
            parent.get_var_def_span(id)
        } else {
            None
        }
    }

    fn get_type_def_span(&self, id: &str) -> Option<FileSpan> {
        if let Some(span) = self.type_def_spans.get(id) {
            Some(*span)
        } else if let Some(parent) = self.parent {
            parent.get_type_def_span(id)
        } else {
            None
        }
    }

    fn get_record_field_decl_spans(&self, id: &str) -> Option<&HashMap<String, FileSpan>> {
        if let Some(spans) = self.record_field_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.get_record_field_decl_spans(id)
        } else {
            None
        }
    }

    fn get_fn_param_decl_spans(&self, id: &str) -> Option<&[FileSpan]> {
        if let Some(spans) = self.fn_param_decl_spans.get(id) {
            Some(spans)
        } else if let Some(parent) = self.parent {
            parent.get_fn_param_decl_spans(id)
        } else {
            None
        }
    }

    fn get_enum_case_param_decl_spans(&self, id: &str) -> Option<&HashMap<String, Vec<FileSpan>>> {
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
    fn resolve_type(&self, ty: &Rc<Type>, def_span: FileSpan) -> Result<Rc<Type>> {
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

    /// `ty` must have already been resolved.
    fn assert_ty(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        expr: &Expr,
        ty: &Rc<Type>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let translate_output = self.typecheck_expr(tmp_generator, level_label, expr)?;
        let expr_type = &translate_output.t;
        let resolved_expr_type = self.resolve_type(expr_type, expr.span)?;
        if ty != &resolved_expr_type {
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeMismatch(ty.clone(), expr_type.clone()),
                expr.span,
            )])
        } else {
            Ok(translate_output)
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
    fn validate_type(&self, ty: &Rc<Type>, def_span: FileSpan) -> Result<()> {
        match ty.as_ref() {
            Type::String | Type::Bool | Type::Unit | Type::Int | Type::Iterator(_) => Ok(()),
            Type::Alias(_) => self.resolve_type(ty, def_span).map(|_| ()),
            Type::Record(_, record) => {
                let mut errors = vec![];
                for RecordField { ty, .. } in record.values() {
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
    ) -> Result<HashMap<Key, FileSpan>>
    where
        I: IntoIterator<Item = &'b Spanned<T>>,
        KeyFn: Fn(&'b Spanned<T>) -> Key,
        ErrGen: Fn(&'b Spanned<T>, FileSpan) -> TypecheckErr,
    {
        // t -> def span
        let mut checked_elems: HashMap<Key, FileSpan> = HashMap::new();
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

    pub fn typecheck_decls(&mut self, tmp_generator: &mut TmpGenerator, file_id: FileId, decls: &[Decl]) -> Result<()> {
        self.current_file = file_id;
        self.first_pass(tmp_generator, decls)?;
        self.second_pass(tmp_generator, decls)
    }

    fn first_pass(&mut self, tmp_generator: &mut TmpGenerator, decls: &[Decl]) -> Result<()> {
        let mut errors = vec![];
        let mut found_cycle = false;
        for decl in decls {
            match self.typecheck_decl_first_pass(tmp_generator, decl) {
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

    fn second_pass(&self, tmp_generator: &mut TmpGenerator, decls: &[Decl]) -> Result<()> {
        let mut errors = vec![];
        for Decl { t: decl, .. } in decls {
            if let DeclType::Fn(fn_decl) = decl {
                match self.typecheck_fn_decl_body(tmp_generator, fn_decl) {
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

    fn typecheck_decl_first_pass(&mut self, tmp_generator: &mut TmpGenerator, decl: &Decl) -> Result<()> {
        match &decl.t {
            DeclType::Fn(fn_decl) => {
                self.typecheck_fn_decl_sig(tmp_generator, &Label::top(), fn_decl)?;
                Ok(())
            }
            DeclType::Type(type_decl) => {
                self.typecheck_type_decl(type_decl)?;
                self.check_for_type_decl_cycles(&type_decl.id, vec![])
            }
            _ => unreachable!(), // DeclType::Error
        }
    }

    fn typecheck_fn_decl_sig(
        &mut self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        fn_decl: &Spanned<FnDecl>,
    ) -> Result<Rc<Type>> {
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

        let ty = Rc::new(Type::Fn(param_types, Rc::new(return_type)));
        // FIXME: Don't assume all formals escape
        let formals = vec![true; fn_decl.type_fields.len()];
        let level = Level::new(tmp_generator, Some(level_label.clone()), &fn_decl.id.t, &formals);

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
    fn typecheck_fn_decl_body(&self, tmp_generator: &mut TmpGenerator, fn_decl: &Spanned<FnDecl>) -> Result<()> {
        let mut new_env = self.new_child(self.current_file);

        let (fn_type, formals, label) = {
            let env_entry = new_env.get_var(&fn_decl.id.t).unwrap();
            let fn_type = env_entry.ty.clone();
            // A level should have already been created by typecheck_fn_decl_sig().
            let label = env_entry.entry_type.fn_label();
            let levels = self.levels.borrow();
            let formals = levels[&label].formals();

            (fn_type, formals, label)
        };

        if let Type::Fn(param_types, return_type) = fn_type.as_ref() {
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

            let body_type = new_env.typecheck_expr(tmp_generator, &label, &fn_decl.body)?.t;
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
            panic!(format!("expected {} to be a function", fn_decl.id.t));
        }

        Ok(())
    }

    fn typecheck_type_decl(&mut self, decl: &Spanned<TypeDecl>) -> Result<()> {
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
                let field_def_spans: HashMap<String, FileSpan> = Self::check_for_duplicates(
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
                    .or_insert_with(HashMap::new);
                for case in cases {
                    decl_spans.insert(case.id.t.clone(), case.params.iter().map(|param| param.span).collect());
                }
            }
            _ => (),
        }
        self.insert_type(id, ty, decl.span);

        Ok(())
    }

    pub(crate) fn typecheck_expr(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        expr: &Expr,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        match &expr.t {
            ExprType::Seq(exprs, returns) => {
                let (mut new_env, return_type) = {
                    // New scope
                    let mut new_env = self.new_child(self.current_file);
                    for expr in &exprs[..exprs.len() - 1] {
                        let _ = new_env.typecheck_expr_mut(tmp_generator, level_label, expr)?;
                    }
                    let last_expr = exprs.last().unwrap();
                    let return_type = if *returns {
                        new_env.typecheck_expr_mut(tmp_generator, level_label, last_expr)?.t
                    } else {
                        let _ = new_env.typecheck_expr_mut(tmp_generator, level_label, last_expr)?;
                        Rc::new(Type::Unit)
                    };

                    // Remove all non-fn decl exprs so that in the second pass they'll already be
                    // defined.
                    let mut fn_decls: HashMap<String, EnvEntry> = HashMap::new();
                    for (id, var) in &new_env.vars {
                        if let Type::Fn(..) = var.ty.as_ref() {
                            fn_decls.insert(id.clone(), var.clone());
                        }
                    }

                    new_env.vars = fn_decls;
                    (new_env, return_type)
                };

                // The only possible place where inline fn decl exprs can be is inside seq exprs
                // since that's the only place where typecheck_expr_mut() is called. We need to
                // typecheck all the expressions again in order to make sure that variables are
                // defined before they get captured.
                for expr in exprs {
                    if let ExprType::FnDecl(fn_decl) = &expr.t {
                        new_env.typecheck_fn_decl_body(tmp_generator, fn_decl)?;
                    } else {
                        new_env.typecheck_expr_mut(tmp_generator, level_label, expr)?;
                    }
                }

                Ok(TranslateOutput {
                    t: return_type,
                    expr: translate::Expr::Expr(ir::Expr::Const(0)),
                })
            }
            ExprType::String(_) => Ok(TranslateOutput {
                t: Rc::new(Type::String),
                expr: translate::Expr::Expr(ir::Expr::Const(0)),
            }),
            ExprType::Number(n) => Ok(TranslateOutput {
                t: Rc::new(Type::Int),
                expr: translate::Expr::Expr(ir::Expr::Const(*n as i64)),
            }),
            ExprType::Neg(expr) => {
                let TranslateOutput { expr, .. } =
                    self.assert_ty(tmp_generator, level_label, expr, &Rc::new(Type::Int))?;
                Ok(TranslateOutput {
                    t: Rc::new(Type::Int),
                    expr: translate::Expr::Expr(ir::Expr::BinOp(
                        Box::new(ir::Expr::Const(-1)),
                        ir::BinOp::Mul,
                        Box::new(expr.unwrap_expr(tmp_generator)),
                    )),
                })
            }
            ExprType::Arith(arith) => self.typecheck_arith(tmp_generator, level_label, arith),
            ExprType::Unit | ExprType::Continue | ExprType::Break => Ok(TranslateOutput {
                t: Rc::new(Type::Unit),
                expr: translate::Expr::Expr(ir::Expr::Const(0)),
            }),
            ExprType::BoolLiteral(b) => Ok(TranslateOutput {
                t: Rc::new(Type::Bool),
                expr: translate::Expr::Expr(ir::Expr::Const(if *b { 1 } else { 0 })),
            }),
            ExprType::Not(expr) => self.assert_ty(tmp_generator, level_label, expr, &Rc::new(Type::Bool)),
            ExprType::Bool(bool_expr) => self.typecheck_bool(tmp_generator, level_label, bool_expr),
            ExprType::LVal(lval) => Ok(self
                .typecheck_lval(tmp_generator, level_label, lval)?
                .map(|lval_props| lval_props.ty)),
            ExprType::Let(_) => Err(vec![TypecheckErr::new_err(TypecheckErrType::IllegalLetExpr, expr.span)]),
            ExprType::FnCall(fn_call) => self.typecheck_fn_call(tmp_generator, level_label, fn_call),
            ExprType::Record(record) => self.typecheck_record(tmp_generator, level_label, record),
            ExprType::Assign(assign) => self.typecheck_assign(tmp_generator, level_label, assign),
            ExprType::Array(array) => self.typecheck_array(tmp_generator, level_label, array),
            ExprType::If(if_expr) => self.typecheck_if(tmp_generator, level_label, if_expr),
            ExprType::Range(range) => self.typecheck_range(tmp_generator, level_label, range),
            ExprType::For(for_expr) => self.typecheck_for(tmp_generator, level_label, for_expr),
            ExprType::While(while_expr) => self.typecheck_while(tmp_generator, level_label, while_expr),
            ExprType::Compare(compare) => self.typecheck_compare(tmp_generator, level_label, compare),
            ExprType::Enum(enum_expr) => self.typecheck_enum(tmp_generator, level_label, enum_expr),
            ExprType::Closure(closure) => self.typecheck_closure(tmp_generator, level_label, closure),
            ExprType::FnDecl(_) => Err(vec![TypecheckErr::new_err(
                TypecheckErrType::IllegalFnDeclExpr,
                expr.span,
            )]),
        }
    }

    pub(crate) fn typecheck_expr_mut(
        &mut self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        expr: &Expr,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        match &expr.t {
            ExprType::Let(let_expr) => self.typecheck_let(tmp_generator, level_label, let_expr),
            ExprType::FnDecl(fn_decl) => self.typecheck_fn_decl_expr(tmp_generator, level_label, fn_decl),
            _ => self.typecheck_expr(tmp_generator, level_label, expr),
        }
    }

    fn typecheck_lval(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        lval: &Spanned<LVal>,
    ) -> Result<TranslateOutput<LValProperties>> {
        match &lval.t {
            LVal::Simple(var) => {
                if let Some(env_entry) = self.get_var(var) {
                    let trexpr = if let EnvEntryType::Var(access) = &env_entry.entry_type {
                        let levels = self.levels.borrow();
                        Env::translate_simple_var(&levels, access, level_label)
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
                    Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::UndefinedVar(var.clone()),
                        lval.span,
                    )])
                }
            }
            LVal::Field(var, field) => {
                let TranslateOutput {
                    t: lval_properties,
                    expr: record_trexpr,
                } = self.typecheck_lval(tmp_generator, level_label, var)?;
                if let Type::Record(_, fields) = self.resolve_type(&lval_properties.ty, var.span)?.as_ref() {
                    if let Some(RecordField { ty: field_type, index }) = fields.get(&field.t) {
                        // A field is mutable only if the record it belongs to is mutable
                        let trexpr = Env::translate_pointer_offset(
                            tmp_generator,
                            &record_trexpr,
                            &translate::Expr::Expr(ir::Expr::Const(*index as i64)),
                        );
                        return Ok(TranslateOutput {
                            t: LValProperties {
                                ty: field_type.clone(),
                                immutable: lval_properties.immutable,
                            },
                            expr: trexpr,
                        });
                    }
                }
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::UndefinedField(field.t.clone()),
                    field.span,
                )])
            }
            LVal::Subscript(var, index) => {
                let TranslateOutput {
                    t: lval_properties,
                    expr: lval_trexpr,
                } = self.typecheck_lval(tmp_generator, level_label, var)?;
                let TranslateOutput {
                    t: index_type,
                    expr: index_trexpr,
                } = self.typecheck_expr(tmp_generator, level_label, index)?;
                if let Type::Array(ty, len) = self.resolve_type(&lval_properties.ty, var.span)?.as_ref() {
                    if self.resolve_type(&index_type, index.span)? == Rc::new(Type::Int) {
                        let index_tmp_trexpr = ir::Expr::Tmp(tmp_generator.new_tmp());
                        let true_label_1 = tmp_generator.new_label();
                        let true_label_2 = tmp_generator.new_label();
                        let false_label = tmp_generator.new_label();
                        let join_label = tmp_generator.new_label();
                        let bounds_check_stmt = vec![
                            ir::Stmt::Move(index_tmp_trexpr.clone(), index_trexpr.unwrap_expr(tmp_generator)),
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

                        Ok(TranslateOutput {
                            t: LValProperties {
                                ty: ty.clone(),
                                immutable: lval_properties.immutable,
                            },
                            expr: translate::Expr::Expr(ir::Expr::Seq(
                                Box::new(ir::Stmt::seq(bounds_check_stmt)),
                                Box::new(
                                    Env::translate_pointer_offset(
                                        tmp_generator,
                                        &lval_trexpr,
                                        &translate::Expr::Expr(index_tmp_trexpr),
                                    )
                                    .unwrap_expr(tmp_generator),
                                ),
                            )),
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

    fn typecheck_let(
        &mut self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        let_expr: &Spanned<Let>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let Let {
            pattern,
            ty: ast_ty,
            immutable,
            expr,
        } = &let_expr.t;
        let TranslateOutput {
            t: expr_type,
            expr: assigned_val_trexpr,
        } = self.typecheck_expr(tmp_generator, level_label, expr)?;
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
            // FIXME: Don't assume all locals escape
            let local = {
                let mut levels = self.levels.borrow_mut();
                let level = levels.get_mut(&level_label).unwrap();
                level.alloc_local(tmp_generator, true)
            };

            self.insert_var(
                var_name.clone(),
                EnvEntry {
                    ty,
                    immutable: immutable.t,
                    entry_type: EnvEntryType::Var(local.clone()),
                },
                let_expr.span,
            );
            let levels = self.levels.borrow();
            let assignee_trexpr = Env::translate_simple_var(&levels, &local, level_label);
            Ok(TranslateOutput {
                t: Rc::new(Type::Unit),
                expr: translate::Expr::Stmt(ir::Stmt::Move(
                    assignee_trexpr.unwrap_expr(tmp_generator),
                    assigned_val_trexpr.unwrap_expr(tmp_generator),
                )),
            })
        } else {
            Ok(TranslateOutput {
                t: Rc::new(Type::Unit),
                expr: translate::Expr::Stmt(ir::Stmt::Seq(
                    Box::new(assigned_val_trexpr.unwrap_stmt(tmp_generator)),
                    Box::new(ir::Stmt::Expr(ir::Expr::Const(0))),
                )),
            })
        }
    }

    fn typecheck_fn_call(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned {
            t: FnCall { id, args },
            span,
        }: &Spanned<FnCall>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        if let Some(fn_type) = self.get_var(&id.t).map(|x| x.ty.clone()) {
            if let Type::Fn(param_types, return_type) = self.resolve_type(&fn_type, id.span)?.as_ref() {
                if args.len() != param_types.len() {
                    return Err(vec![TypecheckErr::new_err(
                        TypecheckErrType::ArityMismatch(param_types.len(), args.len()),
                        *span,
                    )]);
                }

                let mut errors = vec![];
                for (index, (arg, param_type)) in args.iter().zip(param_types.iter()).enumerate() {
                    match self.typecheck_expr(tmp_generator, level_label, arg) {
                        Ok(TranslateOutput { t: ty, .. }) => {
                            // param_type should already be well-defined because we have already
                            // checked for invalid types
                            if self.resolve_type(&ty, arg.span)?
                                != self
                                    .resolve_type(param_type, FileSpan::new(self.current_file, Span::initial()))
                                    .unwrap()
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

                Ok(TranslateOutput {
                    t: return_type.clone(),
                    expr: translate::Expr::Expr(ir::Expr::Const(0)),
                })
            } else {
                Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::NotAFn(id.t.clone()),
                    id.span,
                )])
            }
        } else {
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedFn(id.t.clone()),
                *span,
            )])
        }
    }

    fn typecheck_record(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned {
            t: Record {
                id: record_id,
                field_assigns,
            },
            span,
            ..
        }: &Spanned<Record>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
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

                let record_malloc_trexpr =
                    Frame::external_call("__malloc", vec![ir::Expr::Const(field_types.len() as i64)]);
                let record_tmp_trexpr = ir::Expr::Tmp(tmp_generator.new_tmp());
                let mut assign_stmts = vec![ir::Stmt::Move(record_tmp_trexpr.clone(), record_malloc_trexpr)];
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
                            FileSpan::new(self.current_file, Span::initial()),
                        )
                        .unwrap();
                    let TranslateOutput { t: ty, expr: trexpr } =
                        self.typecheck_expr(tmp_generator, level_label, expr)?;
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

                    let field_index = field_types[&field_id.t].index;
                    let field_trexpr = Env::translate_pointer_offset(
                        tmp_generator,
                        &translate::Expr::Expr(record_tmp_trexpr.clone()),
                        &translate::Expr::Expr(ir::Expr::Const(field_index as i64)),
                    );
                    assign_stmts.push(ir::Stmt::Move(
                        field_trexpr.unwrap_expr(tmp_generator),
                        trexpr.unwrap_expr(tmp_generator),
                    ));
                }

                if !errors.is_empty() {
                    return Err(errors);
                }

                Ok(TranslateOutput {
                    t: ty.clone(),
                    expr: translate::Expr::Expr(ir::Expr::Seq(
                        Box::new(ir::Stmt::seq(assign_stmts)),
                        Box::new(record_tmp_trexpr),
                    )),
                })
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

    fn typecheck_assign(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: assign, span }: &Spanned<Assign>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let TranslateOutput {
            t: lval_properties,
            expr: assignee_trexpr,
        } = self.typecheck_lval(tmp_generator, level_label, &assign.lval)?;
        // Make sure we don't mutate an immutable var.
        if let Some(root) = lval_properties.immutable.as_ref() {
            let def_span = self.get_var_def_span(root).unwrap();
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::MutatingImmutable(root.clone()),
                assign.lval.span,
            )
            .with_source(Spanned::new(format!("{} was defined here", root.clone()), def_span))]);
        }

        let TranslateOutput {
            t: assigned_val_ty,
            expr: assigned_val_trexpr,
        } = self.typecheck_expr(tmp_generator, level_label, &assign.expr)?;
        let resolved_actual_ty = self.resolve_type(&assigned_val_ty, assign.expr.span)?;
        let resolved_expected_ty = self.resolve_type(&lval_properties.ty, assign.lval.span)?;
        if resolved_expected_ty != resolved_actual_ty {
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeMismatch(resolved_expected_ty, resolved_actual_ty),
                *span,
            )]);
        }

        let trexpr = translate::Expr::Stmt(ir::Stmt::Move(
            assignee_trexpr.unwrap_expr(tmp_generator),
            assigned_val_trexpr.unwrap_expr(tmp_generator),
        ));

        Ok(TranslateOutput {
            t: Rc::new(Type::Unit),
            expr: trexpr,
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
                }
            }
            _ => Err(vec![TypecheckErr::new_err(
                TypecheckErrType::NonConstantArithExpr(expr.clone()),
                *span,
            )]),
        }
    }

    fn typecheck_array(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: array, .. }: &Spanned<Array>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let TranslateOutput {
            t: elem_type,
            expr: init_val_trexpr,
        } = self.typecheck_expr(tmp_generator, level_label, &array.initial_value)?;
        let len = Self::eval_arith_const_expr(&array.len)?;
        if len < 0 {
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::NegativeArrayLen(len),
                array.len.span,
            )]);
        }

        let array_malloc_trexpr = Frame::external_call("__malloc", vec![ir::Expr::Const(len as i64)]);
        let array_tmp_trexpr = ir::Expr::Tmp(tmp_generator.new_tmp());

        let mut assign_stmts = vec![ir::Stmt::Move(array_tmp_trexpr.clone(), array_malloc_trexpr)];
        for i in 0..len {
            let elem_trexpr = Env::translate_pointer_offset(
                tmp_generator,
                &translate::Expr::Expr(array_tmp_trexpr.clone()),
                &translate::Expr::Expr(ir::Expr::Const(i as i64)),
            );
            assign_stmts.push(ir::Stmt::Move(
                elem_trexpr.unwrap_expr(tmp_generator),
                init_val_trexpr.clone().unwrap_expr(tmp_generator),
            ));
        }

        Ok(TranslateOutput {
            t: Rc::new(Type::Array(elem_type, len as usize)),
            expr: translate::Expr::Expr(ir::Expr::Seq(
                Box::new(ir::Stmt::seq(assign_stmts)),
                Box::new(array_tmp_trexpr),
            )),
        })
    }

    fn typecheck_if(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, span }: &Spanned<If>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let true_label = tmp_generator.new_label();
        let false_label = tmp_generator.new_label();
        let join_label = tmp_generator.new_label();
        let result = tmp_generator.new_tmp();
        let cond_gen = self
            .assert_ty(tmp_generator, level_label, &expr.cond, &Rc::new(Type::Bool))?
            .expr
            .unwrap_cond();

        let mut injected_labels = None;
        let mut then_expr_was_cond = false;
        let mut else_expr_was_cond = false;

        let TranslateOutput {
            t: then_expr_type,
            expr: translated_then_expr,
        } = self.typecheck_expr(tmp_generator, level_label, &expr.then_expr)?;
        let then_instr = match translated_then_expr {
            translate::Expr::Stmt(stmt) => {
                // If the then expression is a statement, don't bother unwrapping into an Expr::Expr and just run it
                // directly.
                stmt
            }
            translate::Expr::Cond(cond_gen) => {
                let true_label = tmp_generator.new_label();
                let false_label = tmp_generator.new_label();
                let stmt = ir::Stmt::seq(vec![cond_gen(true_label.clone(), false_label.clone())]);
                injected_labels = Some((true_label, false_label));
                then_expr_was_cond = true;
                stmt
            }
            expr => ir::Stmt::Move(ir::Expr::Tmp(result), expr.unwrap_expr(tmp_generator)),
        };

        let else_instr = if let Some(else_expr) = &expr.else_expr {
            let TranslateOutput {
                t: else_expr_type,
                expr: translated_else_expr,
            } = self.typecheck_expr(tmp_generator, level_label, else_expr)?;
            if self.resolve_type(&then_expr_type, expr.then_expr.span)?
                != self.resolve_type(&else_expr_type, else_expr.span)?
            {
                return Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::TypeMismatch(then_expr_type.clone(), else_expr_type.clone()),
                    *span,
                )
                .with_source(Spanned::new(
                    format!(
                        "then branch has type {:?}, but else branch has type {:?}",
                        then_expr_type, else_expr_type
                    ),
                    *span,
                ))]);
            } else {
                Some(match translated_else_expr {
                    translate::Expr::Stmt(stmt) => {
                        // If the then expression is a statement, don't bother unwrapping into an Expr::Expr and just run it
                        // directly.
                        stmt
                    }
                    translate::Expr::Cond(cond_gen) => {
                        let (true_label, false_label) = injected_labels
                            .get_or_insert_with(|| (tmp_generator.new_label(), tmp_generator.new_label()));
                        let stmt = ir::Stmt::seq(vec![cond_gen(true_label.clone(), false_label.clone())]);
                        else_expr_was_cond = true;
                        stmt
                    }
                    expr => ir::Stmt::Move(ir::Expr::Tmp(result), expr.unwrap_expr(tmp_generator)),
                })
            }
        } else {
            if then_expr_type != Rc::new(Type::Unit) {
                return Err(vec![TypecheckErr::new_err(
                    TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), then_expr_type),
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
                Rc::new(Type::Unit)
            },
            expr: translate::Expr::Expr(ir::Expr::Seq(
                Box::new(ir::Stmt::seq(seq)),
                Box::new(ir::Expr::Tmp(result)),
            )),
        })
    }

    fn typecheck_range(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, .. }: &Spanned<Range>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        self.assert_ty(tmp_generator, level_label, &expr.start, &Rc::new(Type::Int))?;
        self.assert_ty(tmp_generator, level_label, &expr.end, &Rc::new(Type::Int))?;
        Ok(TranslateOutput {
            t: Rc::new(Type::Iterator(Rc::new(Type::Int))),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    fn typecheck_for(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, .. }: &Spanned<For>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        self.assert_ty(
            tmp_generator,
            level_label,
            &expr.range,
            &Rc::new(Type::Iterator(Rc::new(Type::Int))),
        )?;
        let mut child_env = self.new_child(self.current_file);
        let local = {
            let mut levels = child_env.levels.borrow_mut();
            let level = levels.get_mut(&level_label).unwrap();
            level.alloc_local(tmp_generator, true)
        };

        child_env.insert_var(
            expr.index.t.clone(),
            EnvEntry {
                ty: Rc::new(Type::Int),
                immutable: false,
                // FIXME
                entry_type: EnvEntryType::Var(local),
            },
            expr.index.span,
        );
        child_env.assert_ty(tmp_generator, level_label, &expr.body, &Rc::new(Type::Unit))?;
        Ok(TranslateOutput {
            t: Rc::new(Type::Unit),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    fn typecheck_while(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, .. }: &Spanned<While>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        self.assert_ty(tmp_generator, level_label, &expr.cond, &Rc::new(Type::Bool))?;
        self.assert_ty(tmp_generator, level_label, &expr.body, &Rc::new(Type::Unit))?;
        Ok(TranslateOutput {
            t: Rc::new(Type::Unit),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    fn typecheck_arith(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, .. }: &Spanned<Arith>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let TranslateOutput { expr: l_expr, .. } =
            self.assert_ty(tmp_generator, level_label, &expr.l, &Rc::new(Type::Int))?;
        let TranslateOutput { expr: r_expr, .. } =
            self.assert_ty(tmp_generator, level_label, &expr.r, &Rc::new(Type::Int))?;
        Ok(TranslateOutput {
            t: Rc::new(Type::Int),
            expr: translate::Expr::Expr(ir::Expr::BinOp(
                Box::new(l_expr.unwrap_expr(tmp_generator)),
                expr.op.t.into(),
                Box::new(r_expr.unwrap_expr(tmp_generator)),
            )),
        })
    }

    fn typecheck_bool(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, .. }: &Spanned<Bool>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        self.assert_ty(tmp_generator, level_label, &expr.l, &Rc::new(Type::Bool))?;
        self.assert_ty(tmp_generator, level_label, &expr.r, &Rc::new(Type::Bool))?;
        Ok(TranslateOutput {
            t: Rc::new(Type::Bool),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    fn typecheck_compare(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, span }: &Spanned<Compare>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let TranslateOutput { t: l_ty, expr: l_expr } = self.typecheck_expr(tmp_generator, level_label, &expr.l)?;
        let TranslateOutput { t: r_ty, expr: r_expr } = self.typecheck_expr(tmp_generator, level_label, &expr.r)?;
        let left_type = self.resolve_type(&l_ty, expr.l.span)?;
        let right_type = self.resolve_type(&r_ty, expr.r.span)?;
        if left_type != right_type {
            return Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeMismatch(left_type.clone(), right_type.clone()),
                *span,
            )]);
        }

        let l_expr = l_expr.unwrap_expr(tmp_generator).clone();
        let r_expr = r_expr.unwrap_expr(tmp_generator).clone();

        let expr = if left_type == Rc::new(Type::String) {
            translate::Expr::Expr(Frame::external_call("__strcmp", vec![l_expr, r_expr]))
        } else {
            let op = expr.op.t.into();
            translate::Expr::Cond(Rc::new(move |true_label, false_label| {
                ir::Stmt::CJump(l_expr.clone(), op, r_expr.clone(), true_label, false_label)
            }))
        };

        Ok(TranslateOutput {
            t: Rc::new(Type::Bool),
            expr,
        })
    }

    fn typecheck_enum(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, .. }: &Spanned<Enum>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
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
                        match self.typecheck_expr(tmp_generator, level_label, arg) {
                            Ok(TranslateOutput { t: ty, .. }) => {
                                // param_type should already be well-defined because we have already
                                // checked for invalid types
                                if self.resolve_type(&ty, arg.span)?
                                    != self
                                        .resolve_type(param_type, FileSpan::new(self.current_file, Span::initial()))
                                        .unwrap()
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

                    Ok(TranslateOutput {
                        t: ty.clone(),
                        expr: translate::Expr::Expr(ir::Expr::Const(0)),
                    })
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

    fn typecheck_closure(
        &self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        Spanned { t: expr, .. }: &Spanned<Closure>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        let child_env = self.new_child(self.current_file);
        Self::check_for_duplicates(
            &expr.type_fields,
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
        )?;

        let mut param_types = vec![];
        let mut errors = vec![];
        for type_field in &expr.type_fields {
            let ty = Rc::new(type_field.ty.t.clone().into());
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

        let return_type = child_env.typecheck_expr(tmp_generator, level_label, &expr.body)?.t;
        if let Err(errs) = child_env.resolve_type(&return_type, expr.body.span) {
            errors.extend(errs);
        }

        if !errors.is_empty() {
            return Err(errors);
        }
        Ok(TranslateOutput {
            t: Rc::new(Type::Fn(param_types, return_type)),
            expr: translate::Expr::Expr(ir::Expr::Const(0)),
        })
    }

    fn typecheck_fn_decl_expr(
        &mut self,
        tmp_generator: &mut TmpGenerator,
        level_label: &Label,
        fn_decl: &Spanned<FnDecl>,
    ) -> Result<TranslateOutput<Rc<Type>>> {
        // At this point, we have already typechecked all type decls, so we can validate the param
        // and return types.
        if let Type::Fn(param_types, return_type) = self
            .typecheck_fn_decl_sig(tmp_generator, level_label, fn_decl)?
            .as_ref()
        {
            let mut errors = vec![];
            for (param_index, param_type) in param_types.iter().enumerate() {
                if let Err(errs) = self.validate_type(param_type, fn_decl.type_fields[param_index].span) {
                    errors.extend(errs);
                }
                if let Some(return_type_decl) = &fn_decl.return_type {
                    if let Err(errs) = self.validate_type(return_type, return_type_decl.span) {
                        errors.extend(errs);
                    }
                }
            }

            // Defer typechecking function body until all function declaration signatures have been
            // typechecked. This allows for mutually recursive functions.
            if !errors.is_empty() {
                Err(errors)
            } else {
                Ok(TranslateOutput {
                    t: Rc::new(Type::Unit),
                    expr: translate::Expr::Expr(ir::Expr::Const(0)),
                })
            }
        } else {
            panic!("expected function type");
        }
    }

    pub fn translate_simple_var(
        levels: &HashMap<Label, Level>,
        access: &Access,
        level_label: &Label,
    ) -> translate::Expr {
        let def_level_label = &access.level_label;
        let def_level = &levels[def_level_label];
        let current_level = &levels[level_label];
        let static_link = Env::static_link(levels, current_level, def_level);
        translate::Expr::Expr(Frame::expr(access.access, &static_link))
    }

    /// `array_expr` is a dereferenced pointer to memory on the heap where the first element of the array lives.
    pub fn translate_pointer_offset(
        tmp_generator: &mut TmpGenerator,
        array_expr: &translate::Expr,
        index_expr: &translate::Expr,
    ) -> translate::Expr {
        translate::Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
            Box::new(array_expr.clone().unwrap_expr(tmp_generator)),
            ir::BinOp::Add,
            Box::new(ir::Expr::BinOp(
                Box::new(index_expr.clone().unwrap_expr(tmp_generator)),
                ir::BinOp::Mul,
                Box::new(ir::Expr::Const(frame::WORD_SIZE)),
            )),
        ))))
    }

    pub fn static_link(levels: &HashMap<Label, Level>, from_level: &Level, to_level: &Level) -> ir::Expr {
        if from_level == to_level {
            ir::Expr::Tmp(*tmp::FP)
        } else {
            let parent_label = from_level.parent_label.as_ref().unwrap();
            let parent_level = &levels[&parent_label];
            ir::Expr::Mem(Box::new(Env::static_link(levels, parent_level, to_level)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast::{self, BoolOp, CompareOp, TypeField},
        frame,
        utils::EMPTY_SOURCEMAP,
    };
    use pretty_assertions::assert_eq;

    #[test]
    fn test_resolve_type() {
        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let var_properties = EnvEntry {
            ty: Rc::new(Type::Int),
            immutable: true,
            entry_type: EnvEntryType::Var(Access {
                level_label: Label::top(),
                access: frame::Access::InFrame(-8),
            }),
        };
        env.insert_var("a".to_owned(), var_properties.clone(), zspan!());
        let mut child_env = env.new_child(env.current_file);
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
    fn test_typecheck_bool_expr() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let expr = zspan!(ExprType::Bool(Box::new(zspan!(Bool {
            l: zspan!(ExprType::BoolLiteral(true)),
            op: zspan!(BoolOp::And),
            r: zspan!(ExprType::BoolLiteral(true)),
        }))));
        let env = Env::new(EMPTY_SOURCEMAP.1);
        assert_eq!(
            env.typecheck_expr(&mut tmp_generator, &level_label, &expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Bool))
        );
    }

    #[test]
    fn test_typecheck_bool_expr_source() {
        let mut tmp_generator = TmpGenerator::default();
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
        let env = Env::new(EMPTY_SOURCEMAP.1);
        assert_eq!(
            env.typecheck_expr(&mut tmp_generator, &level_label, &expr),
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeMismatch(Rc::new(Type::Bool), Rc::new(Type::Int)),
                zspan!(),
            )])
        );
    }

    #[test]
    fn test_typecheck_lval_undefined_var() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let lval = zspan!(LVal::Simple("a".to_owned()));

        assert_eq!(
            env.typecheck_lval(&mut tmp_generator, &level_label, &lval).unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_lval_record_field() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record = hashmap! {
            "f".to_owned() => RecordField{ ty: Rc::new(Type::Int), index: 0}
        };
        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Record("r".to_owned(), record)),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("f".to_owned())
        ));
        assert_eq!(
            env.typecheck_lval(&mut tmp_generator, &level_label, &lval)
                .map(TranslateOutput::unwrap),
            Ok(LValProperties {
                ty: Rc::new(Type::Int),
                immutable: None,
            })
        );
    }

    #[test]
    fn test_typecheck_lval_record_field_err1() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Record("r".to_owned(), HashMap::new())),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.typecheck_lval(&mut tmp_generator, &level_label, &lval)
                .map(TranslateOutput::unwrap),
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedField("g".to_owned()),
                zspan!(),
            )])
        );
    }

    #[test]
    fn test_typecheck_lval_record_field_err2() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Int),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Field(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!("g".to_owned())
        ));
        assert_eq!(
            env.typecheck_lval(&mut tmp_generator, &level_label, &lval)
                .map(TranslateOutput::unwrap),
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::UndefinedField("g".to_owned()),
                zspan!()
            )])
        );
    }

    #[test]
    fn test_typecheck_array_subscript() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_var(
            "x".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Array(Rc::new(Type::Int), 3)),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );
        let lval = zspan!(LVal::Subscript(
            Box::new(zspan!(LVal::Simple("x".to_owned()))),
            zspan!(ExprType::Number(0))
        ));
        assert_eq!(
            env.typecheck_lval(&mut tmp_generator, &level_label, &lval)
                .map(TranslateOutput::unwrap),
            Ok(LValProperties {
                ty: Rc::new(Type::Int),
                immutable: Some("x".to_owned())
            })
        );
    }

    #[test]
    fn test_typecheck_let_type_annotation() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            immutable: zspan!(true),
            ty: Some(zspan!(ast::Type::Type(zspan!("int".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        assert_eq!(
            env.typecheck_let(&mut tmp_generator, &level_label, &let_expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Unit))
        );
        assert_eq!(env.vars["x"].ty, Rc::new(Type::Int));
        assert_eq!(env.vars["x"].immutable, true);
        assert!(env.var_def_spans.contains_key("x"));
    }

    #[test]
    fn test_typecheck_let_type_annotation_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let let_expr = zspan!(Let {
            pattern: zspan!(Pattern::String("x".to_owned())),
            immutable: zspan!(true),
            ty: Some(zspan!(ast::Type::Type(zspan!("string".to_owned())))),
            expr: zspan!(ExprType::Number(0))
        });
        assert_eq!(
            env.typecheck_let(&mut tmp_generator, &level_label, &let_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::String), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_typecheck_fn_call_undefined_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let fn_call_expr = zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![],
        });
        assert_eq!(
            env.typecheck_fn_call(&mut tmp_generator, &level_label, &fn_call_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::UndefinedFn("f".to_owned())
        );
    }

    #[test]
    fn test_typecheck_fn_call_not_fn_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Int),
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
            env.typecheck_fn_call(&mut tmp_generator, &level_label, &fn_call_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::NotAFn("f".to_owned())
        );
    }

    #[test]
    fn test_typecheck_fn_call_arity_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Int))),
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
            env.typecheck_fn_call(&mut tmp_generator, &level_label, &fn_call_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::ArityMismatch(0, 1)
        );
    }

    #[test]
    fn test_typecheck_fn_call_arg_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Fn(vec![Rc::new(Type::Int)], Rc::new(Type::Int))),
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
            env.typecheck_fn_call(&mut tmp_generator, &level_label, &fn_call)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        )
    }

    #[test]
    fn test_typecheck_fn_call_returns_aliased_type() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.insert_type("a".to_owned(), Type::Alias("int".to_owned()), zspan!());
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Alias("a".to_owned())))),
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
            env.typecheck_fn_call(&mut tmp_generator, &level_label, &fn_call)
                .unwrap()
                .unwrap(),
            Rc::new(Type::Alias("a".to_owned()))
        );
    }

    #[test]
    fn test_typecheck_typedef() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
        assert!(env.typecheck_let(&mut tmp_generator, &level_label, &let_expr).is_ok());
    }

    #[test]
    fn test_typecheck_typedef_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
        assert_eq!(
            env.typecheck_let(&mut tmp_generator, &level_label, &let_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Alias("a".to_owned())), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_typecheck_expr_typedef() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
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
        env.typecheck_let(&mut tmp_generator, &level_label, &var_def)
            .expect("typecheck var def");
        assert_eq!(
            env.typecheck_expr_mut(&mut tmp_generator, &level_label, &expr)
                .unwrap()
                .unwrap(),
            Rc::new(Type::Alias("i".to_owned()))
        );
    }

    #[test]
    fn test_recursive_typedef() {
        let mut tmp_generator = TmpGenerator::default();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let type_decl = zspan!(DeclType::Type(zspan!(TypeDecl {
            id: zspan!("i".to_owned()),
            ty: zspan!(ast::TypeDeclType::Record(vec![zspan!(TypeField {
                id: zspan!("i".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("i".to_owned())))
            })]))
        })));
        env.typecheck_decl_first_pass(&mut tmp_generator, &type_decl)
            .expect("typecheck decl");
    }

    #[test]
    fn test_check_for_type_decl_cycles_err1() {
        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.insert_type("a".to_owned(), Type::Alias("a".to_owned()), zspan!());
        assert_eq!(
            env.check_for_type_decl_cycles("a", vec![]).unwrap_err()[0].t.ty,
            TypecheckErrType::TypeDeclCycle("a".to_owned(), Rc::new(Type::Alias("a".to_owned())))
        );
    }

    #[test]
    fn test_check_for_type_decl_cycles_err2() {
        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.insert_type("a".to_owned(), Type::Alias("b".to_owned()), zspan!());
        env.insert_type("b".to_owned(), Type::Alias("c".to_owned()), zspan!());
        assert_eq!(env.check_for_type_decl_cycles("a", vec![]), Ok(()));
    }

    #[test]
    fn test_duplicate_type_decl() {
        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.typecheck_type_decl(&zspan!(ast::TypeDecl {
            id: zspan!("a".to_owned()),
            ty: zspan!(ast::TypeDeclType::Unit)
        }))
        .expect("typecheck type decl");

        assert_eq!(
            env.typecheck_type_decl(&zspan!(ast::TypeDecl {
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
        let mut tmp_generator = TmpGenerator::default();
        let label_f = tmp_generator.new_label();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Fn(vec![], Rc::new(Type::Alias("a".to_owned())))),
                immutable: true,
                entry_type: EnvEntryType::Fn(label_f),
            },
            zspan!(),
        );
        assert_eq!(
            env.check_for_invalid_types().unwrap_err()[0].t.ty,
            TypecheckErrType::UndefinedType("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_first_pass() {
        let mut tmp_generator = TmpGenerator::default();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let result = env.first_pass(
            &mut tmp_generator,
            &[
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
            ],
        );
        assert_eq!(
            result,
            Err(vec![TypecheckErr::new_err(
                TypecheckErrType::TypeDeclCycle("a".to_owned(), Rc::new(Type::Alias("a".to_owned()))),
                zspan!()
            )])
        );
    }

    #[test]
    fn test_typecheck_fn_decl_duplicate() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
        env.typecheck_fn_decl_sig(&mut tmp_generator, &level_label, &fn_decl1)
            .expect("typecheck function signature");

        assert_eq!(
            env.typecheck_fn_decl_sig(&mut tmp_generator, &level_label, &fn_decl2)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::DuplicateFn("f".to_owned())
        );
    }

    #[test]
    fn test_typecheck_fn_decl_duplicate_param() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_fn_decl_sig(&mut tmp_generator, &level_label, &fn_decl)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::DuplicateParam("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_fn_decl() {
        let mut tmp_generator = TmpGenerator::default();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_fn_decl_sig(&mut tmp_generator, &Label::top(), &fn_decl),
            Ok(Rc::new(Type::Fn(vec![Rc::new(Type::Int)], Rc::new(Type::Unit))))
        );
        env.typecheck_fn_decl_body(&mut tmp_generator, &fn_decl)
            .expect("typecheck fn decl body");

        let label = env.vars["f"].entry_type.fn_label();
        let levels = env.levels.borrow();
        let level = &levels[&label];

        assert_eq!(level.parent_label, Some(Label::top()));
        assert_eq!(
            level.formals(),
            vec![Access {
                level_label: label,
                access: frame::Access::InFrame(-8)
            }]
        );
    }

    #[test]
    fn test_validate_type() {
        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let mut record_fields = HashMap::new();
        record_fields.insert(
            "f".to_owned(),
            RecordField {
                ty: Rc::new(Type::Alias("a".to_owned())),
                index: 0,
            },
        );
        record_fields.insert(
            "g".to_owned(),
            RecordField {
                ty: Rc::new(Type::Alias("b".to_owned())),
                index: 1,
            },
        );
        env.insert_type("a".to_owned(), Type::Record("a".to_owned(), record_fields), zspan!());

        let errs = env.check_for_invalid_types().unwrap_err();
        assert_eq!(errs.len(), 1);
        // Recursive type def in records is allowed
        assert_eq!(errs[0].t.ty, TypecheckErrType::UndefinedType("b".to_owned()));
    }

    #[test]
    fn test_typecheck_record_missing_fields() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => RecordField { ty: Rc::new(Type::Int), index: 0 },
            },
        );
        env.insert_type("r".to_owned(), record_type, zspan!());
        let record = zspan!(Record {
            id: zspan!("r".to_owned()),
            field_assigns: vec![]
        });
        assert_eq!(
            env.typecheck_record(&mut tmp_generator, &level_label, &record)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::MissingFields(vec!["a".to_owned()])
        );
    }

    #[test]
    fn test_typecheck_record_invalid_fields() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_record(&mut tmp_generator, &level_label, &record)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::InvalidFields(vec!["b".to_owned()])
        );
    }

    #[test]
    fn test_typecheck_record_duplicate_field() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => RecordField { ty: Rc::new(Type::Int), index: 0 },
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
            env.typecheck_record(&mut tmp_generator, &level_label, &record)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::DuplicateField("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_record() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => RecordField { ty: Rc::new(Type::Int), index: 0 },
                "b".to_owned() => RecordField { ty: Rc::new(Type::String), index: 1},
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

        assert_eq!(
            env.typecheck_record(&mut tmp_generator, &level_label, &record)
                .unwrap()
                .unwrap(),
            Rc::new(record_type)
        );
    }

    #[test]
    fn test_typecheck_record_field_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => RecordField { ty: Rc::new(Type::Int), index: 0 },
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
            env.typecheck_record(&mut tmp_generator, &level_label, &record)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_typecheck_fn_independent() {
        let mut tmp_generator = TmpGenerator::default();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_decls(&mut tmp_generator, EMPTY_SOURCEMAP.1, &[fn1, fn2])
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_fn_body() {
        let mut tmp_generator = TmpGenerator::default();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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

        assert_eq!(
            env.typecheck_decls(&mut tmp_generator, EMPTY_SOURCEMAP.1, &[fn_decl]),
            Ok(())
        );
    }

    #[test]
    fn test_typecheck_seq_independent() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
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

        env.typecheck_expr(&mut tmp_generator, &level_label, &seq_expr1)
            .expect("typecheck expr");
        assert_eq!(
            env.typecheck_expr(&mut tmp_generator, &level_label, &seq_expr2)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::UndefinedVar("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_seq_captures_value() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_var(
            "b".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::String),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );
        let seq = zspan!(ExprType::Seq(
            vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("b".to_owned())))))],
            true
        ));

        assert_eq!(
            env.typecheck_expr(&mut tmp_generator, &level_label, &seq)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::String))
        );
    }

    #[test]
    fn test_illegal_let_expr() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_expr(&mut tmp_generator, &level_label, &expr).unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::IllegalLetExpr
        );
    }

    #[test]
    fn test_assign_immut_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        env.insert_var(
            "a".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::Int),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(
                &mut tmp_generator,
                &level_label,
                &zspan!(Assign {
                    lval: zspan!(LVal::Simple("a".to_owned())),
                    expr: zspan!(ExprType::Number(0))
                })
            )
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::MutatingImmutable("a".to_owned())
        );
    }

    #[test]
    fn test_assign_record_field_immut_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => RecordField { ty: Rc::new(Type::Int), index: 0 },
            },
        );
        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        env.insert_var(
            "r".to_owned(),
            EnvEntry {
                ty: Rc::new(record_type),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(
                &mut tmp_generator,
                &level_label,
                &zspan!(Assign {
                    lval: zspan!(LVal::Field(
                        Box::new(zspan!(LVal::Simple("r".to_owned()))),
                        zspan!("a".to_owned())
                    )),
                    expr: zspan!(ExprType::Number(0))
                })
            )
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::MutatingImmutable("r".to_owned())
        );
    }

    #[test]
    fn test_assign_record_field_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => RecordField { ty: Rc::new(Type::Int), index: 0 },
            },
        );
        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        env.insert_var(
            "r".to_owned(),
            EnvEntry {
                ty: Rc::new(record_type),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(
                &mut tmp_generator,
                &level_label,
                &zspan!(Assign {
                    lval: zspan!(LVal::Field(
                        Box::new(zspan!(LVal::Simple("r".to_owned()))),
                        zspan!("a".to_owned())
                    )),
                    expr: zspan!(ExprType::Unit)
                })
            )
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_assign_record_field_type() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let record_type = Type::Record(
            "r".to_owned(),
            hashmap! {
                "a".to_owned() => RecordField{ ty: Rc::new(Type::Int), index: 0 },
            },
        );
        env.insert_type("r".to_owned(), record_type.clone(), zspan!());
        env.record_field_decl_spans
            .insert("r".to_owned(), hashmap! { "a".to_owned() => zspan!() });

        env.insert_var(
            "r".to_owned(),
            EnvEntry {
                ty: Rc::new(record_type),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(
                &mut tmp_generator,
                &level_label,
                &zspan!(Assign {
                    lval: zspan!(LVal::Field(
                        Box::new(zspan!(LVal::Simple("r".to_owned()))),
                        zspan!("a".to_owned())
                    )),
                    expr: zspan!(ExprType::Number(0))
                })
            )
            .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_assign_array_immut_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let array_type = Type::Array(Rc::new(Type::Int), 1);
        env.insert_var(
            "a".to_owned(),
            EnvEntry {
                ty: Rc::new(array_type),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(
                &mut tmp_generator,
                &level_label,
                &zspan!(Assign {
                    lval: zspan!(LVal::Subscript(
                        Box::new(zspan!(LVal::Simple("a".to_owned()))),
                        zspan!(ExprType::Number(0))
                    )),
                    expr: zspan!(ExprType::Number(0))
                })
            )
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::MutatingImmutable("a".to_owned())
        );
    }

    #[test]
    fn test_assign_array_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

        let array_type = Type::Array(Rc::new(Type::Int), 1);
        env.insert_var(
            "a".to_owned(),
            EnvEntry {
                ty: Rc::new(array_type),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(
                &mut tmp_generator,
                &level_label,
                &zspan!(Assign {
                    lval: zspan!(LVal::Subscript(
                        Box::new(zspan!(LVal::Simple("a".to_owned()))),
                        zspan!(ExprType::Number(0))
                    )),
                    expr: zspan!(ExprType::String("s".to_owned()))
                })
            )
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_assign_array() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }
        let array_type = Type::Array(Rc::new(Type::Int), 1);
        env.insert_var(
            "a".to_owned(),
            EnvEntry {
                ty: Rc::new(array_type),
                immutable: false,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
                }),
            },
            zspan!(),
        );

        assert_eq!(
            env.typecheck_assign(
                &mut tmp_generator,
                &level_label,
                &zspan!(Assign {
                    lval: zspan!(LVal::Subscript(
                        Box::new(zspan!(LVal::Simple("a".to_owned()))),
                        zspan!(ExprType::Number(0))
                    )),
                    expr: zspan!(ExprType::Number(0))
                })
            )
            .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_translate_assign_to_fn() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        let fn_type = Type::Fn(vec![], Rc::new(Type::Unit));
        env.insert_var(
            "f".to_owned(),
            EnvEntry {
                ty: Rc::new(fn_type),
                immutable: true,
                entry_type: EnvEntryType::Fn(Label("f".to_owned())),
            },
            zspan!(),
        );

        assert!(dbg!(env.typecheck_assign(
            &mut tmp_generator,
            &level_label,
            &zspan!(Assign {
                lval: zspan!(LVal::Simple("f".to_owned())),
                expr: zspan!(ExprType::Number(0))
            })
        ))
        .is_err());
    }

    #[test]
    fn test_typecheck_array() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Number(3))
        });

        assert_eq!(
            env.typecheck_array(&mut tmp_generator, &level_label, &array_expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Array(Rc::new(Type::Int), 3)))
        );
    }

    #[test]
    fn test_typecheck_array_const_expr_len() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Arith(Box::new(zspan!(Arith {
                l: zspan!(ExprType::Number(1)),
                op: zspan!(ArithOp::Add),
                r: zspan!(ExprType::Number(2)),
            }))))
        });

        assert_eq!(
            env.typecheck_array(&mut tmp_generator, &level_label, &array_expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Array(Rc::new(Type::Int), 3)))
        );
    }

    #[test]
    fn test_typecheck_array_negative_len_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(ExprType::Neg(Box::new(zspan!(ExprType::Number(3)))))
        });

        assert_eq!(
            env.typecheck_array(&mut tmp_generator, &level_label, &array_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::NegativeArrayLen(-3)
        );
    }

    #[test]
    fn test_typecheck_array_non_constant_expr_err() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let fn_call = ExprType::FnCall(Box::new(zspan!(FnCall {
            id: zspan!("f".to_owned()),
            args: vec![]
        })));
        let array_expr = zspan!(Array {
            initial_value: zspan!(ExprType::Number(0)),
            len: zspan!(fn_call.clone())
        });

        assert_eq!(
            env.typecheck_array(&mut tmp_generator, &level_label, &array_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::NonConstantArithExpr(fn_call)
        );
    }

    #[test]
    fn test_typecheck_if_then() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Unit),
            else_expr: None,
        });

        assert_eq!(
            env.typecheck_if(&mut tmp_generator, &level_label, &if_expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_typecheck_if_then_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: None,
        });

        assert_eq!(
            env.typecheck_if(&mut tmp_generator, &level_label, &if_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_typecheck_if_then_else() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: Some(zspan!(ExprType::Number(1))),
        });

        assert_eq!(
            env.typecheck_if(&mut tmp_generator, &level_label, &if_expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_typecheck_if_then_else_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let if_expr = zspan!(If {
            cond: zspan!(ExprType::BoolLiteral(true)),
            then_expr: zspan!(ExprType::Number(0)),
            else_expr: Some(zspan!(ExprType::String("s".to_owned()))),
        });

        assert_eq!(
            env.typecheck_if(&mut tmp_generator, &level_label, &if_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_typecheck_range() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let range = zspan!(Range {
            start: zspan!(ExprType::Number(0)),
            end: zspan!(ExprType::Number(1)),
        });

        assert_eq!(
            env.typecheck_range(&mut tmp_generator, &level_label, &range)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Iterator(Rc::new(Type::Int))))
        );
    }

    #[test]
    fn test_typecheck_range_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let range = zspan!(Range {
            start: zspan!(ExprType::Number(0)),
            end: zspan!(ExprType::String("a".to_owned())),
        });

        assert_eq!(
            env.typecheck_range(&mut tmp_generator, &level_label, &range)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    fn test_typecheck_for() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

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

        assert_eq!(
            env.typecheck_for(&mut tmp_generator, &level_label, &for_expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_typecheck_for_range_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let for_expr = zspan!(For {
            index: zspan!("i".to_owned()),
            range: zspan!(ExprType::Number(0)),
            body: zspan!(ExprType::Seq(
                vec![zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("i".to_owned())))))],
                false
            ))
        });

        assert_eq!(
            env.typecheck_for(&mut tmp_generator, &level_label, &for_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Iterator(Rc::new(Type::Int))), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_typecheck_for_body_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }

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
            env.typecheck_for(&mut tmp_generator, &level_label, &for_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_typecheck_while() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::BoolLiteral(true)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Unit)], false))
        });

        assert_eq!(
            env.typecheck_while(&mut tmp_generator, &level_label, &while_expr)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_typecheck_while_cond_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::Number(0)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Unit)], false))
        });
        assert_eq!(
            env.typecheck_while(&mut tmp_generator, &level_label, &while_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Bool), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_typecheck_while_body_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let while_expr = zspan!(While {
            cond: zspan!(ExprType::BoolLiteral(true)),
            body: zspan!(ExprType::Seq(vec![zspan!(ExprType::Number(0))], true))
        });

        assert_eq!(
            env.typecheck_while(&mut tmp_generator, &level_label, &while_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Unit), Rc::new(Type::Int))
        );
    }

    #[test]
    fn test_typecheck_compare() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let compare = zspan!(Compare {
            l: zspan!(ExprType::BoolLiteral(true)),
            op: zspan!(CompareOp::Eq),
            r: zspan!(ExprType::BoolLiteral(true)),
        });

        assert_eq!(
            env.typecheck_compare(&mut tmp_generator, &level_label, &compare)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Bool))
        );
    }

    #[test]
    fn test_typecheck_compare_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let compare = zspan!(Compare {
            l: zspan!(ExprType::Number(0)),
            op: zspan!(CompareOp::Eq),
            r: zspan!(ExprType::BoolLiteral(true)),
        });

        assert_eq!(
            env.typecheck_compare(&mut tmp_generator, &level_label, &compare)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::Bool))
        );
    }

    #[test]
    fn test_typecheck_enum_arity_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_enum(&mut tmp_generator, &level_label, &enum_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::ArityMismatch(1, 0)
        );
    }

    #[test]
    fn test_typecheck_enum_type_mismatch() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_enum(&mut tmp_generator, &level_label, &enum_expr)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::TypeMismatch(Rc::new(Type::Int), Rc::new(Type::String))
        );
    }

    #[test]
    #[should_panic] // FIXME
    fn test_typecheck_closure() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        let closure = zspan!(Closure {
            type_fields: vec![zspan!(TypeField {
                id: zspan!("a".to_owned()),
                ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
            })],
            body: zspan!(ExprType::LVal(Box::new(zspan!(LVal::Simple("a".to_owned())))))
        });

        assert_eq!(
            env.typecheck_closure(&mut tmp_generator, &level_label, &closure)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Fn(vec![Rc::new(Type::Int)], Rc::new(Type::Int))))
        );
    }

    #[test]
    #[should_panic] // FIXME
    fn test_typecheck_closure_captures_value() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
        }
        env.insert_var(
            "b".to_owned(),
            EnvEntry {
                ty: Rc::new(Type::String),
                immutable: true,
                entry_type: EnvEntryType::Var(Access {
                    level_label: level_label.clone(),
                    access: frame::Access::InFrame(-8),
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
            env.typecheck_closure(&mut tmp_generator, &level_label, &closure)
                .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Fn(vec![Rc::new(Type::Int)], Rc::new(Type::String))))
        );
    }

    #[test]
    fn test_typecheck_closure_duplicate_param() {
        let mut tmp_generator = TmpGenerator::default();
        let level_label = Label::top();

        let env = Env::new(EMPTY_SOURCEMAP.1);
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
            env.typecheck_closure(&mut tmp_generator, &level_label, &closure)
                .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::DuplicateParam("a".to_owned())
        );
    }

    #[test]
    fn test_typecheck_fn_decl_exprs_in_seq_recursive() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
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
            env.typecheck_expr(
                &mut tmp_generator,
                &level_label,
                &zspan!(ExprType::Seq(vec![fn_decl1, let_expr, fn_decl2], false))
            )
            .map(TranslateOutput::unwrap),
            Ok(Rc::new(Type::Unit))
        );
    }

    #[test]
    fn test_typecheck_fn_decl_exprs_in_seq_captures_correctly() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(EMPTY_SOURCEMAP.1);
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label.clone(), level);
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
            env.typecheck_expr(
                &mut tmp_generator,
                &level_label,
                &zspan!(ExprType::Seq(vec![fn_decl1, let_expr], false))
            )
            .unwrap_err()[0]
                .t
                .ty,
            TypecheckErrType::UndefinedVar("h".to_owned())
        );
    }

    #[test]
    fn test_add_static_link() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);

        assert_eq!(level.frame.formals.len(), 1);
        assert_eq!(level.frame.formals[0], frame::Access::InFrame(0));
    }

    #[test]
    fn test_follows_static_links() {
        {
            let mut tmp_generator = TmpGenerator::default();
            let frame = Frame::new(&mut tmp_generator, "f", &[true]);
            let label = frame.label.clone();
            let mut levels = hashmap! {
                Label::top() => Level::top(),
                label.clone() => Level {
                    parent_label: Some(Label::top()),
                    frame,
                },
            };
            let level = levels.get_mut(&label).unwrap();
            let local = level.alloc_local(&mut tmp_generator, true);

            assert_eq!(
                Env::translate_simple_var(&levels, &local, &label),
                translate::Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                    Box::new(ir::Expr::Tmp(*tmp::FP)),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                ))))
            );
        }

        {
            let mut tmp_generator = TmpGenerator::default();
            let frame_f = Frame::new(&mut tmp_generator, "f", &[true]);
            let label_f = frame_f.label.clone();
            let frame_g = Frame::new(&mut tmp_generator, "g", &[true]);
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
            let local = level.alloc_local(&mut tmp_generator, true);

            assert_eq!(
                Env::translate_simple_var(&levels, &local, &label_g),
                translate::Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                    Box::new(ir::Expr::Mem(Box::new(ir::Expr::Tmp(*tmp::FP)))),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                ))))
            );
        }
    }
}
