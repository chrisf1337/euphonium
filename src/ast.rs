use codespan::ByteSpan;
use std::ops::{Deref, DerefMut};

pub type Decl = Spanned<DeclType>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclType {
    Type(Spanned<TypeDecl>),
    Fn(Spanned<FnDecl>),
    Error,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum _DeclType {
    Type(_TypeDecl),
    Fn(_FnDecl),
    Error,
}

impl From<DeclType> for _DeclType {
    fn from(decl: DeclType) -> Self {
        match decl {
            DeclType::Type(type_decl) => _DeclType::Type(type_decl.t.into()),
            DeclType::Fn(fn_decl) => _DeclType::Fn(fn_decl.t.into()),
            DeclType::Error => _DeclType::Error,
        }
    }
}

/// type A = ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDecl {
    pub id: Spanned<String>,
    pub ty: Spanned<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypeDecl {
    pub type_id: String,
    pub ty: _Type,
}

impl From<TypeDecl> for _TypeDecl {
    fn from(decl: TypeDecl) -> Self {
        Self {
            type_id: decl.id.t,
            ty: decl.ty.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Type(Spanned<String>),
    Enum(Vec<Spanned<EnumCase>>),
    /// { a: int, b: int }
    Record(Vec<Spanned<TypeField>>),
    Array(Box<Spanned<Type>>, Spanned<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum _Type {
    Type(String),
    Enum(Vec<_EnumCase>),
    /// { a: int, b: int }
    Record(Vec<_TypeField>),
    Array(Box<_Type>, usize),
}

impl From<Type> for _Type {
    fn from(ty: Type) -> Self {
        match ty {
            Type::Type(ty) => _Type::Type(ty.t),
            Type::Enum(cases) => _Type::Enum(cases.into_iter().map(|c| c.t.into()).collect()),
            Type::Record(type_fields) => {
                _Type::Record(type_fields.into_iter().map(|tf| tf.t.into()).collect())
            }
            Type::Array(ty, len) => _Type::Array(Box::new(ty.t.into()), len.t),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumCase {
    pub id: Spanned<String>,
    pub params: Vec<Spanned<EnumParam>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _EnumCase {
    pub id: String,
    pub params: Vec<_EnumParam>,
}

impl From<EnumCase> for _EnumCase {
    fn from(e: EnumCase) -> _EnumCase {
        Self {
            id: e.id.t,
            params: e
                .params
                .into_iter()
                .map(|p| _EnumParam::from(p.t))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnumParam {
    Simple(Spanned<String>),
    Record(Vec<Spanned<TypeField>>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum _EnumParam {
    Simple(String),
    Record(Vec<_TypeField>),
}

impl From<EnumParam> for _EnumParam {
    fn from(ep: EnumParam) -> Self {
        match ep {
            EnumParam::Simple(s) => _EnumParam::Simple(s.t),
            EnumParam::Record(type_fields) => {
                _EnumParam::Record(type_fields.into_iter().map(|tf| tf.t.into()).collect())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeField {
    pub id: Spanned<String>,
    pub ty: Spanned<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypeField {
    pub id: String,
    pub ty: _Type,
}

impl From<TypeField> for _TypeField {
    fn from(field: TypeField) -> Self {
        Self {
            id: field.id.t,
            ty: field.ty.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pattern {
    Wildcard,
    String(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Let {
    pub pattern: Spanned<Pattern>,
    pub mutable: Spanned<bool>,
    pub ty: Option<Spanned<Type>>,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _Let {
    pub pattern: Pattern,
    pub mutable: bool,
    pub ty: Option<_Type>,
    pub expr: _ExprType,
}

impl From<Let> for _Let {
    fn from(expr: Let) -> Self {
        Self {
            pattern: expr.pattern.t,
            mutable: expr.mutable.t,
            ty: expr.ty.map(|ty| ty.t.into()),
            expr: expr.expr.t.into(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub t: T,
    pub span: ByteSpan,
}

impl<T> Spanned<T> {
    pub fn new(t: T, span: ByteSpan) -> Spanned<T> {
        Spanned { t, span }
    }

    pub fn map<F, U>(self, f: F) -> Spanned<U>
    where
        F: FnOnce(T) -> U,
    {
        Spanned::new(f(self.t), self.span)
    }

    pub fn unwrap<U>(self) -> U
    where
        U: From<T>,
    {
        self.t.into()
    }
}

impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.t
    }
}

impl<T> DerefMut for Spanned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.t
    }
}

pub type Expr = Spanned<ExprType>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprType {
    /// true if returns last value, false if unit-typed
    Seq(Vec<Expr>, bool),
    String(String),
    Number(usize),
    Neg(Box<Expr>),
    Arith(Box<Expr>, Spanned<ArithOp>, Box<Expr>),
    Unit,
    BoolLiteral(bool),
    Not(Box<Expr>),
    Bool(Box<Expr>, Spanned<BoolOp>, Box<Expr>),
    Continue,
    Break,
    LVal(Box<Spanned<LVal>>),
    Let(Box<Spanned<Let>>),
    FnCall(Box<Spanned<FnCall>>),
    Record(Box<Spanned<Record>>),
    Assign(Box<Spanned<Assign>>),
    Array(Box<Spanned<Array>>),
    If(Box<Spanned<If>>),
    Range(Box<Expr>, Box<Expr>),
    For(Box<Spanned<For>>),
    While(Box<Spanned<While>>),
    Compare(Box<Expr>, Spanned<CompareOp>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum _ExprType {
    Seq(Vec<_ExprType>, bool),
    String(String),
    Number(usize),
    Neg(Box<_ExprType>),
    Arith(Box<_ExprType>, ArithOp, Box<_ExprType>),
    Unit,
    BoolLiteral(bool),
    Not(Box<_ExprType>),
    Bool(Box<_ExprType>, BoolOp, Box<_ExprType>),
    Continue,
    Break,
    LVal(Box<_LVal>),
    Let(Box<_Let>),
    FnCall(Box<_FnCall>),
    Record(Box<_Record>),
    Assign(Box<_Assign>),
    Array(Box<_Array>),
    If(Box<_If>),
    Range(Box<_ExprType>, Box<_ExprType>),
    For(Box<_For>),
    While(Box<_While>),
    Compare(Box<_ExprType>, CompareOp, Box<_ExprType>),
}

impl From<ExprType> for _ExprType {
    fn from(expr: ExprType) -> Self {
        match expr {
            ExprType::Seq(exprs, returns) => _ExprType::Seq(
                exprs.into_iter().map(|expr| expr.t.into()).collect(),
                returns,
            ),
            ExprType::String(s) => _ExprType::String(s),
            ExprType::Number(n) => _ExprType::Number(n),
            ExprType::Neg(expr) => _ExprType::Neg(Box::new(expr.t.into())),
            ExprType::Arith(l, op, r) => {
                _ExprType::Arith(Box::new(l.t.into()), op.t, Box::new(r.t.into()))
            }
            ExprType::Unit => _ExprType::Unit,
            ExprType::BoolLiteral(b) => _ExprType::BoolLiteral(b),
            ExprType::Not(expr) => _ExprType::Not(Box::new(expr.t.into())),
            ExprType::Bool(l, op, r) => {
                _ExprType::Bool(Box::new(l.t.into()), op.t, Box::new(r.t.into()))
            }
            ExprType::Continue => _ExprType::Continue,
            ExprType::Break => _ExprType::Break,
            ExprType::LVal(expr) => _ExprType::LVal(Box::new(expr.t.into())),
            ExprType::Let(expr) => _ExprType::Let(Box::new(expr.t.into())),
            ExprType::FnCall(expr) => _ExprType::FnCall(Box::new(expr.t.into())),
            ExprType::Record(expr) => _ExprType::Record(Box::new(expr.t.into())),
            ExprType::Assign(expr) => _ExprType::Assign(Box::new(expr.t.into())),
            ExprType::Array(expr) => _ExprType::Array(Box::new(expr.t.into())),
            ExprType::If(expr) => _ExprType::If(Box::new(expr.t.into())),
            ExprType::Range(from, to) => {
                _ExprType::Range(Box::new(from.t.into()), Box::new(to.t.into()))
            }
            ExprType::For(expr) => _ExprType::For(Box::new(expr.t.into())),
            ExprType::While(expr) => _ExprType::While(Box::new(expr.t.into())),
            ExprType::Compare(l, op, r) => {
                _ExprType::Compare(Box::new(l.t.into()), op.t, Box::new(r.t.into()))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoolOp {
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareOp {
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnDecl {
    pub id: Spanned<String>,
    pub type_fields: Vec<Spanned<TypeField>>,
    pub return_type: Option<Spanned<Type>>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _FnDecl {
    pub id: String,
    pub type_fields: Vec<_TypeField>,
    pub return_type: Option<_Type>,
    pub body: _ExprType,
}

impl From<FnDecl> for _FnDecl {
    fn from(decl: FnDecl) -> Self {
        Self {
            id: decl.id.t,
            type_fields: decl.type_fields.into_iter().map(|tf| tf.t.into()).collect(),
            return_type: decl.return_type.map(|t| t.t.into()),
            body: decl.body.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LVal {
    Simple(String),
    Field(Box<Spanned<LVal>>, Spanned<String>),
    Subscript(Box<Spanned<LVal>>, Expr),
}

impl From<Spanned<LVal>> for Expr {
    fn from(lval: Spanned<LVal>) -> Expr {
        let span = lval.span;
        Expr {
            t: ExprType::LVal(Box::new(lval)),
            span,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum _LVal {
    Simple(String),
    Field(Box<_LVal>, String),
    Subscript(Box<_LVal>, _ExprType),
}

impl From<LVal> for _LVal {
    fn from(expr: LVal) -> Self {
        match expr {
            LVal::Simple(s) => _LVal::Simple(s),
            LVal::Field(l, s) => _LVal::Field(Box::new(l.t.into()), s.t),
            LVal::Subscript(l, s) => _LVal::Subscript(Box::new(l.t.into()), s.t.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnCall {
    pub id: Spanned<String>,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _FnCall {
    pub id: String,
    pub args: Vec<_ExprType>,
}

impl From<FnCall> for _FnCall {
    fn from(call: FnCall) -> Self {
        Self {
            id: call.id.t,
            args: call.args.into_iter().map(|arg| arg.t.into()).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub id: Spanned<String>,
    pub field_assigns: Vec<FieldAssign>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _Record {
    pub id: String,
    pub field_assigns: Vec<_FieldAssign>,
}

impl From<Record> for _Record {
    fn from(record: Record) -> Self {
        Self {
            id: record.id.t,
            field_assigns: record
                .field_assigns
                .into_iter()
                .map(_FieldAssign::from)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAssign {
    pub id: Spanned<String>,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _FieldAssign {
    pub id: String,
    pub expr: _ExprType,
}

impl From<FieldAssign> for _FieldAssign {
    fn from(assign: FieldAssign) -> Self {
        Self {
            id: assign.id.t,
            expr: assign.expr.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assign {
    pub lval: Spanned<LVal>,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _Assign {
    pub lval: _LVal,
    pub expr: _ExprType,
}

impl From<Assign> for _Assign {
    fn from(a: Assign) -> Self {
        Self {
            lval: a.lval.t.into(),
            expr: a.expr.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array {
    pub initial_value: Expr,
    pub len: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _Array {
    pub initial_value: _ExprType,
    pub len: _ExprType,
}

impl From<Array> for _Array {
    fn from(expr: Array) -> Self {
        Self {
            initial_value: expr.initial_value.t.into(),
            len: expr.len.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct If {
    pub cond: Expr,
    pub then_expr: Expr,
    pub else_expr: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _If {
    pub cond: _ExprType,
    pub then_expr: _ExprType,
    pub else_expr: Option<_ExprType>,
}

impl From<If> for _If {
    fn from(i: If) -> Self {
        Self {
            cond: i.cond.t.into(),
            then_expr: i.then_expr.t.into(),
            else_expr: i.else_expr.map(|e| e.t.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct For {
    pub index: Spanned<String>,
    pub range: Expr,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _For {
    pub index: String,
    pub range: _ExprType,
    pub body: _ExprType,
}

impl From<For> for _For {
    fn from(expr: For) -> Self {
        Self {
            index: expr.index.t,
            range: expr.range.t.into(),
            body: expr.body.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct While {
    pub cond: Expr,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _While {
    pub cond: _ExprType,
    pub body: _ExprType,
}

impl From<While> for _While {
    fn from(expr: While) -> Self {
        Self {
            cond: expr.cond.t.into(),
            body: expr.body.t.into(),
        }
    }
}
