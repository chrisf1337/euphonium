pub type Decl = Spanned<DeclType>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclType {
    Type(TypeDecl),
    Fn(FnDecl),
    Error,
}

/// type A = ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDecl {
    pub type_id: Spanned<String>,
    pub ty: Spanned<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypeDecl {
    pub type_id: String,
    pub ty: Type,
}

impl From<TypeDecl> for _TypeDecl {
    fn from(decl: TypeDecl) -> Self {
        Self {
            type_id: decl.type_id.t,
            ty: decl.ty.t,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Type(String),
    /// { a: int, b: int }
    Record(Vec<TypeField>),
    Array(Box<Spanned<Type>>, Spanned<usize>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeField {
    pub id: Spanned<String>,
    pub ty: Spanned<Type>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypeField {
    pub id: String,
    pub ty: Type,
}

impl From<TypeField> for _TypeField {
    fn from(field: TypeField) -> Self {
        Self {
            id: field.id.t,
            ty: field.ty.t,
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
    pub ty: Option<Type>,
    pub expr: _ExprType,
}

impl From<Let> for _Let {
    fn from(expr: Let) -> Self {
        Self {
            pattern: expr.pattern.t,
            mutable: expr.mutable.t,
            ty: expr.ty.map(|ty| ty.t),
            expr: expr.expr.t.into(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Span {
    pub l: usize,
    pub r: usize,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub t: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(t: T, span: (usize, usize)) -> Spanned<T> {
        Spanned {
            t,
            span: Span {
                l: span.0,
                r: span.1,
            },
        }
    }

    pub fn map<F, U>(self, f: F) -> Spanned<U> where F: FnOnce(T) -> U {
        Spanned::new(f(self.t), (self.span.l, self.span.r))
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
    Arith(Box<Expr>, ArithOp, Box<Expr>),
    Unit,
    BoolLiteral(bool),
    Not(Box<Expr>),
    Bool(Box<Expr>, BoolOp, Box<Expr>),
    Continue,
    Break,
    LVal(Box<LVal>),
    Let(Box<Let>),
    FnCall(Box<FnCall>),
    Record(Box<Record>),
    Assign(Box<Assign>),
    Array(Box<Array>),
    If(Box<If>),
    Range(Box<Expr>, Box<Expr>),
    For(Box<For>),
    While(Box<While>),
    Compare(Box<Expr>, CompareOp, Box<Expr>),
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
    Record(Box<Record>),
    Assign(Box<Assign>),
    Array(Box<_Array>),
    If(Box<If>),
    Range(Box<_ExprType>, Box<_ExprType>),
    For(Box<_For>),
    While(Box<_While>),
    Compare(Box<_ExprType>, CompareOp, Box<_ExprType>),
}

impl From<ExprType> for _ExprType {
    fn from(expr: ExprType) -> Self {
        match expr {
            ExprType::Seq(exprs, returns) => _ExprType::Seq(exprs.into_iter().map(|expr| expr.t.into()).collect(), returns),
            ExprType::String(s) => _ExprType::String(s),
            ExprType::Number(n) => _ExprType::Number(n),
            ExprType::Neg(expr) => _ExprType::Neg(Box::new(expr.t.into())),
            ExprType::Arith(l, op, r) => _ExprType::Arith(Box::new(l.t.into()), op, Box::new(r.t.into())),
            ExprType::Unit => _ExprType::Unit,
            ExprType::BoolLiteral(b) => _ExprType::BoolLiteral(b),
            ExprType::Not(expr) => _ExprType::Not(Box::new(expr.t.into())),
            ExprType::Bool(l, op, r) => _ExprType::Bool(Box::new(l.t.into()), op, Box::new(r.t.into())),
            ExprType::Continue => _ExprType::Continue,
            ExprType::Break => _ExprType::Break,
            ExprType::LVal(expr) => _ExprType::LVal(Box::new((*expr).into())),
            ExprType::Let(expr) => _ExprType::Let(Box::new((*expr).into())),
            ExprType::FnCall(expr) => _ExprType::FnCall(Box::new((*expr).into())),
            ExprType::Record(expr) => _ExprType::Record(Box::new((*expr).into())),
            ExprType::Assign(expr) => _ExprType::Assign(Box::new((*expr).into())),
            ExprType::Array(expr) => _ExprType::Array(Box::new((*expr).into())),
            ExprType::If(expr) => _ExprType::If(Box::new((*expr).into())),
            ExprType::Range(from, to) => _ExprType::Range(Box::new(from.t.into()), Box::new(to.t.into())),
            ExprType::For(expr) => _ExprType::For(Box::new((*expr).into())),
            ExprType::While(expr) => _ExprType::While(Box::new((*expr).into())),
            ExprType::Compare(l, op, r) => _ExprType::Compare(Box::new(l.t.into()), op, Box::new(r.t.into())),
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
    pub type_fields: Vec<TypeField>,
    pub return_type: Option<Spanned<Type>>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _FnDecl {
    pub id: String,
    pub type_fields: Vec<_TypeField>,
    pub return_type: Option<Type>,
    pub body: _ExprType,
}

impl From<FnDecl> for _FnDecl {
    fn from(decl: FnDecl) -> Self {
        Self {
            id: decl.id.t,
            type_fields: decl.type_fields.into_iter().map(_TypeField::from).collect(),
            return_type: decl.return_type.map(|t| t.t),
            body: decl.body.t.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LVal {
    Simple(String),
    Field(Box<LVal>, String),
    Subscript(Box<LVal>, Expr),
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
            LVal::Field(l, s) => _LVal::Field(Box::new((*l).into()), s),
            LVal::Subscript(l, s) => _LVal::Subscript(Box::new((*l).into()), s.t.into()),
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
            field_assigns: record.field_assigns.into_iter().map(_FieldAssign::from).collect(),
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
    pub lval: LVal,
    pub expr: Expr,
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
