#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl {
    pub ty: DeclType,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclType {
    Type(TypeDecl),
    Fn(FnDecl),
    Error,
}

/// type A = ...
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDecl {
    pub type_id: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Type(String),
    /// { a: int, b: int }
    Record(Vec<TypeField>),
    Array(Box<Type>, usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeField {
    pub id: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pattern {
    Wildcard,
    String(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Let {
    pub pattern: Pattern,
    pub mutable: bool,
    pub ty: Option<Type>,
    pub expr: Expr,
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
            span: Span { l: span.0, r: span.1 }
        }
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
    pub id: String,
    pub type_fields: Vec<TypeField>,
    pub return_type: Option<Type>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LVal {
    Simple(String),
    Field(Box<LVal>, String),
    Subscript(Box<LVal>, Expr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnCall {
    pub id: String,
    pub args: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub id: String,
    pub field_assigns: Vec<FieldAssign>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAssign {
    pub id: String,
    pub expr: Expr,
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
pub struct If {
    pub cond: Expr,
    pub then_expr: Expr,
    pub else_expr: Option<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct For {
    pub index: String,
    pub range: Expr,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct While {
    pub cond: Expr,
    pub body: Expr,
}
