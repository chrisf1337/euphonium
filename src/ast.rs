#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl<Expr: StrippableExpr> {
    pub ty: DeclType<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrippedDecl {
    pub ty: DeclType<StrippedExpr>,
}

impl<Expr: StrippableExpr> Decl<Expr> {
    pub fn strip(self) -> StrippedDecl {
        StrippedDecl {
            ty: self.ty.strip(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclType<Expr: StrippableExpr> {
    Type(TypeDecl),
    Fn(FnDecl<Expr>),
    Error,
}

impl<Expr: StrippableExpr> DeclType<Expr> {
    fn strip(self) -> DeclType<StrippedExpr> {
        match self {
            DeclType::Type(decl) => DeclType::Type(decl),
            DeclType::Fn(decl) => DeclType::Fn(decl.strip()),
            DeclType::Error => DeclType::Error,
        }
    }
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
pub struct Let<Expr: StrippableExpr> {
    pub pattern: Pattern,
    pub mutable: bool,
    pub ty: Option<Type>,
    pub expr: Expr,
}

impl<Expr: StrippableExpr> Let<Expr> {
    fn strip(self) -> Let<StrippedExpr> {
        Let {
            pattern: self.pattern,
            mutable: self.mutable,
            ty: self.ty,
            expr: self.expr.strip(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Span {
    pub l: usize,
    pub r: usize,
}

pub trait StrippableExpr {
    fn strip(self) -> StrippedExpr;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    pub ty: ExprType<Self>,
    pub span: Span,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub t: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn unwrap(self) -> T {
        self.t
    }
}

impl StrippableExpr for Expr {
    fn strip(self) -> StrippedExpr {
        StrippedExpr {
            ty: self.ty.strip(),
        }
    }
}

impl Expr {
    pub fn new(ty: ExprType<Self>, span: (usize, usize)) -> Self {
        Self {
            ty,
            span: Span {
                l: span.0,
                r: span.1,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrippedExpr {
    pub ty: ExprType<Self>,
}

impl StrippableExpr for StrippedExpr {
    fn strip(self) -> StrippedExpr {
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprType<Expr: StrippableExpr> {
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
    LVal(Box<LVal<Expr>>),
    Let(Box<Let<Expr>>),
    FnCall(Box<FnCall<Expr>>),
    Record(Box<Record<Expr>>),
    Assign(Box<Assign<Expr>>),
    Array(Box<Array<Expr>>),
    If(Box<If<Expr>>),
    Range(Box<Expr>, Box<Expr>),
    For(Box<For<Expr>>),
    While(Box<While<Expr>>),
    Compare(Box<Expr>, CompareOp, Box<Expr>),
}

impl<Expr: StrippableExpr> ExprType<Expr> {
    fn strip(self) -> ExprType<StrippedExpr> {
        match self {
            ExprType::Seq(exprs, returns) => ExprType::Seq(
                exprs.into_iter().map(StrippableExpr::strip).collect(),
                returns,
            ),
            ExprType::String(s) => ExprType::String(s),
            ExprType::Number(n) => ExprType::Number(n),
            ExprType::Neg(expr) => ExprType::Neg(Box::new(expr.strip())),
            ExprType::Arith(l, op, r) => {
                ExprType::Arith(Box::new(l.strip()), op, Box::new(r.strip()))
            }
            ExprType::Unit => ExprType::Unit,
            ExprType::BoolLiteral(b) => ExprType::BoolLiteral(b),
            ExprType::Not(b) => ExprType::Not(Box::new(b.strip())),
            ExprType::Bool(l, op, r) => {
                ExprType::Bool(Box::new(l.strip()), op, Box::new(r.strip()))
            }
            ExprType::Continue => ExprType::Continue,
            ExprType::Break => ExprType::Break,
            ExprType::LVal(expr) => ExprType::LVal(Box::new(expr.strip())),
            ExprType::Let(expr) => ExprType::Let(Box::new(expr.strip())),
            ExprType::FnCall(expr) => ExprType::FnCall(Box::new(expr.strip())),
            ExprType::Record(expr) => ExprType::Record(Box::new(expr.strip())),
            ExprType::Assign(expr) => ExprType::Assign(Box::new(expr.strip())),
            ExprType::Array(expr) => ExprType::Array(Box::new(expr.strip())),
            ExprType::If(expr) => ExprType::If(Box::new(expr.strip())),
            ExprType::Range(from, to) => {
                ExprType::Range(Box::new(from.strip()), Box::new(to.strip()))
            }
            ExprType::For(expr) => ExprType::For(Box::new(expr.strip())),
            ExprType::While(expr) => ExprType::While(Box::new(expr.strip())),
            ExprType::Compare(l, op, r) => {
                ExprType::Compare(Box::new(l.strip()), op, Box::new(r.strip()))
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
pub struct FnDecl<Expr: StrippableExpr> {
    pub id: String,
    pub type_fields: Vec<TypeField>,
    pub return_type: Option<Type>,
    pub body: Expr,
}

impl<Expr: StrippableExpr> FnDecl<Expr> {
    fn strip(self) -> FnDecl<StrippedExpr> {
        FnDecl {
            id: self.id,
            type_fields: self.type_fields,
            return_type: self.return_type,
            body: self.body.strip(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LVal<Expr: StrippableExpr> {
    Simple(String),
    Field(Box<LVal<Expr>>, String),
    Subscript(Box<LVal<Expr>>, Expr),
}

impl<Expr: StrippableExpr> LVal<Expr> {
    fn strip(self) -> LVal<StrippedExpr> {
        match self {
            LVal::Simple(s) => LVal::Simple(s),
            LVal::Field(lval, field) => LVal::Field(Box::new(lval.strip()), field),
            LVal::Subscript(lval, sub) => LVal::Subscript(Box::new(lval.strip()), sub.strip()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FnCall<Expr: StrippableExpr> {
    pub id: String,
    pub args: Vec<Expr>,
}

impl<Expr: StrippableExpr> FnCall<Expr> {
    fn strip(self) -> FnCall<StrippedExpr> {
        FnCall {
            id: self.id,
            args: self.args.into_iter().map(StrippableExpr::strip).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Record<Expr: StrippableExpr> {
    pub id: String,
    pub field_assigns: Vec<FieldAssign<Expr>>,
}

impl<Expr: StrippableExpr> Record<Expr> {
    fn strip(self) -> Record<StrippedExpr> {
        Record {
            id: self.id,
            field_assigns: self
                .field_assigns
                .into_iter()
                .map(FieldAssign::strip)
                .collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAssign<Expr: StrippableExpr> {
    pub id: String,
    pub expr: Expr,
}

impl<Expr: StrippableExpr> FieldAssign<Expr> {
    fn strip(self) -> FieldAssign<StrippedExpr> {
        FieldAssign {
            id: self.id,
            expr: self.expr.strip(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assign<Expr: StrippableExpr> {
    pub lval: LVal<Expr>,
    pub expr: Expr,
}

impl<Expr: StrippableExpr> Assign<Expr> {
    fn strip(self) -> Assign<StrippedExpr> {
        Assign {
            lval: self.lval.strip(),
            expr: self.expr.strip(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Array<Expr: StrippableExpr> {
    pub initial_value: Expr,
    pub len: Expr,
}

impl<Expr: StrippableExpr> Array<Expr> {
    fn strip(self) -> Array<StrippedExpr> {
        Array {
            initial_value: self.initial_value.strip(),
            len: self.len.strip(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct If<Expr: StrippableExpr> {
    pub cond: Expr,
    pub then_expr: Expr,
    pub else_expr: Option<Expr>,
}

impl<Expr: StrippableExpr> If<Expr> {
    fn strip(self) -> If<StrippedExpr> {
        If {
            cond: self.cond.strip(),
            then_expr: self.then_expr.strip(),
            else_expr: self.else_expr.map(StrippableExpr::strip),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct For<Expr> {
    pub index: String,
    pub range: Expr,
    pub body: Expr,
}

impl<Expr: StrippableExpr> For<Expr> {
    fn strip(self) -> For<StrippedExpr> {
        For {
            index: self.index,
            range: self.range.strip(),
            body: self.body.strip(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct While<Expr: StrippableExpr> {
    pub cond: Expr,
    pub body: Expr,
}

impl<Expr: StrippableExpr> While<Expr> {
    fn strip(self) -> While<StrippedExpr> {
        While {
            cond: self.cond.strip(),
            body: self.body.strip(),
        }
    }
}
