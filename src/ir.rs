use crate::tmp::{Label, Tmp};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Const(i32),
    Label(Label),
    Tmp(Tmp),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    Mem(Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Seq(Box<Stmt>, Box<Expr>),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Stmt {
    Move(Expr, Expr),
    Expr(Expr),
    Jump(Expr, Vec<Label>),
    CJump(Expr, CompareOp, Expr, Label, Label),
    Seq(Box<Stmt>, Box<Stmt>),
    Label(Label),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    LShift,
    RShift,
    ArithRShift,
    Xor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    Ult,
    Ule,
    Ugt,
    Uge,
}
