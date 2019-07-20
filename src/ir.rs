use crate::tmp::{Label, Tmp};

#[cfg(test)]
pub mod interpreter;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Const(i64),
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

impl Stmt {
    pub fn seq(mut stmts: Vec<Stmt>) -> Stmt {
        if stmts.is_empty() {
            panic!("seq called with no statements");
        }

        let last = stmts.pop().unwrap();
        *stmts
            .into_iter()
            .rev()
            .fold(Box::new(last), |acc, next| Box::new(Stmt::Seq(Box::new(next), acc)))
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tmp::TmpGenerator;

    #[test]
    fn test_seq() {
        let mut tmp_generator = TmpGenerator::default();

        let stmt1 = Stmt::Label(tmp_generator.new_anonymous_label());
        let stmt2 = Stmt::Label(tmp_generator.new_anonymous_label());
        let stmt3 = Stmt::Label(tmp_generator.new_anonymous_label());

        assert_eq!(Stmt::seq(vec![stmt1.clone()]), stmt1.clone());
        assert_eq!(
            Stmt::seq(vec![stmt1.clone(), stmt2.clone()]),
            Stmt::Seq(Box::new(stmt1.clone()), Box::new(stmt2.clone()))
        );
        assert_eq!(
            Stmt::seq(vec![stmt1.clone(), stmt2.clone(), stmt3.clone()]),
            Stmt::Seq(
                Box::new(stmt1.clone()),
                Box::new(Stmt::Seq(Box::new(stmt2.clone()), Box::new(stmt3.clone())))
            )
        );
    }
}
