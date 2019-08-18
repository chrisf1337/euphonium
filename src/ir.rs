use crate::{
    ast,
    tmp::{Label, Tmp},
};

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

    pub fn flatten(stmt: Stmt) -> Vec<Stmt> {
        match stmt {
            Stmt::Seq(fst, snd) => {
                let mut v = Stmt::flatten(*fst);
                v.extend(Stmt::flatten(*snd));
                v
            }
            s => vec![s],
        }
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

impl From<ast::ArithOp> for BinOp {
    fn from(op: ast::ArithOp) -> Self {
        match op {
            ast::ArithOp::Add => BinOp::Add,
            ast::ArithOp::Sub => BinOp::Sub,
            ast::ArithOp::Mul => BinOp::Mul,
            ast::ArithOp::Div => BinOp::Div,
        }
    }
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

impl From<ast::CompareOp> for CompareOp {
    fn from(op: ast::CompareOp) -> Self {
        match op {
            ast::CompareOp::Eq => CompareOp::Eq,
            ast::CompareOp::Ne => CompareOp::Ne,
            ast::CompareOp::Gt => CompareOp::Gt,
            ast::CompareOp::Ge => CompareOp::Ge,
            ast::CompareOp::Lt => CompareOp::Lt,
            ast::CompareOp::Le => CompareOp::Le,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tmp::TmpGenerator;

    #[test]
    fn test_seq() {
        let mut tmp_generator = TmpGenerator::default();

        let stmt1 = Stmt::Label(tmp_generator.new_label());
        let stmt2 = Stmt::Label(tmp_generator.new_label());
        let stmt3 = Stmt::Label(tmp_generator.new_label());

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
