use crate::{
    ast,
    tmp::{Label, Tmp},
};
use std::fmt;

#[cfg(test)]
pub mod interpreter;

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    Const(i64),
    Label(Label),
    Tmp(Tmp),
    BinOp(Box<Expr>, BinOp, Box<Expr>),
    Mem(Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Seq(Box<Stmt>, Box<Expr>),
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Const(c) => write!(f, "{:?}", c),
            Expr::Label(l) => write!(f, "{:?}", l),
            Expr::Tmp(t) => write!(f, "{:?}", t),
            Expr::BinOp(l, op, r) => f.debug_tuple("BinOp").field(l).field(op).field(r).finish(),
            Expr::Mem(e) => f.debug_tuple("Mem").field(e).finish(),
            Expr::Call(fun, args) => f.debug_tuple("Call").field(fun).field(args).finish(),
            Expr::Seq(s, e) => {
                let entries: &[&dyn fmt::Debug] = &[s, e];
                f.debug_list().entries(entries).finish()
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Stmt {
    Move(Expr, Expr),
    Expr(Expr),
    Jump(Expr, Vec<Label>),
    CJump(Expr, CompareOp, Expr, Label, Label),
    Seq(Box<Stmt>, Box<Stmt>),
    Label(Label),
}

impl fmt::Debug for Stmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Stmt::Move(dst, src) => f.debug_tuple("Move").field(dst).field(src).finish(),
            Stmt::Expr(e) => f.debug_tuple("Expr").field(e).finish(),
            Stmt::Jump(l, ls) => f.debug_tuple("Jump").field(l).field(ls).finish(),
            Stmt::CJump(l, op, r, tl, fl) => f
                .debug_tuple("CJump")
                .field(l)
                .field(op)
                .field(r)
                .field(tl)
                .field(fl)
                .finish(),
            Stmt::Seq(s1, s2) => f.debug_list().entries(&[s1, s2]).finish(),
            Stmt::Label(l) => write!(f, "{:?}", l),
        }
    }
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

    pub fn appending(mut self, stmt: Stmt) -> Stmt {
        self = Stmt::Seq(Box::new(self), Box::new(stmt));
        self
    }

    pub fn flatten(&self) -> Vec<Stmt> {
        let mut stmts = vec![];
        match self {
            Stmt::Move(dst, src) => {
                let (src_stmts, src_expr) = src.flatten();
                let (dst_stmts, dst_expr) = dst.flatten();
                stmts.extend(src_stmts);
                stmts.extend(dst_stmts);
                stmts.push(Stmt::Move(dst_expr, src_expr));
                stmts
            }
            Stmt::Expr(expr) => {
                let (mut stmts, expr) = expr.flatten();
                stmts.push(Stmt::Expr(expr.clone()));
                stmts
            }
            Stmt::Jump(expr, labels) => {
                let (mut stmts, expr) = expr.flatten();
                stmts.push(Stmt::Jump(expr.clone(), labels.clone()));
                stmts
            }
            Stmt::CJump(l, op, r, true_label, false_label) => {
                let mut stmts = vec![];
                let (l_stmts, l_expr) = l.flatten();
                let (r_stmts, r_expr) = r.flatten();
                stmts.extend(l_stmts);
                stmts.extend(r_stmts);
                stmts.push(Stmt::CJump(
                    l_expr,
                    *op,
                    r_expr,
                    true_label.clone(),
                    false_label.clone(),
                ));
                stmts
            }
            Stmt::Seq(stmt1, stmt2) => {
                let mut stmts = vec![];
                stmts.extend(stmt1.flatten());
                stmts.extend(stmt2.flatten());
                stmts
            }
            stmt => vec![stmt.clone()],
        }
    }
}

impl Expr {
    pub fn flatten(&self) -> (Vec<Stmt>, Expr) {
        match self {
            Expr::Const(c) => (vec![], Expr::Const(*c)),
            Expr::Label(label) => (vec![], Expr::Label(label.clone())),
            Expr::Tmp(tmp) => (vec![], Expr::Tmp(*tmp)),
            Expr::BinOp(l, op, r) => {
                let mut stmts = vec![];
                let (l_stmts, l_expr) = l.flatten();
                let (r_stmts, r_expr) = r.flatten();
                stmts.extend(l_stmts);
                stmts.extend(r_stmts);
                (stmts, Expr::BinOp(Box::new(l_expr), *op, Box::new(r_expr)))
            }
            Expr::Mem(expr) => {
                let (stmts, expr) = expr.flatten();
                (stmts, Expr::Mem(Box::new(expr)))
            }
            Expr::Call(func, args) => {
                let mut stmts = vec![];
                // Assume that if func is ever not just a label, it must be a Seq whose expr is Label
                let (func_stmts, func_expr) = func.flatten();
                stmts.extend(func_stmts);

                let mut arg_exprs = vec![];
                for arg in args {
                    let (arg_stmts, arg_expr) = arg.flatten();
                    stmts.extend(arg_stmts);
                    arg_exprs.push(arg_expr);
                }
                (stmts, Expr::Call(Box::new(func_expr), arg_exprs))
            }
            Expr::Seq(stmt, expr) => {
                let mut stmts = stmt.flatten();
                let (expr_stmts, expr) = expr.flatten();
                stmts.extend(expr_stmts);
                (stmts, expr)
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
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

impl fmt::Debug for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
            BinOp::Div => write!(f, "/"),
            BinOp::And => write!(f, "&&"),
            BinOp::Or => write!(f, "||"),
            BinOp::LShift => write!(f, "<<"),
            BinOp::RShift => write!(f, ">>"),
            BinOp::ArithRShift => write!(f, "A>>"),
            BinOp::Xor => write!(f, "^"),
        }
    }
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
    use std::collections::HashMap;

    fn tmp_eq_in_expr(tmp_table: &mut HashMap<Tmp, Tmp>, expr1: &Expr, expr2: &Expr) -> bool {
        match (expr1, expr2) {
            (Expr::Tmp(tmp1), Expr::Tmp(tmp2)) => {
                if let Some(expected_tmp2) = tmp_table.get(tmp1) {
                    if tmp2 != expected_tmp2 {
                        return false;
                    }
                } else {
                    tmp_table.insert(*tmp1, *tmp2);
                }
                true
            }
            (Expr::BinOp(l1, _, r1), Expr::BinOp(l2, _, r2)) => {
                tmp_eq_in_expr(tmp_table, l1, l2) && tmp_eq_in_expr(tmp_table, r1, r2)
            }
            (Expr::Mem(expr1), Expr::Mem(expr2)) => tmp_eq_in_expr(tmp_table, expr1, expr2),
            (Expr::Call(func1, args1), Expr::Call(func2, args2)) => {
                tmp_eq_in_expr(tmp_table, func1, func2)
                    && args1
                        .iter()
                        .zip(args2.iter())
                        .fold(true, |acc, (arg1, arg2)| acc && tmp_eq_in_expr(tmp_table, arg1, arg2))
            }
            (Expr::Seq(stmt1, expr1), Expr::Seq(stmt2, expr2)) => {
                tmp_eq_in_stmt(tmp_table, stmt1, stmt2) && tmp_eq_in_expr(tmp_table, expr1, expr2)
            }
            _ => expr1 == expr2,
        }
    }

    fn tmp_eq_in_stmt(tmp_table: &mut HashMap<Tmp, Tmp>, stmt1: &Stmt, stmt2: &Stmt) -> bool {
        match (stmt1, stmt2) {
            (Stmt::Move(dst1, src1), Stmt::Move(dst2, src2)) => {
                tmp_eq_in_expr(tmp_table, dst1, dst2) && tmp_eq_in_expr(tmp_table, src1, src2)
            }
            (Stmt::Expr(expr1), Stmt::Expr(expr2)) => tmp_eq_in_expr(tmp_table, expr1, expr2),
            (Stmt::Jump(expr1, _), Stmt::Jump(expr2, _)) => tmp_eq_in_expr(tmp_table, expr1, expr2),
            (Stmt::CJump(l1, _, r1, _, _), Stmt::CJump(l2, _, r2, _, _)) => {
                tmp_eq_in_expr(tmp_table, l1, l2) && tmp_eq_in_expr(tmp_table, r1, r2)
            }
            (Stmt::Seq(l_stmt1, l_stmt2), Stmt::Seq(r_stmt1, r_stmt2)) => {
                tmp_eq_in_stmt(tmp_table, l_stmt1, r_stmt1) && tmp_eq_in_stmt(tmp_table, l_stmt2, r_stmt2)
            }
            _ => stmt1 == stmt2,
        }
    }

    fn tmp_eq(stmts1: &[Stmt], stmts2: &[Stmt]) -> bool {
        let mut tmp_table = HashMap::new();
        if stmts1.len() != stmts2.len() {
            return false;
        }
        for (stmt1, stmt2) in stmts1.iter().zip(stmts2.iter()) {
            if !tmp_eq_in_stmt(&mut tmp_table, stmt1, stmt2) {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_seq() {
        let tmp_generator = TmpGenerator::default();

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

    #[test]
    fn test_flatten() {
        let stmts = Stmt::Move(
            Expr::BinOp(
                Box::new(Expr::Seq(
                    Box::new(Stmt::Expr(Expr::Const(0))),
                    Box::new(Expr::Const(0)),
                )),
                BinOp::Add,
                Box::new(Expr::Const(8)),
            ),
            Expr::Const(0),
        )
        .flatten();
        assert!(tmp_eq(
            &stmts,
            &[
                Stmt::Expr(Expr::Const(0)),
                Stmt::Move(
                    Expr::BinOp(Box::new(Expr::Const(0)), BinOp::Add, Box::new(Expr::Const(8)),),
                    Expr::Const(0),
                ),
            ]
        ));
    }
}
