use crate::{
    ir,
    tmp::{Label, TmpGenerator},
};
use std::{fmt, rc::Rc};

#[derive(Clone)]
pub enum Expr {
    Expr(ir::Expr),
    Stmt(ir::Stmt),
    /// Given true and false destination labels, the function creates a conditional `ir::Stmt` (i.e.
    /// either `Jump` or `CJump`) that evaluates some conditionals and then jumps to one of the
    /// destinations.
    Cond(Rc<dyn Fn(Label, Label) -> ir::Stmt>),
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Expr(expr) => f.debug_tuple("Expr").field(expr).finish(),
            Expr::Stmt(stmt) => f.debug_tuple("Stmt").field(stmt).finish(),
            Expr::Cond(_) => write!(f, "Cond(closure)"),
        }
    }
}

impl PartialEq for Expr {
    fn eq(&self, other: &Expr) -> bool {
        match self {
            Expr::Expr(expr) => {
                if let Expr::Expr(other_expr) = other {
                    expr == other_expr
                } else {
                    false
                }
            }
            Expr::Stmt(stmt) => {
                if let Expr::Stmt(other_stmt) = other {
                    stmt == other_stmt
                } else {
                    false
                }
            }
            Expr::Cond(cond) => {
                if let Expr::Cond(other_cond) = other {
                    let l: *const dyn Fn(Label, Label) -> ir::Stmt = cond.as_ref();
                    let r: *const dyn Fn(Label, Label) -> ir::Stmt = other_cond.as_ref();
                    l == r
                } else {
                    false
                }
            }
        }
    }
}

impl Eq for Expr {}

impl Expr {
    pub fn unwrap_expr(self, tmp_generator: &TmpGenerator) -> ir::Expr {
        match self {
            Expr::Expr(expr) => expr,
            Expr::Stmt(stmt) => ir::Expr::Seq(Box::new(stmt), Box::new(ir::Expr::Const(0))),
            Expr::Cond(gen_stmt) => {
                let result = tmp_generator.new_tmp();
                let true_label = tmp_generator.new_label();
                let false_label = tmp_generator.new_label();
                ir::Expr::Seq(
                    Box::new(ir::Stmt::seq(vec![
                        ir::Stmt::Move(ir::Expr::Tmp(result), ir::Expr::Const(1)),
                        gen_stmt(true_label.clone(), false_label.clone()),
                        ir::Stmt::Label(false_label),
                        ir::Stmt::Move(ir::Expr::Tmp(result), ir::Expr::Const(0)),
                        ir::Stmt::Label(true_label),
                    ])),
                    Box::new(ir::Expr::Tmp(result)),
                )
            }
        }
    }

    pub fn unwrap_stmt(self, tmp_generator: &TmpGenerator) -> ir::Stmt {
        match self {
            Expr::Expr(expr) => ir::Stmt::Expr(expr),
            Expr::Stmt(stmt) => stmt,
            Expr::Cond(gen_stmt) => {
                let label = tmp_generator.new_label();
                ir::Stmt::Seq(
                    Box::new(gen_stmt(label.clone(), label.clone())),
                    Box::new(ir::Stmt::Label(label)),
                )
            }
        }
    }

    pub fn unwrap_cond(self) -> Rc<dyn Fn(Label, Label) -> ir::Stmt> {
        match self {
            Expr::Expr(ir::Expr::Const(0)) => Rc::new(|_, f| ir::Stmt::Jump(ir::Expr::Label(f.clone()), vec![f])),
            Expr::Expr(ir::Expr::Const(1)) => Rc::new(|t, _| ir::Stmt::Jump(ir::Expr::Label(t.clone()), vec![t])),
            Expr::Expr(expr) => {
                Rc::new(move |t, f| ir::Stmt::CJump(expr.clone(), ir::CompareOp::Eq, ir::Expr::Const(0), f, t))
            }
            Expr::Stmt(_) => unreachable!("cannot call unwrap_cond() on Expr::Stmt"),
            Expr::Cond(gen_stmt) => gen_stmt,
        }
    }

    pub fn reference(&self) -> ir::Expr {
        if let Expr::Expr(expr) = self {
            expr.reference()
        } else {
            panic!("cannot take reference of non-Expr");
        }
    }
}
