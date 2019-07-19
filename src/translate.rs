use crate::{
    frame::{self, Frame},
    ir,
    tmp::{Label, TmpGenerator},
};
use std::{collections::HashMap, fmt};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Access {
    pub level_label: Label,
    pub access: frame::Access,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Level {
    pub parent_label: Option<Label>,
    frame: Frame,
}

impl Level {
    pub fn new(tmp_generator: &mut TmpGenerator, parent_label: Option<Label>, name: &str, formals: &[bool]) -> Self {
        // Add static link
        let mut frame_formals = vec![false; formals.len() + 1];
        frame_formals[0] = true;
        frame_formals[1..].copy_from_slice(formals);
        Self {
            parent_label,
            frame: Frame::new(tmp_generator, name, &frame_formals),
        }
    }

    pub fn top() -> Self {
        Level {
            parent_label: None,
            frame: Frame {
                label: Label("top".to_owned()),
                formals: vec![],
                n_locals: 0,
            },
        }
    }

    pub fn formals(&self) -> Vec<Access> {
        // Strip static link
        self.frame.formals[1..]
            .iter()
            .cloned()
            .map(|formal| Access {
                level_label: self.frame.label.clone(),
                access: formal,
            })
            .collect()
    }

    pub fn alloc_local(&mut self, tmp_generator: &mut TmpGenerator, escapes: bool) -> Access {
        Access {
            level_label: self.frame.label.clone(),
            access: self.frame.alloc_local(tmp_generator, escapes),
        }
    }

    pub fn label(&self) -> &Label {
        &self.frame.label
    }
}

pub enum Expr {
    Expr(ir::Expr),
    Stmt(ir::Stmt),
    /// Given true and false destination labels, the function creates a conditional `ir::Stmt` (i.e.
    /// either `Jump` or `CJump`) that evaluates some conditionals and then jumps to one of the
    /// destinations.
    Cond(Box<dyn Fn(Label, Label) -> ir::Stmt>),
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Expr::Expr(expr) => f.debug_tuple("Expr").field(expr).finish(),
            Expr::Stmt(stmt) => f.debug_tuple("Stmt").field(stmt).finish(),
            Expr::Cond(cond) => write!(f, "Cond(closure)"),
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
    pub fn unwrap_expr(self, tmp_generator: &mut TmpGenerator) -> ir::Expr {
        match self {
            Expr::Expr(expr) => expr,
            Expr::Stmt(stmt) => ir::Expr::Seq(Box::new(stmt), Box::new(ir::Expr::Const(0))),
            Expr::Cond(gen_stmt) => {
                let result = tmp_generator.new_tmp();
                let true_label = tmp_generator.new_anonymous_label();
                let false_label = tmp_generator.new_anonymous_label();
                ir::Expr::Seq(
                    Box::new(ir::Stmt::seq(vec![
                        ir::Stmt::Move(ir::Expr::Tmp(result.clone()), ir::Expr::Const(1)),
                        gen_stmt(true_label.clone(), false_label.clone()),
                        ir::Stmt::Label(false_label.clone()),
                        ir::Stmt::Move(ir::Expr::Tmp(result.clone()), ir::Expr::Const(0)),
                        ir::Stmt::Label(true_label.clone()),
                    ])),
                    Box::new(ir::Expr::Tmp(result)),
                )
            }
        }
    }

    pub fn unwrap_stmt(self, tmp_generator: &mut TmpGenerator) -> ir::Stmt {
        match self {
            Expr::Expr(expr) => ir::Stmt::Expr(expr),
            Expr::Stmt(stmt) => stmt,
            Expr::Cond(gen_stmt) => {
                let label = tmp_generator.new_anonymous_label();
                ir::Stmt::Seq(
                    Box::new(gen_stmt(label.clone(), label.clone())),
                    Box::new(ir::Stmt::Label(label)),
                )
            }
        }
    }

    pub fn unwrap_cond(self) -> Box<dyn Fn(Label, Label) -> ir::Stmt> {
        match self {
            Expr::Expr(ir::Expr::Const(0)) => Box::new(|_, f| ir::Stmt::Jump(ir::Expr::Label(f.clone()), vec![f])),
            Expr::Expr(ir::Expr::Const(1)) => Box::new(|t, _| ir::Stmt::Jump(ir::Expr::Label(t.clone()), vec![t])),
            Expr::Expr(expr) => {
                Box::new(move |t, f| ir::Stmt::CJump(expr.clone(), ir::CompareOp::Eq, ir::Expr::Const(0), f, t))
            }
            Expr::Stmt(_) => unreachable!("cannot call unwrap_cond() on Expr::Stmt"),
            Expr::Cond(gen_stmt) => gen_stmt,
        }
    }
}

pub fn translate_simple_var(levels: &HashMap<Label, Level>, access: &Access, level_label: &Label) -> Expr {
    fn _translate_simple_var(
        levels: &HashMap<Label, Level>,
        access: &Access,
        level_label: &Label,
        acc_expr: ir::Expr,
    ) -> ir::Expr {
        if level_label == &Label::top() {
            panic!("simple_var() reached top");
        }
        if level_label == &access.level_label {
            Frame::expr(&access.access, &acc_expr)
        } else {
            let level = &levels[level_label];
            let parent_label = level.parent_label.as_ref().expect("level has no parent");
            _translate_simple_var(levels, access, parent_label, ir::Expr::Mem(Box::new(acc_expr)))
        }
    }

    Expr::Expr(_translate_simple_var(
        levels,
        access,
        level_label,
        ir::Expr::Tmp(frame::FP.clone()),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frame, tmp::TmpGenerator};
    use maplit::hashmap;

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
            let mut levels = hashmap! {
                Label::top() => Level::top(),
                Label("Lf0".to_owned()) => Level {
                    parent_label: Some(Label::top()),
                    frame: Frame::new(&mut tmp_generator, "f", &[]),
                },
            };
            let level = levels.get_mut(&Label("Lf0".to_owned())).unwrap();
            let local = level.alloc_local(&mut tmp_generator, true);

            assert_eq!(
                translate_simple_var(&levels, &local, &Label("Lf0".to_owned())),
                Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                    Box::new(ir::Expr::Tmp(frame::FP.clone())),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                ))))
            );
        }

        {
            let mut tmp_generator = TmpGenerator::default();
            let mut levels = hashmap! {
                Label::top() => Level::top(),
                Label("Lf0".to_owned()) => Level {
                    parent_label: Some(Label::top()),
                    frame: Frame::new(&mut tmp_generator, "f", &[])
                },
                Label("Lg1".to_owned()) => Level {
                    parent_label: Some(Label("Lf0".to_owned())),
                    frame: Frame::new(&mut tmp_generator, "g", &[])
                }
            };
            let level = levels.get_mut(&Label("Lf0".to_owned())).unwrap();
            let local = level.alloc_local(&mut tmp_generator, true);

            assert_eq!(
                translate_simple_var(&levels, &local, &Label("Lg1".to_owned())),
                Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                    Box::new(ir::Expr::Mem(Box::new(ir::Expr::Tmp(frame::FP.clone())))),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                ))))
            )
        }
    }
}
