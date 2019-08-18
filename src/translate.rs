use crate::{
    ast,
    frame::{self, Frame},
    ir,
    tmp::{self, Label, TmpGenerator},
};
use std::{collections::HashMap, rc::Rc};

mod expr;
mod level;

pub use expr::*;
pub use level::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Access {
    pub level_label: Label,
    pub access: frame::Access,
}

pub struct Translator<'a> {
    tmp_generator: &'a mut TmpGenerator,
}

impl<'a> Translator<'a> {
    pub fn new(tmp_generator: &'a mut TmpGenerator) -> Self {
        Translator { tmp_generator }
    }

    pub fn translate_expr(&mut self, expr: &ast::Expr) -> Expr {
        match &expr.t {
            ast::ExprType::Compare(compare) => self.translate_compare(compare),
            _ => unimplemented!(),
        }
    }

    pub fn translate_compare(&mut self, expr: &ast::Compare) -> Expr {
        let l = self.translate_expr(&expr.l).unwrap_expr(self.tmp_generator);
        let r = self.translate_expr(&expr.r).unwrap_expr(self.tmp_generator);
        let op = expr.op.t.into();
        Expr::Cond(Rc::new(move |true_label, false_label| {
            ir::Stmt::CJump(l.clone(), op, r.clone(), true_label, false_label)
        }))
    }

    pub fn translate_simple_var(&mut self, levels: &HashMap<Label, Level>, access: Access, level_label: Label) -> Expr {
        fn _translate_simple_var(
            levels: &HashMap<Label, Level>,
            access: Access,
            level_label: Label,
            acc_expr: ir::Expr,
        ) -> ir::Expr {
            if level_label == Label::top() {
                panic!("simple_var() reached top");
            }
            if level_label == access.level_label {
                Frame::expr(&access.access, &acc_expr)
            } else {
                let level = &levels[&level_label];
                let parent_label = level.parent_label.expect("level has no parent");
                _translate_simple_var(levels, access, parent_label, ir::Expr::Mem(Box::new(acc_expr)))
            }
        }

        Expr::Expr(_translate_simple_var(
            levels,
            access,
            level_label,
            ir::Expr::Tmp(tmp::FP.clone()),
        ))
    }

    /// `array_expr` is a dereferenced pointer to memory on the heap where the first element of the array lives.
    pub fn translate_pointer_offset(&mut self, array_expr: &Expr, index_expr: &Expr) -> Expr {
        Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
            Box::new(array_expr.clone().unwrap_expr(self.tmp_generator)),
            ir::BinOp::Add,
            Box::new(ir::Expr::BinOp(
                Box::new(index_expr.clone().unwrap_expr(self.tmp_generator)),
                ir::BinOp::Mul,
                Box::new(ir::Expr::Const(frame::WORD_SIZE)),
            )),
        ))))
    }

    pub fn translate_arith(&mut self, expr: &'a ast::Arith) -> Expr {
        Expr::Expr(ir::Expr::BinOp(
            Box::new(self.translate_expr(&expr.l).unwrap_expr(self.tmp_generator)),
            expr.op.t.into(),
            Box::new(self.translate_expr(&expr.r).unwrap_expr(self.tmp_generator)),
        ))
    }

    pub fn translate_if(&mut self, expr: &'a ast::If) -> Expr {
        let true_label = self.tmp_generator.new_label();
        let false_label = self.tmp_generator.new_label();
        let join_label = self.tmp_generator.new_label();
        let result = self.tmp_generator.new_tmp();
        let cond_gen = self.translate_expr(&expr.cond).unwrap_cond();

        let mut injected_labels = None;
        let mut then_expr_was_cond = false;
        let mut else_expr_was_cond = false;

        let then_instr = match self.translate_expr(&expr.then_expr) {
            Expr::Stmt(stmt) => {
                // If the then expression is a statement, don't bother unwrapping into an Expr::Expr and just run it
                // directly.
                stmt
            }
            Expr::Cond(cond_gen) => {
                let true_label = self.tmp_generator.new_label();
                let false_label = self.tmp_generator.new_label();
                let stmt = ir::Stmt::seq(vec![cond_gen(true_label.clone(), false_label.clone())]);
                injected_labels = Some((true_label, false_label));
                then_expr_was_cond = true;
                stmt
            }
            expr => ir::Stmt::Move(ir::Expr::Tmp(result.clone()), expr.unwrap_expr(self.tmp_generator)),
        };

        let else_instr = if let Some(expr) = &expr.else_expr {
            let else_instr = match self.translate_expr(expr) {
                Expr::Stmt(stmt) => {
                    // If the then expression is a statement, don't bother unwrapping into an Expr::Expr and just run it
                    // directly.
                    stmt
                }
                Expr::Cond(cond_gen) => {
                    let (true_label, false_label) = injected_labels
                        .get_or_insert_with(|| (self.tmp_generator.new_label(), self.tmp_generator.new_label()));
                    let stmt = ir::Stmt::seq(vec![cond_gen(true_label.clone(), false_label.clone())]);
                    else_expr_was_cond = true;
                    stmt
                }
                expr => ir::Stmt::Move(ir::Expr::Tmp(result.clone()), expr.unwrap_expr(self.tmp_generator)),
            };
            Some(else_instr)
        } else {
            None
        };

        let cond_stmt = if else_instr.is_some() {
            cond_gen(true_label.clone(), false_label.clone())
        } else {
            // Jump directly to join_label if there's no else branch
            cond_gen(true_label.clone(), join_label.clone())
        };

        let mut seq = vec![cond_stmt, ir::Stmt::Label(true_label), then_instr];
        // then_instr will be a CJump if then_expr was Cond, so only insert a Jump if then_expr wasn't Cond
        if !then_expr_was_cond {
            seq.extend_from_slice(&[ir::Stmt::Jump(
                ir::Expr::Label(join_label.clone()),
                vec![join_label.clone()],
            )]);
        }
        if let Some(else_instr) = else_instr {
            seq.extend(vec![ir::Stmt::Label(false_label), else_instr]);
            // Same logic here
            if !else_expr_was_cond {
                seq.extend_from_slice(&[ir::Stmt::Jump(
                    ir::Expr::Label(join_label.clone()),
                    vec![join_label.clone()],
                )]);
            }
        }

        if then_expr_was_cond || else_expr_was_cond {
            let (true_label, false_label) = injected_labels.unwrap();
            seq.extend_from_slice(&[
                ir::Stmt::Label(true_label),
                ir::Stmt::Move(ir::Expr::Tmp(result.clone()), ir::Expr::Const(1)),
                ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
                ir::Stmt::Label(false_label),
                ir::Stmt::Move(ir::Expr::Tmp(result.clone()), ir::Expr::Const(0)),
                ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
            ]);
        }
        seq.push(ir::Stmt::Label(join_label));
        Expr::Expr(ir::Expr::Seq(
            Box::new(ir::Stmt::seq(seq)),
            Box::new(ir::Expr::Tmp(result)),
        ))
    }
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
                Label(1) => Level {
                    parent_label: Some(Label::top()),
                    frame: Frame::new(&mut tmp_generator, "f", &[]),
                },
            };
            let level = levels.get_mut(&Label(1)).unwrap();
            let local = level.alloc_local(&mut tmp_generator, true);
            let mut translator = Translator::new(&mut tmp_generator);

            assert_eq!(
                translator.translate_simple_var(&levels, local, Label(1)),
                Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                    Box::new(ir::Expr::Tmp(tmp::FP.clone())),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                ))))
            );
        }

        {
            let mut tmp_generator = TmpGenerator::default();
            let mut levels = hashmap! {
                Label::top() => Level::top(),
                Label(1) => Level {
                    parent_label: Some(Label::top()),
                    frame: Frame::new(&mut tmp_generator, "f", &[])
                },
                Label(2) => Level {
                    parent_label: Some(Label(1)),
                    frame: Frame::new(&mut tmp_generator, "g", &[])
                }
            };
            let level = levels.get_mut(&Label(1)).unwrap();
            let local = level.alloc_local(&mut tmp_generator, true);
            let mut translator = Translator::new(&mut tmp_generator);

            assert_eq!(
                translator.translate_simple_var(&levels, local, Label(2)),
                Expr::Expr(ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                    Box::new(ir::Expr::Mem(Box::new(ir::Expr::Tmp(tmp::FP.clone())))),
                    ir::BinOp::Add,
                    Box::new(ir::Expr::Const(-frame::WORD_SIZE))
                ))))
            );
        }
    }

    #[test]
    fn test_translate_if() {
        let if_expr = ast::If {
            cond: zspan!(ast::ExprType::BoolLiteral(true)),
            then_expr: zspan!(ast::ExprType::BoolLiteral(true)),
            else_expr: None,
        };
    }
}
