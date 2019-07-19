use crate::{
    ir,
    tmp::{Label, Tmp, TmpGenerator},
};
use lazy_static::lazy_static;

pub const WORD_SIZE: i32 = 8;

lazy_static! {
    pub static ref FP: Tmp = Tmp("FP".to_owned());
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Access {
    InFrame(i32),
    InReg(Tmp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub label: Label,
    pub formals: Vec<Access>,
    pub n_locals: usize,
}

impl Frame {
    pub fn new(tmp_generator: &mut TmpGenerator, name: &str, formals: &[bool]) -> Self {
        let mut offset: i32 = 0;
        let mut accesses = vec![];
        for &formal in formals {
            if formal {
                accesses.push(Access::InFrame(offset));
                offset += WORD_SIZE;
            } else {
                accesses.push(Access::InReg(tmp_generator.new_tmp()));
            }
        }

        Frame {
            label: tmp_generator.new_label(name),
            formals: accesses,
            n_locals: 0,
        }
    }

    pub fn alloc_local(&mut self, tmp_generator: &mut TmpGenerator, escapes: bool) -> Access {
        if escapes {
            self.n_locals += 1;
            Access::InFrame(-WORD_SIZE * self.n_locals as i32)
        } else {
            Access::InReg(tmp_generator.new_tmp())
        }
    }

    pub fn expr(access: &Access, fp_expr: &ir::Expr) -> ir::Expr {
        match access {
            Access::InReg(tmp) => ir::Expr::Tmp(tmp.clone()),
            Access::InFrame(offset) => ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                Box::new(fp_expr.clone()),
                ir::BinOp::Add,
                Box::new(ir::Expr::Const(*offset)),
            ))),
        }
    }
}
