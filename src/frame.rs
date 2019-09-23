use crate::{
    fragment::{Fragment, StringFragment},
    ir,
    tmp::{Label, Tmp, TmpGenerator},
};

pub const WORD_SIZE: i64 = std::mem::size_of::<u64>() as i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    InFrame(i64),
    InReg(Tmp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub name: String,
    pub label: Label,
    pub formals: Vec<Access>,
    pub n_locals: usize,
}

impl Frame {
    pub fn new(tmp_generator: &mut TmpGenerator, name: impl Into<String>, formals: &[bool]) -> Self {
        let mut offset: i64 = 0;
        let mut accesses = vec![];
        for &formal in formals {
            if formal {
                accesses.push(Access::InFrame(offset));
                offset += -WORD_SIZE;
            } else {
                accesses.push(Access::InReg(tmp_generator.new_tmp()));
            }
        }

        let name = name.into();
        let label = tmp_generator.new_named_label(&name);
        Frame {
            name,
            label,
            formals: accesses,
            n_locals: 0,
        }
    }

    fn escaping_formals(&self) -> Vec<&Access> {
        self.formals
            .iter()
            .filter(|access| if let Access::InFrame(_) = access { true } else { false })
            .collect()
    }

    pub fn alloc_local(&mut self, tmp_generator: &mut TmpGenerator, escapes: bool) -> Access {
        let escaping_formals_len = self.escaping_formals().len();
        if escapes {
            let access = Access::InFrame(-WORD_SIZE * (self.n_locals + escaping_formals_len) as i64);
            self.n_locals += 1;
            access
        } else {
            Access::InReg(tmp_generator.new_tmp())
        }
    }

    /// For in-register temporaries, simply returns the temporary. For in-memory access, returns the
    /// dereferenced address, suitable for assignment to or from.
    pub fn expr(access: Access, fp_expr: &ir::Expr) -> ir::Expr {
        match access {
            Access::InReg(tmp) => ir::Expr::Tmp(tmp),
            Access::InFrame(offset) => ir::Expr::Mem(Box::new(ir::Expr::BinOp(
                Box::new(fp_expr.clone()),
                ir::BinOp::Add,
                Box::new(ir::Expr::Const(offset)),
            ))),
        }
    }

    pub fn external_call(fn_name: impl Into<String>, args: Vec<ir::Expr>) -> ir::Expr {
        ir::Expr::Call(Box::new(ir::Expr::Label(Label(fn_name.into()))), args)
    }

    // TODO: Dedup literals.
    pub fn string(tmp_generator: &mut TmpGenerator, string: impl Into<String>) -> (ir::Expr, Fragment) {
        let label = tmp_generator.new_label();
        (
            ir::Expr::Label(label.clone()),
            Fragment::String(StringFragment {
                label,
                string: string.into(),
            }),
        )
    }
}
