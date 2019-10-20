use crate::{
    fragment::{Fragment, StringFragment},
    frame, ir,
    tmp::{Label, Tmp, TmpGenerator},
};

pub const WORD_SIZE: i64 = std::mem::size_of::<u64>() as i64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Access {
    pub ty: AccessType,
    pub size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AccessType {
    InFrame(i64),
    InReg(Tmp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub name: String,
    pub label: Label,
    pub formals: Vec<Access>,
    pub locals_offset: i64,
}

impl Frame {
    pub fn new(tmp_generator: &TmpGenerator, name: impl Into<String>, formals: &[(usize, bool)]) -> Self {
        let mut offset: i64 = 0;
        let mut accesses = vec![];
        for &(size, formal) in formals {
            if formal {
                accesses.push(Access {
                    ty: AccessType::InFrame(offset),
                    size,
                });
                offset -= size as i64;
            } else {
                accesses.push(Access {
                    ty: AccessType::InReg(tmp_generator.new_tmp()),
                    size,
                });
            }
        }

        let name = name.into();
        let label = tmp_generator.new_named_label(&name);
        Frame {
            name,
            label,
            formals: accesses,
            locals_offset: offset,
        }
    }

    pub fn alloc_local(&mut self, tmp_generator: &TmpGenerator, size: usize, escapes: bool) -> Access {
        if escapes {
            let aligned_size = if size % frame::WORD_SIZE as usize == 0 {
                size
            } else {
                size + size % frame::WORD_SIZE as usize
            };
            let access = Access {
                ty: AccessType::InFrame(self.locals_offset),
                size,
            };
            self.locals_offset -= aligned_size as i64;
            access
        } else {
            assert!(
                size <= frame::WORD_SIZE as usize,
                "local is too large to fit in a register"
            );
            Access {
                ty: AccessType::InReg(tmp_generator.new_tmp()),
                size,
            }
        }
    }

    /// For in-register temporaries, simply returns the temporary. For in-memory access, returns the
    /// pointer to that memory.
    pub fn expr(access: Access, fp_expr: &ir::Expr) -> ir::Expr {
        match access.ty {
            AccessType::InReg(tmp) => ir::Expr::Tmp(tmp),
            AccessType::InFrame(offset) => {
                assert!(offset <= 0);
                fp_expr.pointer_offset(offset)
            }
        }
    }

    pub fn external_call(fn_name: impl Into<String>, args: Vec<ir::Expr>) -> ir::Expr {
        ir::Expr::Call(Box::new(ir::Expr::Label(Label(fn_name.into()))), args)
    }

    // TODO: Dedup literals.
    pub fn string(tmp_generator: &TmpGenerator, string: impl Into<String>) -> (ir::Expr, Fragment) {
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
