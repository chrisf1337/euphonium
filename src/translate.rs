use crate::{frame, ir, tmp::Label, ty::TypeInfo};

mod expr;
mod level;

pub use expr::*;
pub use level::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Access {
    pub level_label: Label,
    pub access: frame::Access,
}

/// `src` and `dst` are pointers to regions of memory. This must only be used for frame-allocated
/// values.
fn memcpy(dst: ir::Expr, src: ir::Expr, size: usize) -> ir::Stmt {
    assert_eq!(
        size % frame::WORD_SIZE as usize,
        0,
        "memory size must be a multiple of {} bytes",
        frame::WORD_SIZE
    );
    let words = size / (frame::WORD_SIZE as usize);

    let mut stmts = vec![];
    for i in 0..words {
        stmts.push(ir::Stmt::Move(
            ir::Expr::Mem(
                Box::new(ir::Expr::BinOp(
                    Box::new(dst.clone()),
                    ir::BinOp::Sub,
                    Box::new(ir::Expr::Const(i as i64 * frame::WORD_SIZE)),
                )),
                frame::WORD_SIZE as usize,
            ),
            ir::Expr::Mem(
                Box::new(ir::Expr::BinOp(
                    Box::new(src.clone()),
                    ir::BinOp::Sub,
                    Box::new(ir::Expr::Const(i as i64 * frame::WORD_SIZE)),
                )),
                frame::WORD_SIZE as usize,
            ),
        ));
    }
    ir::Stmt::seq(stmts)
}

/// `dst` is a pointer to a region of memory. `src` is either also a pointer to a region of memory,
/// if the assigned value is frame-allocated, or simply a value, if the value is not.
pub fn copy(dst: ir::Expr, src: ir::Expr, ty: &TypeInfo) -> ir::Stmt {
    if ty.is_frame_allocated() {
        memcpy(dst, src, ty.size())
    } else {
        ir::Stmt::Move(ir::Expr::Mem(Box::new(dst), ty.size()), src)
    }
}
