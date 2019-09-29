use crate::{ir, tmp::Label};

#[derive(Debug, Clone)]
pub enum Fragment {
    Fn(FnFragment),
    String(StringFragment),
}

#[derive(Debug, Clone)]
pub struct FnFragment {
    pub body: ir::Stmt,
    pub label: Label,
}

#[derive(Debug, Clone)]
pub struct StringFragment {
    pub label: Label,
    pub string: String,
}
