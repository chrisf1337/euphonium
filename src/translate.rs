use crate::{frame, tmp::Label};

mod expr;
mod level;

pub use expr::*;
pub use level::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Access {
    pub level_label: Label,
    pub access: frame::Access,
}
