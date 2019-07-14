use crate::{
    frame::{self, Frame},
    tmp::{Label, TmpGenerator},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Access {
    pub level_label: Label,
    pub access: frame::Access,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Level {
    parent_label: Option<Label>,
    frame: Frame,
}

impl Level {
    pub fn new(tmp_generator: &mut TmpGenerator, parent_label: Option<Label>, name: String, formals: &[bool]) -> Self {
        Self {
            parent_label,
            frame: Frame::new(tmp_generator, name, formals),
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
        self.frame
            .formals
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
