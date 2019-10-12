use crate::{
    frame::Frame,
    tmp::{Label, TmpGenerator},
    translate::Access,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Level {
    pub parent_label: Option<Label>,
    pub frame: Frame,
}

impl Level {
    pub fn new(
        tmp_generator: &TmpGenerator,
        parent_label: Option<Label>,
        name: impl Into<String>,
        formals: &[(usize, bool)],
    ) -> Self {
        // Add static link
        let mut frame_formals = vec![(0, false); formals.len() + 1];
        frame_formals[0] = (std::mem::size_of::<u64>(), true);
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
                name: "top".to_owned(),
                label: Label::top(),
                formals: vec![],
                locals_offset: 0,
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

    pub fn alloc_local(&mut self, tmp_generator: &TmpGenerator, size: usize, escapes: bool) -> Access {
        Access {
            level_label: self.frame.label.clone(),
            access: self.frame.alloc_local(tmp_generator, size, escapes),
        }
    }

    pub fn label(&self) -> &Label {
        &self.frame.label
    }
}
