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
    pub parent_label: Option<Label>,
    frame: Frame,
}

impl Level {
    pub fn new(tmp_generator: &mut TmpGenerator, parent_label: Option<Label>, name: String, formals: &[bool]) -> Self {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{frame, tmp::TmpGenerator};

    #[test]
    fn test_add_static_link() {
        let mut tmp_generator = TmpGenerator::default();
        let level = Level::new(&mut tmp_generator, Some(Label::top()), "f".to_owned(), &[]);

        assert_eq!(level.frame.formals.len(), 1);
        assert_eq!(level.frame.formals[0], frame::Access::InFrame(0));
    }
}
