use lazy_static::lazy_static;
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Copy, Clone)]
pub enum ReservedTmps {
    Fp = 0,
    Sp,
    Count,
}

lazy_static! {
    pub static ref FP: Tmp = Tmp(ReservedTmps::Fp as usize);
    pub static ref SP: Tmp = Tmp(ReservedTmps::Sp as usize);
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tmp(pub usize);

impl std::fmt::Debug for Tmp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

impl std::fmt::Display for Tmp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "t{}", self.0)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Label(pub String);

impl Label {
    pub fn top() -> Self {
        Label("top".to_owned())
    }
}

impl std::fmt::Debug for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct _TmpGenerator {
    current_label_count: usize,
    current_tmp_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TmpGenerator(Rc<RefCell<_TmpGenerator>>);

impl Default for TmpGenerator {
    fn default() -> Self {
        TmpGenerator(Rc::new(RefCell::new(_TmpGenerator {
            current_tmp_count: ReservedTmps::Count as usize,
            // 0 is reserved for top-level label
            current_label_count: 1,
        })))
    }
}

impl TmpGenerator {
    pub fn new_tmp(&self) -> Tmp {
        let mut tmp_generator = self.0.borrow_mut();
        let tmp_count = tmp_generator.current_tmp_count;
        tmp_generator.current_tmp_count += 1;
        Tmp(tmp_count)
    }

    pub fn new_label(&self) -> Label {
        let mut tmp_generator = self.0.borrow_mut();
        let label_count = tmp_generator.current_label_count;
        tmp_generator.current_label_count += 1;
        Label(format!("L{}", label_count))
    }

    pub fn new_named_label(&self, name: &str) -> Label {
        let mut tmp_generator = self.0.borrow_mut();
        let label_count = tmp_generator.current_label_count;
        tmp_generator.current_label_count += 1;
        Label(format!("L{}{}", name, label_count))
    }
}
