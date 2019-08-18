use lazy_static::lazy_static;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tmp(pub usize);

impl std::fmt::Display for Tmp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(pub usize);

impl Label {
    pub fn top() -> Self {
        Label(0)
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TmpGenerator {
    current_label_count: usize,
    current_tmp_count: usize,
}

impl Default for TmpGenerator {
    fn default() -> Self {
        TmpGenerator {
            current_tmp_count: ReservedTmps::Count as usize,
            // 0 is reserved for top-level label
            current_label_count: 1,
        }
    }
}

impl TmpGenerator {
    pub fn new_tmp(&mut self) -> Tmp {
        let tmp_count = self.current_tmp_count;
        self.current_tmp_count += 1;
        Tmp(tmp_count)
    }

    pub fn new_label(&mut self) -> Label {
        let label_count = self.current_label_count;
        self.current_label_count += 1;
        Label(label_count)
    }
}
