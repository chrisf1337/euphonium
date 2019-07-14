#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tmp(String);

impl std::fmt::Display for Tmp {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label(pub String);

impl Label {
    pub fn top() -> Self {
        Label("top".to_owned())
    }
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct TmpGenerator {
    current_label_count: usize,
    current_tmp_count: usize,
}

impl TmpGenerator {
    pub fn new_tmp(&mut self) -> Tmp {
        let tmp_count = self.current_tmp_count;
        self.current_tmp_count += 1;
        Tmp(format!("t{}", tmp_count))
    }

    pub fn new_label(&mut self, s: String) -> Label {
        let label_count = self.current_label_count;
        self.current_label_count += 1;
        Label(format!("L{}{}", s, label_count))
    }
}
