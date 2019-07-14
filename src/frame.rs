use crate::tmp::{Label, Tmp, TmpGenerator};

const WORD_SIZE: i32 = 8;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Access {
    InFrame(i32),
    InReg(Tmp),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub label: Label,
    pub formals: Vec<Access>,
    pub n_locals: usize,
}

impl Frame {
    pub fn new(tmp_generator: &mut TmpGenerator, name: String, formals: &[bool]) -> Self {
        let mut offset: i32 = 0;
        let mut accesses = vec![];
        for &formal in formals {
            if formal {
                accesses.push(Access::InFrame(offset));
                offset += WORD_SIZE;
            } else {
                accesses.push(Access::InReg(tmp_generator.new_tmp()));
            }
        }

        Frame {
            label: tmp_generator.new_label(name),
            formals: accesses,
            n_locals: 0,
        }
    }

    pub fn alloc_local(&mut self, tmp_generator: &mut TmpGenerator, escapes: bool) -> Access {
        if escapes {
            self.n_locals += 1;
            Access::InFrame(-WORD_SIZE * self.n_locals as i32)
        } else {
            Access::InReg(tmp_generator.new_tmp())
        }
    }
}
