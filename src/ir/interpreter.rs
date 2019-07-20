use crate::{frame, ir, tmp::Tmp};
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

const MEMORY_SIZE: usize = 1024;

pub struct Interpreter {
    tmps: HashMap<Tmp, i64>,
    memory: [u8; MEMORY_SIZE],
    fp: u64,
    sp: u64,
}

impl Interpreter {
    pub fn new() -> Interpreter {
        Interpreter {
            tmps: HashMap::new(),
            memory: [0; MEMORY_SIZE],
            fp: MEMORY_SIZE as u64 - 1,
            sp: MEMORY_SIZE as u64 - 1,
        }
    }

    pub fn interpret_expr_rvalue(&self, expr: &ir::Expr) -> i64 {
        match expr {
            ir::Expr::Const(c) => *c,
            ir::Expr::Tmp(tmp) => self.tmps[tmp],
            ir::Expr::BinOp(l, op, r) => match op {
                ir::BinOp::Add => self.interpret_expr_rvalue(l) + self.interpret_expr_rvalue(r),
                ir::BinOp::Sub => self.interpret_expr_rvalue(l) - self.interpret_expr_rvalue(r),
                ir::BinOp::Mul => self.interpret_expr_rvalue(l) * self.interpret_expr_rvalue(r),
                ir::BinOp::Div => self.interpret_expr_rvalue(l) / self.interpret_expr_rvalue(r),
                _ => unimplemented!(),
            },
            ir::Expr::Mem(expr) => {
                let index = self.interpret_expr_rvalue(expr) as usize;
                self.read_i64(index)
            }
            _ => unimplemented!(),
        }
    }

    pub fn write_u64(&mut self, u: u64, index: usize) {
        let bytes = u.to_le_bytes();
        self.memory[index..index + 8].copy_from_slice(&bytes);
    }

    pub fn read_u64(&self, index: usize) -> u64 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.memory[index..index + 8]);
        u64::from_le_bytes(bytes)
    }

    pub fn write_u64s(&mut self, us: &[u64], mut index: usize) {
        for &u in us {
            self.write_u64(u, index);
            index += 8;
        }
    }

    pub fn read_u64s(&self, index: usize, n: usize) -> Vec<u64> {
        let mut bytes = vec![];
        for i in (index..index + n * 8).step_by(8) {
            bytes.push(self.read_u64(i));
        }
        bytes
    }

    pub fn write_i64(&mut self, i: i64, index: usize) {
        let bytes = i.to_le_bytes();
        self.memory[index..index + 8].copy_from_slice(&bytes);
    }

    pub fn read_i64(&self, index: usize) -> i64 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.memory[index..index + 8]);
        i64::from_le_bytes(bytes)
    }

    pub fn write_i64s(&mut self, is: &[i64], mut index: usize) {
        for &i in is {
            self.write_i64(i, index);
            index += 8;
        }
    }

    pub fn read_i64s(&self, index: usize, n: usize) -> Vec<i64> {
        let mut bytes = vec![];
        for i in (index..index + n * 8).step_by(8) {
            bytes.push(self.read_i64(i));
        }
        bytes
    }

    pub fn dump_to_file(&self) {
        let path = match env::var("CARGO_MANIFEST_DIR") {
            Ok(dir) => {
                let dir_path = Path::new(&dir);
                dir_path.join("scratch/dump.bin")
            }
            Err(_) => {
                // Default to placing in the current directory
                let mut path = PathBuf::new();
                path.push("dump.bin");
                path
            }
        };
        let mut file = File::create(path).expect("failed to create file");
        file.write_all(&self.memory).expect("failed to write to file");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        frame,
        tmp::{Label, TmpGenerator},
        translate::{translate_simple_var, Level},
        utils::u64ext::U64Ext,
    };
    use maplit::hashmap;
    use std::convert::TryInto;

    #[test]
    fn test_read_write() {
        let mut interpreter = Interpreter::new();
        interpreter.write_u64(0x12345, 0);
        assert_eq!(interpreter.read_u64(0), 0x12345);

        interpreter.write_u64s(&[0x1, 0x2, 0x3, 0x4], 0x123);
        assert_eq!(interpreter.read_u64s(0x123, 4), vec![0x1, 0x2, 0x3, 0x4]);

        interpreter.write_i64(-0x1, 0);
        assert_eq!(interpreter.read_i64(0), -0x1);

        interpreter.write_i64s(&[-0x1, -0x2, 0x3, 0x4], 0x123);
        assert_eq!(interpreter.read_i64s(0x123, 4), vec![-0x1, -0x2, 0x3, 0x4]);
    }

    #[test]
    fn test_deref() {
        let mut interpreter = Interpreter::new();
        interpreter.write_u64(0x54321, 0x234);
        interpreter.write_u64(0x234, 0x123);
        assert_eq!(
            interpreter.interpret_expr_rvalue(&ir::Expr::Mem(Box::new(ir::Expr::Const(0x123)))),
            0x234
        );
        assert_eq!(
            interpreter.interpret_expr_rvalue(&ir::Expr::Mem(Box::new(ir::Expr::Mem(Box::new(ir::Expr::Const(
                0x123
            )))))),
            0x54321
        );
    }

    #[test]
    fn test_simple_var() {
        let mut tmp_generator = TmpGenerator::default();
        let mut interpreter = Interpreter::new();
        interpreter.tmps.insert(frame::FP.clone(), interpreter.fp as i64);
        interpreter.tmps.insert(frame::SP.clone(), interpreter.fp as i64);
        let mut level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let label = level.label().clone();
        let local = level.alloc_local(&mut tmp_generator, true);
        let levels = hashmap! {
            Label::top() => Level::top(),
            label.clone() => level,
        };
        let expr = translate_simple_var(&levels, &local, &label);

        let addr = {
            if let frame::Access::InFrame(offset) = &local.access {
                interpreter.fp.iadd(*offset)
            } else {
                unreachable!()
            }
        }
        .try_into()
        .unwrap();
        interpreter.write_u64(0x1234, addr);

        assert_eq!(
            interpreter.interpret_expr_rvalue(&expr.unwrap_expr(&mut tmp_generator)),
            0x1234
        );
    }
}
