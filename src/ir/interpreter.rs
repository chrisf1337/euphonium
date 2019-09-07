use crate::{
    frame, ir,
    tmp::{self, Label, Tmp, TmpGenerator},
    translate::{Access, Expr, Level},
};
use maplit::hashmap;
use std::{
    collections::HashMap,
    env,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

const MEMORY_SIZE: usize = 1024;

pub struct Interpreter {
    stmts: Vec<ir::Stmt>,
    tmps: HashMap<Tmp, u64>,
    memory: [u8; MEMORY_SIZE],
    ip: usize,
    label_table: HashMap<Label, usize>,
}

impl Interpreter {
    fn new(expr: Expr) -> Interpreter {
        let mut tmp_generator = TmpGenerator::default();
        let stmts = expr.unwrap_stmt(&mut tmp_generator).flatten();
        let mut label_table = HashMap::new();
        for (i, stmt) in stmts.iter().enumerate() {
            if let ir::Stmt::Label(label) = stmt {
                label_table.insert(label.clone(), i);
            }
        }
        let fp = MEMORY_SIZE as u64;
        let sp = MEMORY_SIZE as u64;
        let tmps = hashmap! {
            tmp::FP.clone() => fp,
            tmp::SP.clone() => sp,
        };
        Interpreter {
            stmts,
            tmps,
            memory: [0; MEMORY_SIZE],
            ip: 0,
            label_table,
        }
    }

    fn jump(&mut self, label: &Label) {
        self.ip = self.label_table[label];
    }

    fn step(&mut self) {
        let stmt = self.stmts[self.ip].clone();
        self.interpret_stmt(&stmt);
        self.ip += 1;
    }

    fn run(&mut self) {
        while self.ip < self.stmts.len() {
            self.step();
        }
    }

    fn sp(&self) -> u64 {
        self.tmps[&tmp::SP]
    }

    fn sp_mut(&mut self) -> &mut u64 {
        self.tmps.get_mut(&tmp::SP).unwrap()
    }

    fn fp(&self) -> u64 {
        self.tmps[&tmp::FP]
    }

    fn fp_mut(&mut self) -> &mut u64 {
        self.tmps.get_mut(&tmp::FP).unwrap()
    }

    fn interpret_expr_as_rvalue(&mut self, expr: &ir::Expr) -> u64 {
        match expr {
            ir::Expr::Const(c) => unsafe { std::mem::transmute(*c) },
            ir::Expr::Tmp(tmp) => self.tmps[tmp],
            ir::Expr::BinOp(l, op, r) => {
                let l = self.interpret_expr_as_rvalue(l);
                let r = self.interpret_expr_as_rvalue(r);
                match op {
                    ir::BinOp::Add => l.wrapping_add(r),
                    ir::BinOp::Sub => l.wrapping_sub(r),
                    ir::BinOp::Mul => l.wrapping_mul(r),
                    ir::BinOp::Div => l.wrapping_div(r),
                    _ => unimplemented!(),
                }
            }
            ir::Expr::Mem(expr) => {
                let addr = self.interpret_expr_as_rvalue(expr);
                self.read_u64(addr)
            }
            ir::Expr::Seq(..) => panic!("cannot interpret {:?}", expr),
            _ => unimplemented!(),
        }
    }

    fn interpret_stmt(&mut self, stmt: &ir::Stmt) {
        match stmt {
            ir::Stmt::Move(dst, src) => {
                let value = self.interpret_expr_as_rvalue(src);
                match dst {
                    ir::Expr::Tmp(tmp) => {
                        let tmp = self.tmps.get_mut(&tmp).unwrap();
                        *tmp = value;
                    }
                    ir::Expr::Mem(addr_expr) => {
                        let addr = self.interpret_expr_as_rvalue(addr_expr);
                        self.write_u64(value, addr);
                    }
                    _ => panic!("cannot move to non Tmp or Mem"),
                }
            }
            ir::Stmt::CJump(l, op, r, t_label, f_label) => {
                let l = self.interpret_expr_as_rvalue(l);
                let r = self.interpret_expr_as_rvalue(r);
                match op {
                    ir::CompareOp::Eq => {
                        if l == r {
                            self.jump(t_label);
                        } else {
                            self.jump(f_label);
                        }
                    }
                    _ => unimplemented!(),
                }
            }
            ir::Stmt::Label(_) => (),
            ir::Stmt::Jump(expr, _) => {
                if let ir::Expr::Label(label) = expr {
                    self.jump(label)
                } else {
                    panic!("unexpected jump target: {:?}", expr)
                }
            }
            _ => unimplemented!("{:?}", stmt),
        }
    }

    fn write_u64(&mut self, u: u64, addr: u64) {
        let bytes = u.to_le_bytes();
        let addr = addr as usize;
        self.memory[addr..addr + frame::WORD_SIZE as usize].copy_from_slice(&bytes);
    }

    fn read_u64(&self, addr: u64) -> u64 {
        let mut bytes = [0u8; frame::WORD_SIZE as usize];
        let addr = addr as usize;
        bytes.copy_from_slice(&self.memory[addr..addr + frame::WORD_SIZE as usize]);
        u64::from_le_bytes(bytes)
    }

    fn write_u64s(&mut self, us: &[u64], mut addr: u64) {
        for &u in us {
            self.write_u64(u, addr);
            addr += std::mem::size_of::<u64>() as u64;
        }
    }

    fn read_u64s(&self, addr: u64, n: usize) -> Vec<u64> {
        let mut bytes = vec![];
        for i in (addr..addr + (n * std::mem::size_of::<u64>()) as u64).step_by(std::mem::size_of::<u64>()) {
            bytes.push(self.read_u64(i));
        }
        bytes
    }

    fn write_i64(&mut self, i: i64, addr: u64) {
        let bytes = i.to_le_bytes();
        let addr = addr as usize;
        self.memory[addr..addr + frame::WORD_SIZE as usize].copy_from_slice(&bytes);
    }

    fn read_i64(&self, addr: u64) -> i64 {
        let mut bytes = [0u8; frame::WORD_SIZE as usize];
        let addr = addr as usize;
        bytes.copy_from_slice(&self.memory[addr..addr + frame::WORD_SIZE as usize]);
        i64::from_le_bytes(bytes)
    }

    fn write_i64s(&mut self, is: &[i64], mut addr: u64) {
        for &i in is {
            self.write_i64(i, addr);
            addr += std::mem::size_of::<i64>() as u64;
        }
    }

    fn read_i64s(&self, addr: u64, n: usize) -> Vec<i64> {
        let mut bytes = vec![];
        for i in (addr..addr + (n * std::mem::size_of::<u64>()) as u64).step_by(std::mem::size_of::<u64>()) {
            bytes.push(self.read_i64(i));
        }
        bytes
    }

    fn dump_to_file(&self, name: &str) {
        let path = match env::var("CARGO_MANIFEST_DIR") {
            Ok(dir) => {
                let dir_path = Path::new(&dir);
                dir_path.join(format!("scratch/{}.bin", name))
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

    fn alloc_local(
        &mut self,
        tmp_generator: &mut TmpGenerator,
        level: &mut Level,
        escapes: bool,
    ) -> (Access, Option<u64>) {
        let local = level.alloc_local(tmp_generator, escapes);
        let addr = if escapes {
            *self.sp_mut() -= frame::WORD_SIZE as u64;
            Some(self.sp())
        } else {
            None
        };
        (local, addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast, ir,
        tmp::{Label, TmpGenerator},
        translate::{self, Expr},
        typecheck::Env,
    };
    use maplit::hashmap;

    #[test]
    fn test_read_write() {
        let mut interpreter = Interpreter::new(Expr::Stmt(ir::Stmt::Expr(ir::Expr::Const(0))));
        interpreter.write_u64(0x12345, 0);
        assert_eq!(interpreter.read_u64(0), 0x12345);

        interpreter.write_u64s(&[0x1, 0x2, 0x3, 0x4], 0x200);
        assert_eq!(interpreter.read_u64s(0x200, 4), vec![0x1, 0x2, 0x3, 0x4]);

        interpreter.write_i64(-0x1, 0);
        assert_eq!(interpreter.read_i64(0), -0x1);

        interpreter.write_i64s(&[-0x1, -0x2, 0x3, 0x4], 0x200);
        assert_eq!(interpreter.read_i64s(0x200, 4), vec![-0x1, -0x2, 0x3, 0x4]);
    }

    #[test]
    fn test_deref() {
        let mut interpreter = Interpreter::new(Expr::Stmt(ir::Stmt::Expr(ir::Expr::Const(0))));
        interpreter.write_u64(0x54321, 0x234);
        interpreter.write_u64(0x234, 0x123);
        assert_eq!(
            interpreter.interpret_expr_as_rvalue(&ir::Expr::Mem(Box::new(ir::Expr::Const(0x123)))),
            0x234
        );
        assert_eq!(
            interpreter.interpret_expr_as_rvalue(&ir::Expr::Mem(Box::new(ir::Expr::Mem(Box::new(ir::Expr::Const(
                0x123
            )))))),
            0x54321
        );
    }

    #[test]
    fn test_translate_simple_var() {
        let mut tmp_generator = TmpGenerator::default();
        let mut interpreter = Interpreter::new(Expr::Stmt(ir::Stmt::Expr(ir::Expr::Const(0))));
        let mut level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let (local, addr) = interpreter.alloc_local(&mut tmp_generator, &mut level, true);
        let label = level.label();
        let levels = hashmap! {
            Label::top() => Level::top(),
            label.clone() => level.clone(),
        };
        let expr = Env::translate_simple_var(&levels, local, label).clone();
        interpreter.write_u64(0x1234, addr.unwrap());

        assert_eq!(
            interpreter.interpret_expr_as_rvalue(&expr.unwrap_expr(&mut tmp_generator)),
            0x1234
        );
    }

    #[test]
    fn test_translate_pointer_offset() {
        let mut tmp_generator = TmpGenerator::default();
        let mut interpreter = Interpreter::new(Expr::Stmt(ir::Stmt::Expr(ir::Expr::Const(0))));
        let mut level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let (local, addr) = interpreter.alloc_local(&mut tmp_generator, &mut level, true);
        let label = level.label();
        let levels = hashmap! {
            Label::top() => Level::top(),
            label.clone() => level.clone(),
        };

        // Array located at 0x100. Write address into local.
        interpreter.write_u64(0x100, addr.unwrap());
        interpreter.write_u64s(&[1, 2, 3], 0x100);

        let (expr1, expr2, expr3) = {
            let array_expr = Env::translate_simple_var(&levels, local, label);
            (
                Env::translate_pointer_offset(
                    &mut tmp_generator,
                    &array_expr,
                    &translate::Expr::Expr(ir::Expr::Const(0)),
                ),
                Env::translate_pointer_offset(
                    &mut tmp_generator,
                    &array_expr,
                    &translate::Expr::Expr(ir::Expr::Const(1)),
                ),
                Env::translate_pointer_offset(
                    &mut tmp_generator,
                    &array_expr,
                    &translate::Expr::Expr(ir::Expr::Const(2)),
                ),
            )
        };

        assert_eq!(
            interpreter.interpret_expr_as_rvalue(&expr1.unwrap_expr(&mut tmp_generator)),
            1
        );
        assert_eq!(
            interpreter.interpret_expr_as_rvalue(&expr2.unwrap_expr(&mut tmp_generator)),
            2
        );
        assert_eq!(
            interpreter.interpret_expr_as_rvalue(&expr3.unwrap_expr(&mut tmp_generator)),
            3
        );
    }

    #[test]
    fn test_move() {
        let mut interpreter = Interpreter::new(Expr::Stmt(ir::Stmt::Expr(ir::Expr::Const(0))));
        *interpreter.sp_mut() -= frame::WORD_SIZE as u64;
        interpreter.interpret_stmt(&ir::Stmt::Move(
            ir::Expr::Mem(Box::new(ir::Expr::Tmp(*tmp::SP))),
            ir::Expr::Const(0x200),
        ));
        assert_eq!(interpreter.read_u64(interpreter.tmps[&tmp::SP]), 0x200);
    }

    #[test]
    fn test_cjump() {
        let mut tmp_generator = TmpGenerator::default();
        let t_label = tmp_generator.new_label();
        let f_label = tmp_generator.new_label();
        let join_label = tmp_generator.new_label();
        let stmts = vec![
            ir::Stmt::CJump(
                ir::Expr::Const(0),
                ir::CompareOp::Eq,
                ir::Expr::Const(0),
                t_label.clone(),
                f_label.clone(),
            ),
            ir::Stmt::Label(t_label),
            ir::Stmt::Move(ir::Expr::Mem(Box::new(ir::Expr::Const(0x200))), ir::Expr::Const(123)),
            ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
            ir::Stmt::Label(f_label),
            ir::Stmt::Move(ir::Expr::Mem(Box::new(ir::Expr::Const(0x200))), ir::Expr::Const(456)),
            ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
            ir::Stmt::Label(join_label),
        ];
        let mut interpreter = Interpreter::new(Expr::Stmt(ir::Stmt::seq(stmts)));
        interpreter.run();

        assert_eq!(interpreter.read_u64(0x200), 123);
    }

    #[test]
    fn test_if_ir() {
        let mut tmp_generator = TmpGenerator::default();
        let mut env = Env::default();
        let let_expr = zspan!(ast::ExprType::Let(Box::new(zspan!(ast::Let {
            pattern: zspan!(ast::Pattern::String("a".to_owned())),
            immutable: zspan!(false),
            ty: None,
            expr: zspan!(ast::ExprType::Number(0)),
        }))));
        env.typecheck_expr_mut(&mut tmp_generator, &Label::top(), &let_expr)
            .expect("typecheck_let failed");
        println!("{:#?}", env);
        // let if_expr = zspan!(ast::If {
        //     cond: zspan!(ast::ExprType::BoolLiteral(true)),
        //     then_expr: zspan!(ast::ExprType::)
        // });
    }
}
