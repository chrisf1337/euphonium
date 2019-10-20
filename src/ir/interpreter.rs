use crate::{
    fragment::{Fragment, StringFragment},
    frame, ir,
    tmp::{self, Label, Tmp, TmpGenerator},
    translate::{Access, Expr, Level},
};
use lazy_static::lazy_static;
use maplit::{hashmap, hashset};
use std::{
    collections::{HashMap, HashSet},
    env,
    fs::File,
    io::Write,
    path::{Path, PathBuf},
};

const MEMORY_SIZE: usize = 1024;

lazy_static! {
    static ref RUNTIME_FNS: HashSet<String> = hashset! {
        "__malloc".to_owned(),
        "__panic".to_owned(),
        "__strcmp".to_owned(),
    };
}

#[derive(Clone)]
pub struct Interpreter {
    stmts: Vec<ir::Stmt>,
    tmps: HashMap<Tmp, i64>,
    memory: [u8; MEMORY_SIZE],
    ip: usize,
    /// Map of label to instruction index in `stmts`.
    jump_table: HashMap<Label, usize>,
    /// Map of label to address in `memory`.
    string_table: HashMap<Label, i64>,
    /// Heap pointer
    hp: u64,
    panicked: bool,
}

impl Default for Interpreter {
    fn default() -> Interpreter {
        let fp = MEMORY_SIZE as i64;
        let sp = MEMORY_SIZE as i64;
        let tmps = hashmap! {
            tmp::FP.clone() => fp,
            tmp::SP.clone() => sp,
        };
        Interpreter {
            stmts: vec![],
            tmps,
            memory: [0; MEMORY_SIZE],
            ip: 0,
            jump_table: HashMap::new(),
            string_table: HashMap::new(),
            hp: 0,
            panicked: false,
        }
    }
}

impl Interpreter {
    fn jump(&mut self, label: &Label) {
        self.ip = self.jump_table[label];
    }

    fn step(&mut self) -> Option<i64> {
        let stmt = self.stmts[self.ip].clone();
        let val = self.interpret_stmt(&stmt);
        self.ip += 1;
        val
    }

    fn run(&mut self) -> Option<i64> {
        let mut val: Option<i64> = None;
        while self.ip < self.stmts.len() {
            val = self.step();
            if self.panicked {
                return None;
            }
        }
        val
    }

    fn run_expr(&mut self, tmp_generator: &TmpGenerator, expr: Expr) -> Option<i64> {
        let stmts = expr.unwrap_stmt(tmp_generator).flatten();
        let mut jump_table = HashMap::new();
        for (i, stmt) in stmts.iter().enumerate() {
            if let ir::Stmt::Label(label) = stmt {
                jump_table.insert(label.clone(), i);
            }
        }
        self.stmts = stmts;
        self.jump_table = jump_table;
        self.ip = 0;
        self.panicked = false;
        self.run()
    }

    fn sp(&self) -> i64 {
        self.tmps[&tmp::SP]
    }

    fn sp_mut(&mut self) -> &mut i64 {
        self.tmp_mut(*tmp::SP)
    }

    fn fp(&self) -> i64 {
        self.tmps[&tmp::FP]
    }

    fn fp_mut(&mut self) -> &mut i64 {
        self.tmp_mut(*tmp::FP)
    }

    fn tmp_mut(&mut self, tmp: Tmp) -> &mut i64 {
        self.tmps.entry(tmp).or_default()
    }

    fn set_string_table(&mut self, fragments: &[Fragment]) {
        for fragment in fragments {
            if let Fragment::String(StringFragment { label, string }) = fragment {
                self.string_table.insert(label.clone(), self.hp as i64);

                let mut p = self.hp as usize;
                let string = string.clone().into_bytes();
                self.write_u64(string.len() as u64, p);
                p += std::mem::size_of::<u64>();
                self.write_u8s(&string, p);
                p += string.len();
                let padding = vec![0; std::mem::size_of::<u64>() - (string.len() % std::mem::size_of::<u64>())];
                self.write_u8s(&padding, p);
                p += padding.len();
                self.hp = p as u64;
            }
        }
    }

    fn interpret_expr_as_value(&mut self, expr: &ir::Expr) -> i64 {
        match expr {
            ir::Expr::Const(c) => *c,
            ir::Expr::Tmp(tmp) => *self.tmp_mut(*tmp),
            ir::Expr::BinOp(l, op, r) => {
                let l = self.interpret_expr_as_value(l);
                let r = self.interpret_expr_as_value(r);
                match op {
                    ir::BinOp::Add => l + r,
                    ir::BinOp::Sub => l - r,
                    ir::BinOp::Mul => l * r,
                    ir::BinOp::Div => l / r,
                    ir::BinOp::Mod => l % r,
                    _ => unimplemented!("{:?}", op),
                }
            }
            ir::Expr::Mem(expr, ..) => {
                // FIXME: Use size?
                let addr = self.interpret_expr_as_value(expr);
                self.read_i64(addr as usize)
            }
            ir::Expr::Seq(stmt, expr) => {
                self.interpret_stmt(stmt);
                self.interpret_expr_as_value(expr)
            }
            ir::Expr::Call(label, args) => {
                let name = if let ir::Expr::Label(Label(label)) = label.as_ref() {
                    label
                } else {
                    unreachable!("expected {:?} to be a Label", label)
                };
                if RUNTIME_FNS.contains(name) {
                    self.runtime_call(name, args)
                } else {
                    unimplemented!("{}", name)
                }
            }
            ir::Expr::Label(label) => self.string_table[label],
        }
    }

    fn interpret_stmt(&mut self, stmt: &ir::Stmt) -> Option<i64> {
        match stmt {
            ir::Stmt::Move(dst, src) => {
                let value = self.interpret_expr_as_value(src);
                match dst {
                    ir::Expr::Tmp(tmp) => {
                        let tmp = self.tmp_mut(*tmp);
                        *tmp = value;
                    }
                    ir::Expr::Mem(addr_expr, size) => {
                        assert!(*size <= frame::WORD_SIZE as usize);
                        let addr = self.interpret_expr_as_value(addr_expr);
                        self.write_i64(value, addr as usize);
                    }
                    _ => panic!("cannot move to non Tmp or Mem"),
                }
                None
            }
            ir::Stmt::CJump(l, op, r, t_label, f_label) => {
                let l: i64 = self.interpret_expr_as_value(l);
                let r: i64 = self.interpret_expr_as_value(r);
                match op {
                    ir::CompareOp::Eq => {
                        if l == r {
                            self.jump(t_label);
                        } else {
                            self.jump(f_label);
                        }
                    }
                    ir::CompareOp::Ge => {
                        if l >= r {
                            self.jump(t_label);
                        } else {
                            self.jump(f_label);
                        }
                    }
                    ir::CompareOp::Gt => {
                        if l > r {
                            self.jump(t_label);
                        } else {
                            self.jump(f_label);
                        }
                    }
                    ir::CompareOp::Lt => {
                        if l < r {
                            self.jump(t_label);
                        } else {
                            self.jump(f_label);
                        }
                    }
                    _ => unimplemented!("{:?}", op),
                }
                None
            }
            ir::Stmt::Label(_) => None,
            ir::Stmt::Jump(expr, _) => {
                if let ir::Expr::Label(label) = expr {
                    self.jump(label);
                    None
                } else {
                    panic!("unexpected jump target: {:?}", expr);
                }
            }
            ir::Stmt::Expr(expr) => Some(self.interpret_expr_as_value(expr)),
            ir::Stmt::Seq(stmt1, stmt2) => {
                self.interpret_stmt(stmt1);
                self.interpret_stmt(stmt2)
            }
        }
    }

    fn write_u8(&mut self, u: u8, addr: usize) {
        self.memory[addr] = u;
    }

    fn read_u8(&self, addr: usize) -> u8 {
        self.memory[addr]
    }

    fn write_u8s(&mut self, us: &[u8], addr: usize) {
        self.memory[addr..addr + us.len()].copy_from_slice(us)
    }

    fn read_u8s(&self, addr: usize, n: usize) -> Vec<u8> {
        self.memory[addr..addr + n].to_vec()
    }

    fn write_u64(&mut self, u: u64, addr: usize) {
        let bytes = u.to_le_bytes();
        self.write_u8s(&bytes, addr);
    }

    fn read_u64(&self, addr: usize) -> u64 {
        let mut bytes = [0u8; frame::WORD_SIZE as usize];
        bytes.copy_from_slice(&self.read_u8s(addr, std::mem::size_of::<u64>()));
        u64::from_le_bytes(bytes)
    }

    fn write_u64s(&mut self, us: &[u64], mut addr: usize) {
        for &u in us {
            self.write_u64(u, addr);
            addr += std::mem::size_of::<u64>();
        }
    }

    fn read_u64s(&self, addr: usize, n: usize) -> Vec<u64> {
        let mut bytes = vec![];
        for i in (addr..addr + n * std::mem::size_of::<u64>()).step_by(std::mem::size_of::<u64>()) {
            bytes.push(self.read_u64(i));
        }
        bytes
    }

    fn write_i64(&mut self, i: i64, addr: usize) {
        let bytes = i.to_le_bytes();
        self.write_u8s(&bytes, addr);
    }

    fn read_i64(&self, addr: usize) -> i64 {
        let mut bytes = [0u8; frame::WORD_SIZE as usize];
        bytes.copy_from_slice(&self.read_u8s(addr, std::mem::size_of::<i64>()));
        i64::from_le_bytes(bytes)
    }

    fn write_i64s(&mut self, is: &[i64], mut addr: usize) {
        for &i in is {
            self.write_i64(i, addr);
            addr += std::mem::size_of::<i64>();
        }
    }

    fn read_i64s(&self, addr: usize, n: usize) -> Vec<i64> {
        let mut bytes = vec![];
        for i in (addr..addr + n * std::mem::size_of::<u64>()).step_by(std::mem::size_of::<u64>()) {
            bytes.push(self.read_i64(i));
        }
        bytes
    }

    fn read_str(&self, mut addr: usize) -> String {
        let len = self.read_u64(addr) as usize;
        addr += std::mem::size_of::<u64>();

        let bytes = self.read_u8s(addr, len);
        std::str::from_utf8(&bytes)
            .unwrap_or_else(|_| panic!("invalid string: {:?}", bytes))
            .to_owned()
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
        tmp_generator: &TmpGenerator,
        level: &mut Level,
        size: usize,
        escapes: bool,
    ) -> (Access, Option<i64>) {
        let local = level.alloc_local(tmp_generator, size, escapes);
        let addr = if escapes {
            *self.sp_mut() -= frame::WORD_SIZE;
            Some(self.sp())
        } else {
            None
        };
        (local, addr)
    }

    fn runtime_call(&mut self, name: &str, args: &[ir::Expr]) -> i64 {
        match name {
            "__malloc" => self.__malloc(args),
            "__panic" => self.__panic(args),
            _ => unreachable!("{} is not a runtime function", name),
        }
    }

    fn malloc(&mut self, words: u64) -> i64 {
        let addr = self.hp;
        self.hp += frame::WORD_SIZE as u64 * words;
        addr as i64
    }

    fn __malloc(&mut self, args: &[ir::Expr]) -> i64 {
        let len = self.interpret_expr_as_value(&args[0]);
        self.malloc(len as u64)
    }

    fn __panic(&mut self, _args: &[ir::Expr]) -> i64 {
        self.panicked = true;
        0
    }

    // fn __strcmp(&self, args: &[ir::Expr]) -> u64 {
    //     let l = self.interpret_expr_as_value(&args[0]);
    //     let r = self.interpret_expr_as_value(&args[1]);
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ast, ir, parser,
        tmp::{Label, TmpGenerator},
        translate::Expr,
        typecheck::{Env, TypecheckError},
        utils::{dump_vec, EMPTY_SOURCEMAP},
    };
    use maplit::hashmap;

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum InterpreterTestError {
        ParseErr(Vec<parser::ParseError>),
        TypecheckError(Vec<TypecheckError>),
    }

    impl From<Vec<parser::ParseError>> for InterpreterTestError {
        fn from(errs: Vec<parser::ParseError>) -> Self {
            Self::ParseErr(errs)
        }
    }

    impl From<Vec<TypecheckError>> for InterpreterTestError {
        fn from(errs: Vec<TypecheckError>) -> Self {
            Self::TypecheckError(errs)
        }
    }

    #[test]
    fn read_write() {
        let mut interpreter = Interpreter::default();
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
    fn deref() {
        let mut interpreter = Interpreter::default();
        interpreter.write_u64(0x54321, 0x234);
        interpreter.write_u64(0x234, 0x123);
        assert_eq!(
            interpreter.interpret_expr_as_value(&ir::Expr::Mem(
                Box::new(ir::Expr::Const(0x123)),
                std::mem::size_of::<u64>()
            )),
            0x234
        );
        assert_eq!(
            interpreter.interpret_expr_as_value(&ir::Expr::Mem(
                Box::new(ir::Expr::Mem(
                    Box::new(ir::Expr::Const(0x123)),
                    std::mem::size_of::<u64>()
                )),
                std::mem::size_of::<u64>()
            )),
            0x54321
        );
    }

    #[test]
    fn translate_simple_var() {
        let mut tmp_generator = TmpGenerator::default();
        let mut interpreter = Interpreter::default();
        let mut level = Level::new(&mut tmp_generator, Some(Label::top()), "f", &[]);
        let (local, addr) = interpreter.alloc_local(&mut tmp_generator, &mut level, std::mem::size_of::<u64>(), true);
        let label = level.label();
        let levels = hashmap! {
            Label::top() => Level::top(),
            label.clone() => level.clone(),
        };
        let expr = ir::Expr::Mem(
            Box::new(Env::translate_simple_var(&levels, &local, label)),
            frame::WORD_SIZE as usize,
        )
        .clone();
        interpreter.write_u64(0x1234, addr.unwrap() as usize);

        assert_eq!(interpreter.interpret_expr_as_value(&expr), 0x1234);
    }

    #[test]
    fn r#move() {
        let mut interpreter = Interpreter::default();
        *interpreter.sp_mut() -= frame::WORD_SIZE;
        interpreter.interpret_stmt(&ir::Stmt::Move(
            ir::Expr::Mem(Box::new(ir::Expr::Tmp(*tmp::SP)), std::mem::size_of::<u64>()),
            ir::Expr::Const(0x200),
        ));
        assert_eq!(interpreter.read_u64(interpreter.tmps[&tmp::SP] as usize), 0x200);
    }

    #[test]
    fn cjump() {
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
            ir::Stmt::Move(
                ir::Expr::Mem(Box::new(ir::Expr::Const(0x200)), std::mem::size_of::<u64>()),
                ir::Expr::Const(123),
            ),
            ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
            ir::Stmt::Label(f_label),
            ir::Stmt::Move(
                ir::Expr::Mem(Box::new(ir::Expr::Const(0x200)), std::mem::size_of::<u64>()),
                ir::Expr::Const(456),
            ),
            ir::Stmt::Jump(ir::Expr::Label(join_label.clone()), vec![join_label.clone()]),
            ir::Stmt::Label(join_label),
        ];
        let mut interpreter = Interpreter::default();
        interpreter.run_expr(&mut tmp_generator, Expr::Stmt(ir::Stmt::seq(stmts)));

        assert_eq!(interpreter.read_u64(0x200), 123);
    }

    #[test]
    fn if_ir() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        // let a = 0;
        let let_expr = zspan!(ast::ExprType::Let(Box::new(zspan!(ast::Let {
            pattern: zspan!(ast::Pattern::String("a".to_owned())),
            immutable: zspan!(false),
            ty: None,
            expr: zspan!(ast::ExprType::Number(0)),
        }))));
        let mut stmt = env
            .typecheck_expr_mut(&let_expr)
            .expect("typecheck failed")
            .expr
            .unwrap_stmt(&tmp_generator);

        // if true { a = 1 };
        let if_expr = zspan!(ast::ExprType::If(Box::new(zspan!(ast::If {
            cond: zspan!(ast::ExprType::BoolLiteral(true)),
            then_expr: zspan!(ast::ExprType::Assign(Box::new(zspan!(ast::Assign {
                lval: zspan!(ast::LVal::Simple("a".to_owned())),
                expr: zspan!(ast::ExprType::Number(1)),
            })))),
            else_expr: None,
        }))));
        stmt = stmt.appending(
            env.typecheck_expr(&if_expr)
                .expect("typecheck failed")
                .expr
                .unwrap_stmt(&tmp_generator),
        );

        let mut interpreter = Interpreter::default();
        interpreter.run_expr(&tmp_generator, Expr::Stmt(stmt));
        let addr = interpreter.fp() as i64 - frame::WORD_SIZE;
        assert_eq!(interpreter.read_u64(addr as usize), 1);
    }

    #[test]
    fn if_as_expr_ir() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        // let a = 0;
        let let_expr = zspan!(ast::ExprType::Let(Box::new(zspan!(ast::Let {
            pattern: zspan!(ast::Pattern::String("a".to_owned())),
            immutable: zspan!(false),
            ty: None,
            expr: zspan!(ast::ExprType::Number(0)),
        }))));
        let mut stmt = env
            .typecheck_expr_mut(&let_expr)
            .expect("typecheck failed")
            .expr
            .unwrap_stmt(&tmp_generator);

        // a = if a == 0 { 123 } else { 456 };
        let if_expr = zspan!(ast::ExprType::If(Box::new(zspan!(ast::If {
            cond: zspan!(ast::ExprType::Compare(Box::new(zspan!(ast::Compare {
                l: zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Simple("a".to_owned()))))),
                op: zspan!(ast::CompareOp::Eq),
                r: zspan!(ast::ExprType::Number(0))
            })))),
            then_expr: zspan!(ast::ExprType::Number(123)),
            else_expr: Some(zspan!(ast::ExprType::Number(456))),
        }))));
        let assign_expr = zspan!(ast::ExprType::Assign(Box::new(zspan!(ast::Assign {
            lval: zspan!(ast::LVal::Simple("a".to_owned())),
            expr: if_expr,
        }))));

        stmt = stmt.appending(
            env.typecheck_expr(&assign_expr)
                .expect("typecheck failed")
                .expr
                .unwrap_stmt(&tmp_generator),
        );
        dump_vec(&stmt.flatten());

        let mut interpreter = Interpreter::default();
        interpreter.run_expr(&tmp_generator, Expr::Stmt(stmt));
        let addr = interpreter.fp() as i64 - frame::WORD_SIZE;
        assert_eq!(interpreter.read_u64(addr as usize), 123);
    }

    #[test]
    fn array_subscript() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        // let a = [123; 10];
        let let_expr = zspan!(ast::ExprType::Let(Box::new(zspan!(ast::Let {
            pattern: zspan!(ast::Pattern::String("a".to_owned())),
            immutable: zspan!(false),
            ty: None,
            expr: zspan!(ast::ExprType::Array(Box::new(zspan!(ast::Array {
                initial_value: zspan!(ast::ExprType::Number(123)),
                len: zspan!(ast::ExprType::Number(10)),
            })))),
        }))));
        let mut stmt = env
            .typecheck_expr_mut(&let_expr)
            .expect("typecheck failed")
            .expr
            .unwrap_stmt(&tmp_generator);

        // a[0] = 1;
        let assign_expr = zspan!(ast::ExprType::Assign(Box::new(zspan!(ast::Assign {
            lval: zspan!(ast::LVal::Subscript(
                Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
                zspan!(ast::ExprType::Number(0))
            )),
            expr: zspan!(ast::ExprType::Number(1)),
        }))));
        stmt = stmt.appending(
            env.typecheck_expr_mut(&assign_expr)
                .expect("typecheck failed")
                .expr
                .unwrap_stmt(&tmp_generator),
        );

        // a[3] = 3;
        let assign_expr = zspan!(ast::ExprType::Assign(Box::new(zspan!(ast::Assign {
            lval: zspan!(ast::LVal::Subscript(
                Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
                zspan!(ast::ExprType::Number(3))
            )),
            expr: zspan!(ast::ExprType::Number(3)),
        }))));
        stmt = stmt.appending(
            env.typecheck_expr_mut(&assign_expr)
                .expect("typecheck failed")
                .expr
                .unwrap_stmt(&tmp_generator),
        );

        let mut interpreter = Interpreter::default();
        dump_vec(&stmt.flatten());
        interpreter.run_expr(&tmp_generator, Expr::Stmt(stmt));
        interpreter.dump_to_file("array_subscript");

        {
            let expr = zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Subscript(
                Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
                zspan!(ast::ExprType::Number(0))
            )))));
            let trexpr = env.typecheck_expr(&expr).expect("typecheck").expr;
            let val = interpreter.run_expr(&tmp_generator, trexpr);
            assert_eq!(val, Some(1));
        }

        {
            let expr = zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Subscript(
                Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
                zspan!(ast::ExprType::Number(1))
            )))));
            let trexpr = env.typecheck_expr(&expr).expect("typecheck").expr;
            let val = interpreter.run_expr(&tmp_generator, trexpr);
            assert_eq!(val, Some(123));
        }

        {
            let expr = zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Subscript(
                Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
                zspan!(ast::ExprType::Number(3))
            )))));
            let trexpr = env.typecheck_expr(&expr).expect("typecheck").expr;
            let val = interpreter.run_expr(&tmp_generator, trexpr);
            assert_eq!(val, Some(3));
        }
    }

    #[test]
    fn array_bounds_checking_under() {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        // let a = [0; 10];
        let let_expr = zspan!(ast::ExprType::Let(Box::new(zspan!(ast::Let {
            pattern: zspan!(ast::Pattern::String("a".to_owned())),
            immutable: zspan!(false),
            ty: None,
            expr: zspan!(ast::ExprType::Array(Box::new(zspan!(ast::Array {
                initial_value: zspan!(ast::ExprType::Number(0)),
                len: zspan!(ast::ExprType::Number(10)),
            })))),
        }))));
        let mut stmt = env
            .typecheck_expr_mut(&let_expr)
            .expect("typecheck failed")
            .expr
            .unwrap_stmt(&tmp_generator);

        // a[-1];
        let subscript_expr = zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Subscript(
            Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
            zspan!(ast::ExprType::Neg(Box::new(zspan!(ast::ExprType::Number(1)))))
        )))));
        stmt = stmt.appending(
            env.typecheck_expr_mut(&subscript_expr)
                .expect("typecheck failed")
                .expr
                .unwrap_stmt(&tmp_generator),
        );

        let mut interpreter = Interpreter::default();
        interpreter.run_expr(&tmp_generator, Expr::Stmt(stmt));
        assert!(interpreter.panicked);

        // a[10];
        let subscript_expr = zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Subscript(
            Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
            zspan!(ast::ExprType::Number(10))
        )))));
        let trexpr = env.typecheck_expr_mut(&subscript_expr).expect("typecheck failed").expr;
        interpreter.run_expr(&tmp_generator, trexpr);
        assert!(interpreter.panicked);
    }

    #[test]
    fn record_expr() -> Result<(), Vec<TypecheckError>> {
        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let mut env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let record_decl = zspan!(ast::TypeDecl {
            id: zspan!("r".to_owned()),
            ty: zspan!(ast::TypeDeclType::Record(vec![
                zspan!(ast::TypeField {
                    id: zspan!("a".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("int".to_owned()))),
                }),
                zspan!(ast::TypeField {
                    id: zspan!("b".to_owned()),
                    ty: zspan!(ast::Type::Type(zspan!("string".to_owned()))),
                })
            ]))
        });
        env.typecheck_type_decl(&record_decl)?;
        env.convert_pre_types();

        let assign_expr = zspan!(ast::ExprType::Let(Box::new(zspan!(ast::Let {
            pattern: zspan!(ast::Pattern::String("a".to_owned())),
            immutable: zspan!(true),
            ty: None,
            expr: zspan!(ast::ExprType::Record(Box::new(zspan!(ast::Record {
                id: zspan!("r".to_owned()),
                field_assigns: vec![
                    zspan!(ast::FieldAssign {
                        id: zspan!("a".to_owned()),
                        expr: zspan!(ast::ExprType::Number(123)),
                    }),
                    zspan!(ast::FieldAssign {
                        id: zspan!("b".to_owned()),
                        expr: zspan!(ast::ExprType::String("string".to_owned())),
                    }),
                ]
            }))))
        }))));
        let stmt = env.typecheck_expr_mut(&assign_expr)?.expr.unwrap_stmt(&tmp_generator);
        dump_vec(&stmt.flatten());

        let mut interpreter = Interpreter::default();
        {
            let fragments = env.fragments.borrow();
            interpreter.set_string_table(&fragments);
        }

        interpreter.run_expr(&tmp_generator, Expr::Stmt(stmt));

        println!("{:?}", interpreter.string_table);
        interpreter.dump_to_file("test_record_expr");

        {
            let subscript_expr = zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Field(
                Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
                zspan!("a".to_owned())
            )))));
            let trexpr = env.typecheck_expr(&subscript_expr)?.expr;
            dump_vec(&trexpr.clone().unwrap_stmt(&tmp_generator).flatten());
            assert_eq!(interpreter.run_expr(&tmp_generator, trexpr).expect("run_expr"), 123);
        }

        {
            let subscript_expr = zspan!(ast::ExprType::LVal(Box::new(zspan!(ast::LVal::Field(
                Box::new(zspan!(ast::LVal::Simple("a".to_owned()))),
                zspan!("b".to_owned())
            )))));
            let trexpr = env.typecheck_expr(&subscript_expr)?.expr;
            let result = interpreter.run_expr(&tmp_generator, trexpr).expect("run_expr");
            assert_eq!(interpreter.read_str(result as usize), "string".to_owned());
        }

        Ok(())
    }

    #[test]
    fn simple_while() -> Result<(), InterpreterTestError> {
        let expr = parser::parse_expr(
            r#"
            {
                let mut a = 10;
                let mut b = 0;
                while a > 0 {
                    b = b + 1;
                    a = a - 1;
                };
                b
            }
        "#,
        )?;

        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let expr = env.typecheck_expr(&expr)?.expr;

        let mut interpreter = Interpreter::default();
        let result = interpreter.run_expr(&tmp_generator, expr);
        assert_eq!(result, Some(10));

        Ok(())
    }

    #[test]
    fn while_break() -> Result<(), InterpreterTestError> {
        let expr = parser::parse_expr(
            r#"
            {
                let mut a = 0;
                while true {
                    a = a + 1;
                    if a == 10 {
                        break;
                    };
                };
                a
            }
        "#,
        )?;

        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let expr = env.typecheck_expr(&expr)?.expr;

        let mut interpreter = Interpreter::default();
        let result = interpreter.run_expr(&tmp_generator, expr);
        assert_eq!(result, Some(10));
        Ok(())
    }

    #[test]
    fn while_continue() -> Result<(), InterpreterTestError> {
        let expr = parser::parse_expr(
            r#"
            {
                let mut a = 0;
                let mut b = true;
                while true {
                    a = a + 1;
                    if b {
                        b = false;
                        continue;
                    } else {
                        break;
                    };
                };
                a
            }
        "#,
        )?;

        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let expr = env.typecheck_expr(&expr)?.expr;
        dump_vec(&expr.clone().unwrap_stmt(&tmp_generator).flatten());

        let mut interpreter = Interpreter::default();
        let result = interpreter.run_expr(&tmp_generator, expr);
        assert_eq!(result, Some(2));
        Ok(())
    }

    #[test]
    fn simple_for() -> Result<(), InterpreterTestError> {
        let expr = parser::parse_expr(
            r#"
            {
                let mut a = 0;
                for i in 0..10 {
                    a = a + i;
                };
                a
            }
        "#,
        )?;

        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let expr = env.typecheck_expr(&expr)?.expr;
        dump_vec(&expr.clone().unwrap_stmt(&tmp_generator).flatten());

        let mut interpreter = Interpreter::default();
        let result = interpreter.run_expr(&tmp_generator, expr);
        assert_eq!(result, Some(45));
        Ok(())
    }

    #[test]
    fn for_continue() -> Result<(), InterpreterTestError> {
        let expr = parser::parse_expr(
            r#"
            {
                let mut a = 0;
                for i in 0..10 {
                    if i % 2 == 0 {
                        continue;
                    };
                    a = a + i;
                };
                a
            }
        "#,
        )?;

        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let expr = env.typecheck_expr(&expr)?.expr;

        let mut interpreter = Interpreter::default();
        let result = interpreter.run_expr(&tmp_generator, expr);
        assert_eq!(result, Some(1 + 3 + 5 + 7 + 9));
        Ok(())
    }

    #[test]
    fn for_break() -> Result<(), InterpreterTestError> {
        let expr = parser::parse_expr(
            r#"
            {
                let mut a = 0;
                for i in 1..10 {
                    if i % 3 == 0 {
                        break;
                    };
                    a = a + i;
                };
                a
            }
        "#,
        )?;

        let tmp_generator = TmpGenerator::default();
        let level = Level::new(&tmp_generator, Some(Label::top()), "f", &[]);
        let level_label = level.frame.label.clone();

        let env = Env::new(tmp_generator.clone(), EMPTY_SOURCEMAP.1, level_label.clone());
        {
            let mut levels = env.levels.borrow_mut();
            levels.insert(level_label, level);
        }

        let expr = env.typecheck_expr(&expr)?.expr;

        let mut interpreter = Interpreter::default();
        let result = interpreter.run_expr(&tmp_generator, expr);
        assert_eq!(result, Some(1 + 2));
        Ok(())
    }
}
