use crate::{
    ast,
    parser::{parse_program, ParseError},
};
use codespan::{FileId, Files};
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

#[derive(Clone)]
pub struct Sourcemap {
    pub files: Files,
}

impl Default for Sourcemap {
    fn default() -> Self {
        Sourcemap { files: Files::new() }
    }
}

impl Sourcemap {
    pub fn add_file_from_disk<P: AsRef<Path>>(&mut self, path: P) -> (FileId, Result<Vec<ast::Decl>, Vec<ParseError>>) {
        match File::open(path.as_ref()) {
            Ok(file) => {
                let mut buf_reader = BufReader::new(file);
                let mut contents = String::new();
                match buf_reader.read_to_string(&mut contents) {
                    Ok(_) => self.add_file(path.as_ref().to_path_buf().to_string_lossy().to_string(), contents),
                    Err(err) => panic!("{:?}", err),
                }
            }
            Err(err) => panic!("{:?}", err),
        }
    }

    pub fn add_file(
        &mut self,
        filename: impl Into<String>,
        file: impl Into<String>,
    ) -> (FileId, Result<Vec<ast::Decl>, Vec<ParseError>>) {
        let file = file.into();
        let file_id = self.files.add(filename, file.clone());
        (file_id, parse_program(file_id, &file))
    }

    pub fn add_empty_file(&mut self) -> FileId {
        self.files.add("empty", "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_file() {
        let mut sourcemap = Sourcemap::default();
        dbg!(sourcemap
            .add_file("test/test_exprs.euph", "fn f() = {};")
            .1
            .expect("failed to parse"));
    }
}
