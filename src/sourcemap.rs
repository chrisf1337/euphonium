use crate::{ast, lexer, parser};
use codespan::{ByteIndex, ByteOffset, FileId, Files, RawOffset};
use lalrpop_util::{ErrorRecovery, ParseError};
use std::{
    collections::HashMap,
    convert::TryFrom,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
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

#[derive(Debug)]
pub struct SourcemapError {
    pub errors: Vec<ParseError<ByteIndex, lexer::Tok, lexer::LexError>>,
}

impl Sourcemap {
    pub fn add_file_from_disk<P: AsRef<Path>>(&mut self, path: P) -> (FileId, Result<Vec<ast::Decl>, SourcemapError>) {
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

    pub fn add_file(&mut self, filename: String, file: String) -> (FileId, Result<Vec<ast::Decl>, SourcemapError>) {
        let lexer = lexer::Lexer::new(&file);
        let mut errors = vec![];
        let file_id = self.files.add(filename, file);
        let span = self.files.source_span(file_id);
        match parser::ProgramParser::new().parse(file_id, &mut errors, lexer) {
            Ok(decls) => {
                if errors.is_empty() {
                    (file_id, Ok(decls))
                } else {
                    let mut converted_errors = vec![];
                    for ErrorRecovery { error, .. } in errors {
                        converted_errors
                            .push(error.map_location(|loc| ByteIndex(span.start().0 + u32::try_from(loc).unwrap())));
                    }
                    (
                        file_id,
                        Err(SourcemapError {
                            errors: converted_errors,
                        }),
                    )
                }
            }
            Err(err) => {
                let mut converted_errors = vec![];
                for ErrorRecovery { error, .. } in errors {
                    converted_errors
                        .push(error.map_location(|loc| ByteIndex(span.start().0 + u32::try_from(loc).unwrap())));
                }
                converted_errors.push(err.map_location(|loc| ByteIndex(span.start().0 + u32::try_from(loc).unwrap())));
                (
                    file_id,
                    Err(SourcemapError {
                        errors: converted_errors,
                    }),
                )
            }
        }
    }

    pub fn add_empty_file(&mut self) -> FileId {
        self.files.add("empty", "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_file() {
        let mut sourcemap = Sourcemap::default();
        dbg!(sourcemap
            .add_file(
                codespan::FileName::Real(PathBuf::from("test/test_exprs.euph")),
                "fn f() = {};".to_owned(),
            )
            .expect("failed to parse"));
    }
}
