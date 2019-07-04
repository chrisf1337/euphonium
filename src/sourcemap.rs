use crate::{ast, lexer, parser};
use codespan::{ByteIndex, ByteOffset, CodeMap, FileMap, RawOffset};
use lalrpop_util::{ErrorRecovery, ParseError};
use std::{
    collections::HashMap,
    convert::TryFrom,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Default)]
pub struct Sourcemap {
    pub codemap: CodeMap,
    filemaps: HashMap<PathBuf, Arc<FileMap>>,
}

#[derive(Debug)]
pub enum SourcemapError {
    Parse(Vec<ParseError<ByteIndex, lexer::Tok, lexer::LexError>>),
    Io(PathBuf, std::io::Error),
}

impl Sourcemap {
    pub fn add_file_from_disk<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<ast::Decl>, SourcemapError> {
        match File::open(path.as_ref()) {
            Ok(file) => {
                let mut buf_reader = BufReader::new(file);
                let mut contents = String::new();
                match buf_reader.read_to_string(&mut contents) {
                    Ok(_) => self.add_file(codespan::FileName::Real(path.as_ref().to_path_buf()), contents),
                    Err(err) => Err(SourcemapError::Io(path.as_ref().to_path_buf(), err)),
                }
            }
            Err(err) => Err(SourcemapError::Io(path.as_ref().to_path_buf(), err)),
        }
    }

    pub fn add_file(&mut self, filename: codespan::FileName, file: String) -> Result<Vec<ast::Decl>, SourcemapError> {
        let lexer = lexer::Lexer::new(&file);
        let mut errors = vec![];
        let filemap = self.codemap.add_filemap(filename, file);
        match parser::ProgramParser::new().parse(ByteOffset(RawOffset::from(filemap.span().start().0)), &mut errors, lexer)
        {
            Ok(decls) => {
                if errors.is_empty() {
                    Ok(decls)
                } else {
                    let mut converted_errors = vec![];
                    for ErrorRecovery { error, .. } in errors {
                        converted_errors.push(
                            error.map_location(|loc| ByteIndex(filemap.span().start().0 + u32::try_from(loc).unwrap())),
                        );
                    }
                    Err(SourcemapError::Parse(converted_errors))
                }
            }
            Err(err) => {
                let mut converted_errors = vec![];
                for ErrorRecovery { error, .. } in errors {
                    converted_errors.push(
                        error.map_location(|loc| ByteIndex(filemap.span().start().0 + u32::try_from(loc).unwrap())),
                    );
                }
                converted_errors
                    .push(err.map_location(|loc| ByteIndex(filemap.span().start().0 + u32::try_from(loc).unwrap())));
                Err(SourcemapError::Parse(converted_errors))
            }
        }
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
