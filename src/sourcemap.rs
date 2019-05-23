use crate::{ast, lexer, parser};
use codespan::{ByteOffset, CodeMap, FileMap, RawOffset};
use lalrpop_util::{ErrorRecovery, ParseError};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Default)]
pub struct Sourcemap {
    codemap: CodeMap,
    filemaps: HashMap<PathBuf, Arc<FileMap>>,
}

#[derive(Debug)]
pub enum SourcemapError {
    Parse(
        ParseError<usize, crate::lexer::Tok, crate::lexer::LexError>,
        Vec<ErrorRecovery<usize, crate::lexer::Tok, crate::lexer::LexError>>,
    ),
    Io(std::io::Error),
}

impl From<std::io::Error> for SourcemapError {
    fn from(err: std::io::Error) -> Self {
        SourcemapError::Io(err)
    }
}

impl Sourcemap {
    pub fn add_file_from_disk<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<Vec<ast::Decl>, SourcemapError> {
        let file = File::open(path.as_ref())?;
        let mut buf_reader = BufReader::new(file);
        let mut contents = String::new();
        buf_reader.read_to_string(&mut contents)?;
        self.add_file(
            codespan::FileName::Real(path.as_ref().to_path_buf()),
            contents,
        )
    }

    pub fn add_file(
        &mut self,
        filename: codespan::FileName,
        file: String,
    ) -> Result<Vec<ast::Decl>, SourcemapError> {
        let lexer = lexer::Lexer::new(&file);
        let mut errors = vec![];
        let filemap = self.codemap.add_filemap(filename, file);
        match parser::ProgramParser::new().parse(
            ByteOffset(filemap.span().start().0 as RawOffset),
            &mut errors,
            lexer,
        ) {
            Ok(decls) => Ok(decls),
            Err(err) => Err(SourcemapError::Parse(err, errors)),
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
