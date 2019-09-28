use crate::{
    ast::Decl,
    lexer::{LexError, Lexer, Tok},
};
use codespan::{FileId, Span};
use lalrpop_util::lalrpop_mod;
use std::convert::TryFrom;

lalrpop_mod! {
    #[allow(clippy::all)]
    lalrpop_parser
}

pub type ParseError = lalrpop_util::ParseError<u32, Tok, LexError>;
pub type Result<T> = std::result::Result<T, Vec<ParseError>>;

pub fn parse_program(file_id: FileId, program: &str) -> Result<Vec<Decl>> {
    let mut errors = vec![];
    let lexer = Lexer::new(program);
    match lalrpop_parser::ProgramParser::new().parse(file_id, &mut errors, lexer) {
        Ok(decls) => {
            if errors.is_empty() {
                Ok(decls)
            } else {
                Err(errors.into_iter().map(|e| e.error).collect())
            }
        }
        Err(err) => {
            let mut errors: Vec<ParseError> = errors.into_iter().map(|e| e.error).collect();
            errors.insert(0, err);
            Err(errors)
        }
    }
}

pub trait ParseErrorExt {
    fn span(&self) -> Span;
}

impl ParseErrorExt for ParseError {
    fn span(&self) -> Span {
        match self {
            ParseError::InvalidToken { location } => Span::new(*location, *location + 1),
            ParseError::UnrecognizedToken { token: (l, _, r), .. } => Span::new(*l, *r),
            ParseError::UnrecognizedEOF { location, .. } => Span::new(*location, *location + 1),
            ParseError::ExtraToken { token: (l, _, r) } => Span::new(*l, *r),
            ParseError::User { error } => Span::new(
                u32::try_from(error.index).unwrap(),
                u32::try_from(error.index + 1).unwrap(),
            ),
        }
    }
}
