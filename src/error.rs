use crate::{ast::Spanned, lexer::LexError};
use codespan::{ByteIndex, ByteOffset, Span};
use lalrpop_util::ParseError;
use std::{convert::TryFrom, fmt::Debug};

pub trait Error: Debug {
    fn span(&self) -> Span;
}

impl<T: Debug> Error for Spanned<T> {
    fn span(&self) -> Span {
        self.span.span
    }
}

impl<T> Error for ParseError<ByteIndex, T, LexError>
where
    T: Debug,
{
    fn span(&self) -> Span {
        match self {
            ParseError::InvalidToken { location } => Span::new(*location, *location + ByteOffset(1)),
            ParseError::UnrecognizedToken { token: (l, _, r), .. } => Span::new(*l, *r),
            ParseError::UnrecognizedEOF { location, .. } => Span::new(*location, *location + ByteOffset(1)),
            ParseError::ExtraToken { token: (l, _, r) } => Span::new(*l, *r),
            ParseError::User { error } => Span::new(
                ByteIndex(u32::try_from(error.index).unwrap()),
                ByteIndex(u32::try_from(error.index + 1).unwrap()),
            ),
        }
    }
}
