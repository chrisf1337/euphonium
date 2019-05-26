use crate::{ast::Spanned, lexer::LexError};
use codespan::{ByteIndex, ByteOffset, ByteSpan};
use lalrpop_util::ParseError;
use std::{convert::TryFrom, fmt::Debug};

pub trait Error: Debug {
    fn span(&self) -> ByteSpan;
}

impl<T: Debug> Error for Spanned<T> {
    fn span(&self) -> ByteSpan {
        self.span
    }
}

impl<T> Error for ParseError<ByteIndex, T, LexError>
where
    T: Debug,
{
    fn span(&self) -> ByteSpan {
        match self {
            ParseError::InvalidToken { location } => {
                ByteSpan::new(*location, *location + ByteOffset(1))
            }
            ParseError::UnrecognizedToken {
                token: (l, _, r), ..
            } => ByteSpan::new(*l, *r),
            ParseError::UnrecognizedEOF { location, .. } => {
                ByteSpan::new(*location, *location + ByteOffset(1))
            }
            ParseError::ExtraToken { token: (l, _, r) } => ByteSpan::new(*l, *r),
            ParseError::User { error } => ByteSpan::new(
                ByteIndex(u32::try_from(error.index).unwrap()),
                ByteIndex(u32::try_from(error.index + 1).unwrap()),
            ),
        }
    }
}
