pub mod u64ext;

use crate::sourcemap::Sourcemap;
use codespan::FileId;
use lazy_static::lazy_static;

lazy_static! {
    static ref EMPTY_SOURCEMAP: (Sourcemap, FileId) = {
        let mut sourcemap = Sourcemap::default();
        let file_id = sourcemap.add_empty_file();
        (sourcemap, file_id)
    };
}

macro_rules! zspan {
    ( ) => {
        ast::FileSpan::new(EMPTY_SOURCEMAP.1, codespan::Span::initial())
    };
    ( $file_id:expr ) => {
        ast::FileSpan::new($file_id, codespan::Span::initial())
    };
    ( $file_id:expr, $e:expr ) => {
        ast::Spanned::new($e, ast::FileSpan::new($file_id, codespan::Span::initial()))
    };
}
