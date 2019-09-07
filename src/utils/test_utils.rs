use crate::sourcemap::Sourcemap;
use codespan::FileId;
use lazy_static::lazy_static;

lazy_static! {
    pub static ref EMPTY_SOURCEMAP: (Sourcemap, FileId) = {
        let mut sourcemap = Sourcemap::default();
        let file_id = sourcemap.add_empty_file();
        (sourcemap, file_id)
    };
}

macro_rules! zspan {
    ( ) => {
        ast::FileSpan::new(crate::utils::EMPTY_SOURCEMAP.1, codespan::Span::initial())
    };
    ( $e:expr ) => {
        ast::Spanned::new(
            $e,
            ast::FileSpan::new(crate::utils::EMPTY_SOURCEMAP.1, codespan::Span::initial()),
        )
    };
}
