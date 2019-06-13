macro_rules! zspan {
    ( ) => {
        codespan::ByteSpan::new(codespan::ByteIndex::none(), codespan::ByteIndex::none())
    };
    ( $e:expr ) => {
        Spanned::new($e, zspan!())
    };
}
