macro_rules! expr {
    ( $ty:expr ) => {
        Expr::new($ty, span!(0, 0, codespan::ByteOffset(0)))
    };
}

macro_rules! zspan {
    ( $e:expr ) => {
        Spanned::new($e, span!(0, 0, codespan::ByteOffset(0)))
    };
}
