macro_rules! expr {
    ( $ty:expr ) => {
        Expr::new($ty, (0, 0))
    };
}

macro_rules! span {
    ( $e:expr ) => {
        Spanned::new($e, (0, 0))
    };
}
