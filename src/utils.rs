macro_rules! zspan {
    ( $e:expr ) => {
        Spanned::new($e, span!(0, 0, codespan::ByteOffset(0)))
    };
}

macro_rules! span {
    ( $l:expr , $r:expr , $off:expr) => {{
        use codespan::{ByteIndex, ByteSpan};
        ByteSpan::new(ByteIndex($l as u32) + $off, ByteIndex($r as u32) + $off)
    }};
}
