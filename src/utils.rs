pub mod u64ext;

macro_rules! zspan {
    ( ) => {
        codespan::ByteSpan::new(codespan::ByteIndex::none(), codespan::ByteIndex::none())
    };
    ( $e:expr ) => {
        ast::Spanned::new($e, zspan!())
    };
}

macro_rules! span {
    ( $l:expr , $r:expr , $off:expr) => {{
        use codespan::{ByteIndex, ByteSpan};
        ByteSpan::new(ByteIndex($l as u32) + $off, ByteIndex($r as u32) + $off)
    }};
}
