macro_rules! span {
    ( $l:expr , $r:expr , $off:expr) => {{
        use codespan::{ByteIndex, ByteSpan};
        ByteSpan::new(ByteIndex($l as u32) + $off, ByteIndex($r as u32) + $off)
    }};
}
