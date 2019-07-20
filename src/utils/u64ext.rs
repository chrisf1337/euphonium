pub trait U64Ext {
    fn iadd(self, i: i64) -> Self;
}

impl U64Ext for u64 {
    fn iadd(self, i: i64) -> u64 {
        if i < 0 {
            self - (-i as u64)
        } else {
            self + (i as u64)
        }
    }
}
