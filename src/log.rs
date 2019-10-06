use lazy_static::lazy_static;
use slog::{o, Drain, Logger};
use slog_async;
use slog_term;
use std::fs::OpenOptions;

lazy_static! {
    pub static ref LOG: Logger = {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open("target/euphc.log")
            .unwrap();
        let decorator = slog_term::PlainDecorator::new(file);
        let drain = slog_term::FullFormat::new(decorator).build().fuse();
        let drain = slog_async::Async::new(drain).build().fuse();
        Logger::root(drain, o!("component" => "root"))
    };
    pub static ref TYPECHECK_LOG: Logger = { LOG.new(o!("component" => "typecheck")) };
}
