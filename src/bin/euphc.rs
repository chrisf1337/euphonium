use clap::{App, Arg};
use codespan_reporting::termcolor::{ColorChoice, StandardStream};
use euphonium::sourcemap::{Sourcemap, SourcemapError};

fn main() -> Result<(), SourcemapError> {
    let matches = App::new("euphc")
        .about("Euphonium compiler")
        .arg(
            Arg::with_name("FILES")
                .required(true)
                .multiple(true)
                .help(""),
        )
        .get_matches();
    let files = matches.values_of("FILES").unwrap();
    let mut sourcemap = Sourcemap::default();
    for file in files {
        let _decls = sourcemap.add_file_from_disk(file)?;
    }
    let mut _writer = StandardStream::stderr(ColorChoice::Auto);

    Ok(())
}
