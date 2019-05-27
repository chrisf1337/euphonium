#![allow(clippy::all)]

use clap::{App, Arg};
use codespan_reporting::{
    termcolor::{ColorChoice, StandardStream},
    Diagnostic, Label,
};
use euphonium::{
    error::Error,
    sourcemap::{Sourcemap, SourcemapError},
};

fn main() -> Result<(), std::io::Error> {
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
    let mut decls = vec![];
    let mut errors = vec![];
    for file in files {
        match sourcemap.add_file_from_disk(file) {
            Ok(decl) => decls.push(decl),
            Err(err) => errors.push(err),
        }
    }
    let mut writer = StandardStream::stderr(ColorChoice::Auto);

    if errors.is_empty() {
        println!("{:?}", decls);
    } else {
        for err in errors {
            match err {
                SourcemapError::Parse(parse_errs) => {
                    for parse_err in parse_errs {
                        codespan_reporting::emit(
                            &mut writer,
                            &sourcemap.codemap,
                            &Diagnostic::new_error("parse error").with_label(
                                Label::new_primary(parse_err.span())
                                    .with_message(format!("{:?}", parse_err)),
                            ),
                        )?;
                    }
                }
                SourcemapError::Io(path, err) => {
                    codespan_reporting::emit(
                        &mut writer,
                        &sourcemap.codemap,
                        &Diagnostic::new_error(format!(
                            "{}: IO error: {:?}",
                            path.to_str().unwrap_or(""),
                            err
                        )),
                    )?;
                }
            }
        }
    }

    Ok(())
}
