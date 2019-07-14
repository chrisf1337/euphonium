#![allow(clippy::all)]

use clap::{App, Arg};
use codespan_reporting::{
    termcolor::{ColorChoice, StandardStream},
    Diagnostic, Label,
};
use euphonium::{
    error::Error,
    sourcemap::{Sourcemap, SourcemapError},
    tmp::TmpGenerator,
    typecheck::Env,
};

#[derive(Debug)]
enum EuphcErr {
    Io(std::io::Error),
    ParseErr,
    TypecheckErr,
}

impl From<std::io::Error> for EuphcErr {
    fn from(err: std::io::Error) -> Self {
        EuphcErr::Io(err)
    }
}

fn main() -> Result<(), EuphcErr> {
    let matches = App::new("euphc")
        .about("Euphonium compiler")
        .arg(Arg::with_name("FILES").required(true).multiple(true).help(""))
        .get_matches();
    let files = matches.values_of("FILES").unwrap();
    let mut sourcemap = Sourcemap::default();
    let mut decls = vec![];
    let mut errors = vec![];
    for file in files {
        match sourcemap.add_file_from_disk(file) {
            Ok(ds) => decls.extend(ds),
            Err(err) => errors.push(err),
        }
    }
    let mut writer = StandardStream::stderr(ColorChoice::Auto);

    if !errors.is_empty() {
        for err in errors {
            match err {
                SourcemapError::Parse(parse_errs) => {
                    for parse_err in parse_errs {
                        codespan_reporting::emit(
                            &mut writer,
                            &sourcemap.codemap,
                            &Diagnostic::new_error("parse error").with_label(
                                Label::new_primary(parse_err.span()).with_message(format!("{:?}", parse_err)),
                            ),
                        )?;
                    }
                }
                SourcemapError::Io(path, err) => {
                    codespan_reporting::emit(
                        &mut writer,
                        &sourcemap.codemap,
                        &Diagnostic::new_error(format!("{}: IO error: {:?}", path.to_str().unwrap_or(""), err)),
                    )?;
                }
            }
        }
        return Err(EuphcErr::ParseErr);
    }

    let mut env = Env::default();
    let mut tmp_generator = TmpGenerator::default();
    match env.translate_decls(&mut tmp_generator, &decls) {
        Ok(()) => (),
        Err(typecheck_errors) => {
            for err in typecheck_errors {
                codespan_reporting::emit(
                    &mut writer,
                    &sourcemap.codemap,
                    &err.ty.diagnostic(&env).with_labels(err.labels()),
                )?;
            }
            return Err(EuphcErr::TypecheckErr);
        }
    }

    println!("{:#?}", env);

    Ok(())
}
