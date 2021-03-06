#![allow(clippy::all)]

use clap::{App, Arg};
use codespan_reporting::{
    diagnostic::{Diagnostic, Label},
    term::{
        termcolor::{ColorChoice, StandardStream},
        Config,
    },
};
use euphonium::{
    parser::ParseErrorExt,
    sourcemap::Sourcemap,
    tmp::{self, TmpGenerator},
    typecheck::Env,
};

#[derive(Debug)]
enum EuphcError {
    Io(std::io::Error),
    ParseError,
    TypecheckError,
}

impl From<std::io::Error> for EuphcError {
    fn from(err: std::io::Error) -> Self {
        EuphcError::Io(err)
    }
}

fn main() -> Result<(), EuphcError> {
    let matches = App::new("euphc")
        .about("Euphonium compiler")
        .arg(Arg::with_name("FILES").required(true).multiple(true).help(""))
        .get_matches();
    let files = matches.values_of("FILES").unwrap();
    let mut sourcemap = Sourcemap::default();
    let empty_file_id = sourcemap.add_empty_file();
    let mut decls = vec![];
    let mut errors = vec![];
    for file in files {
        match sourcemap.add_file_from_disk(file) {
            (file_id, Ok(ds)) => decls.push((file_id, ds)),
            (file_id, Err(err)) => errors.push((file_id, err)),
        }
    }
    let mut writer = StandardStream::stderr(ColorChoice::Auto);
    let config = Config::default();

    if !errors.is_empty() {
        for (file_id, parse_errs) in errors {
            for parse_err in parse_errs {
                let label = Label::new(file_id, parse_err.span(), format!("{:?}", parse_err));
                codespan_reporting::term::emit(
                    &mut writer,
                    &config,
                    &sourcemap.files,
                    &Diagnostic::new_error("parse error", label),
                )?;
            }
        }
        return Err(EuphcError::ParseError);
    }

    let tmp_generator = TmpGenerator::default();
    let mut env = Env::new(tmp_generator, empty_file_id, tmp::Label::top());
    for (file_id, decls) in decls {
        match env.typecheck_decls(file_id, &decls) {
            Ok(()) => (),
            Err(typecheck_errors) => {
                for err in typecheck_errors {
                    codespan_reporting::term::emit(&mut writer, &config, &sourcemap.files, &err.diagnostic(&env))?;
                }
                return Err(EuphcError::TypecheckError);
            }
        }
    }

    println!("{:#?}", env);

    Ok(())
}
