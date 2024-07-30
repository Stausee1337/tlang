use std::{env, fs::File, io::{Read, self}, path::Path, process::ExitCode};

use getopts::{Options, ParsingStyle};
use parse::ParseContext;

mod lexer;
mod symbol;
mod parse;

fn print_usage(program: &str, opts: Options) {
    let brief = format!("Usage: {} [options] [-c cmd | file] [arg]", program);
    print!("{}", opts.usage(&brief));
}

fn read_entire_file(filename: &Path) -> Result<String, io::Error> {
    let mut result = String::new();
    File::open(filename)?
        .read_to_string(&mut result)?;
    Ok(result)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    let program = args[0].clone();

    let mut opts = Options::new();
    opts.parsing_style(ParsingStyle::StopAtFirstFree);
    opts.optopt("c", "", "run immediate code", "CODE");
    opts.optflag("i", "interactive", "drop into interactive mode after executing the file");

    opts.optflag("", "print-ast", "print the parsed ast");
    opts.optflag("", "print-bytecode", "print generated bytecode");

    opts.optflag("h", "help", "print this help menu");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => { m }
        Err(f) => { panic!("{}", f.to_string()) }
    };
    if matches.opt_present("h") {
        print_usage(&program, opts);
        return ExitCode::SUCCESS;
    }

    let input = if !matches.free.is_empty() {
        matches.free
    } else {
        todo!("Interactive mode is not supported yet");
    };

    if let Some(filename) = input.first() {
        let mut filepath = std::env::current_dir().unwrap();
        filepath.push(filename);
        // let filename = PathBuf::from_str(filename).unwrap();
        let contents = match read_entire_file(&filepath) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("{}: can't open file {:?}: {}", program, filepath, err);
                return ExitCode::FAILURE;
            }
        };

        let tokens = lexer::tokenize(&contents).unwrap();
        let ctx = ParseContext::new(tokens);
        let module = ctx.parse().unwrap();
        println!("{:#?}", module);

        return ExitCode::SUCCESS;
    }


    return ExitCode::SUCCESS;
}
