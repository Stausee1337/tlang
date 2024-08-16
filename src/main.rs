#![feature(fn_traits)]
#![feature(tuple_trait)]
#![feature(unboxed_closures)]
#![feature(more_qualified_paths)]

use std::{env, fs::File, io::{Read, self}, path::Path, process::ExitCode};

extern crate self as tlang;

use bumpalo::Bump;
use eval::TArgsBuffer;
use getopts::{Options, ParsingStyle};
use interop::TPolymorphicCallable;
use memory::GCRef;
use serde::Deserialize;
use tvalue::{TString, TFunction, TInteger};
use vm::TModule;

use crate::{bytecode::BytecodeGenerator, tvalue::TValue, interop::VMCast, symbol::Symbol};

mod lexer;
mod symbol;
mod parse;
mod codegen;
mod bytecode;
mod tvalue;
mod eval;
mod vm;
mod memory;
mod bigint;
mod interop;

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

#[derive(serde::Serialize, serde::Deserialize)]
#[repr(packed)]
struct Test {
    x: u8,
    b: u16
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

    let vm = vm::VM::init();

    let input = if !matches.free.is_empty() {
        matches.free
    } else {
        todo!("Interactive mode is not supported yet");
    };

    if let Some(filename) = input.first() {
        let mut filepath = std::env::current_dir().unwrap();
        filepath.push(filename);
        let contents = match read_entire_file(&filepath) {
            Ok(s) => s,
            Err(err) => {
                eprintln!("{}: can't open file {:?}: {}", program, filepath, err);
                return ExitCode::FAILURE;
            }
        };

        let source = TString::from_slice(&vm, &contents);

        let arena = Bump::new();
        let ast = parse::parse_from_tstring(vm.clone(), source, &arena).unwrap();

        let modname = if let Some(idx) = filename.find('.') {
            filename.split_at(idx).0
        } else {
            filename
        };

        let mut module = TModule::new(&vm, TString::from_slice(&vm, modname));
        module.set_source(Some(source));


        let printfn = TFunction::rustfunc(module, Some("print"), |msg| tvalue::print(module, msg));

        module.set_global(
            Symbol![print],
            printfn.into(),
            true
        );

        let generator = BytecodeGenerator::new(module);
        let module_func: TPolymorphicCallable<_, ()> = codegen::generate_module(ast, generator).unwrap().into();
        module_func();

        drop(vm);

        return ExitCode::SUCCESS;
    }

    drop(vm);

    return ExitCode::SUCCESS;
}
