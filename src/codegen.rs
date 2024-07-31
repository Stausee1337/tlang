
use crate::lexer::Span;
use crate::parse::{IfBranch, Break, Return, Continue, Import, ForLoop, WhileLoop, Variable, Function, AssignExpr, Literal, Ident, BinaryExpr, UnaryExpr, CallExpr, AttributeExpr, SubscriptExpr, ListExpr, ObjectExpr, TupleExpr, Lambda, Module, Statement, LiteralKind};

use crate::bytecode::{CGValue, BytecodeGenerator};

#[derive(Debug)]
pub enum CodegenErr {
    SyntaxError {
        message: Option<String>,
        span: Span
    }
}

pub type CodegenResult = Result<Option<CGValue>, CodegenErr>;

pub trait GeneratorNode {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult;
}

pub fn generate_module<'ast>(generator: &mut BytecodeGenerator, module: Module<'ast>) -> CodegenResult {
    generate_body(generator, module.body) 
}

pub fn generate_body<'ast>(generator: &mut BytecodeGenerator, body: &'ast [&'ast Statement]) -> CodegenResult {
    for stmt in body {
        match stmt {
            Statement::Variable(ref var) =>
                generator.register_variable(var.name, var.constant)
                .map_err(|()| CodegenErr::SyntaxError {
                    span: var.span,
                    message: Some(format!("identifier `{}` has already been declared", var.name.symbol.get()))
                })?,
            Statement::If(..) |
            Statement::ForLoop(..) | Statement::WhileLoop(..) =>
                break,
            _ => ()
        }
    }

    for stmt in body {
        stmt.generate_bytecode(generator)?;
    }
    Ok(None)
}

/// STATEMENTS

impl<'ast> GeneratorNode for IfBranch<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Break {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Return<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Continue {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Import<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for ForLoop<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for WhileLoop<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Variable<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Function<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

/// EXPRESSIONS

impl<'ast> GeneratorNode for AssignExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Literal<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        Ok(Some(match self.kind {
            LiteralKind::Null => CGValue::null(),
            LiteralKind::Integer(int) => generator.make_int(int),
            LiteralKind::Float(float) => generator.make_float(float),
            LiteralKind::Boolean(bool) => CGValue::bool(bool),
            LiteralKind::String(str) => 
                generator.make_string_literal(str)
                .map_err(|err| CodegenErr::SyntaxError {
                    message: Some(err.to_string()),
                    span: self.span
                })?,
        }))
    }
}

impl<'ast> GeneratorNode for Ident {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for BinaryExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for UnaryExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for CallExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for AttributeExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for SubscriptExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for ListExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for ObjectExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for TupleExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for Lambda<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

