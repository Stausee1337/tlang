
use crate::lexer::Span;
use crate::parse::{IfBranch, Break, Return, Continue, Import, ForLoop, WhileLoop, Variable, Function, AssignExpr, Literal, Ident, BinaryExpr, UnaryExpr, CallExpr, AttributeExpr, SubscriptExpr, ListExpr, ObjectExpr, TupleExpr, Lambda, Module, Statement, LiteralKind, Expression};

use crate::bytecode::{Operand, BytecodeGenerator, CodeLabel};

#[derive(Debug)]
pub enum CodegenErr {
    SyntaxError {
        message: Option<String>,
        span: Span
    }
}

pub type CodegenResult = Result<Option<Operand>, CodegenErr>;

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
                generator.register_local(var.name, var.constant)
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
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        match self.lhs {
            Expression::Ident(ident) => {
                let src = self.rhs.generate_bytecode(generator)?.unwrap();
                if let Some(local) = generator.find_local(ident.symbol) {
                    if local.constant {
                        generator.emit_error();
                        return Ok(Some(Operand::null()));
                    }

                    generator.emit_mov(src, generator.get_local_reg(ident.symbol));
                    return Ok(Some(src));
                }

                generator.emit_set_global(src, ident.symbol);

                Ok(Some(src))
            }
            Expression::AttributeExpr(attr) => {
                let src = self.rhs.generate_bytecode(generator)?.unwrap();
                let base = attr.base.generate_bytecode(generator)?.unwrap();

                generator.emit_set_attribute(src, base, attr.attr.symbol);

                Ok(Some(src))
            }
            Expression::SubscriptExpr(subs) => {
                let src = self.rhs.generate_bytecode(generator)?.unwrap();
                let base = subs.base.generate_bytecode(generator)?.unwrap();
                let index = subs.argument.generate_bytecode(generator)?.unwrap();

                generator.emit_set_subscript(src, base, index);

                Ok(Some(src))
            }
            _ => return Err(CodegenErr::SyntaxError {
                message: Some("invalid left hand side in assignment expression".to_string()),
                span: self.span
            })
        }
    }
}

impl<'ast> GeneratorNode for Literal<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        Ok(Some(match self.kind {
            LiteralKind::Null => Operand::null(),
            LiteralKind::Integer(int) => generator.make_int(int),
            LiteralKind::Float(float) => generator.make_float(float),
            LiteralKind::Boolean(bool) => Operand::bool(bool),
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
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        if generator.find_local(self.symbol).is_some() {
            return Ok(Some(generator.get_local_reg(self.symbol)));
        }

        let dst = generator.allocate_reg();
        generator.emit_get_global(self.symbol, dst);
        Ok(Some(dst))
    }
}

impl<'ast> BinaryExpr<'ast> {
    fn generate_as_jump(
        &self,
        true_target: CodeLabel,
        false_target: CodeLabel,
        generator: &mut BytecodeGenerator
    ) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for BinaryExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let result = generator.allocate_reg();

        Ok(Some(result))
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

