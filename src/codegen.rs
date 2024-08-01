
use crate::lexer::Span;
use crate::parse::{IfBranch, Break, Return, Continue, Import, ForLoop, WhileLoop, Variable, Function, AssignExpr, Literal, Ident, BinaryExpr, UnaryExpr, CallExpr, AttributeExpr, SubscriptExpr, ListExpr, ObjectExpr, TupleExpr, Lambda, Module, Statement, LiteralKind, Expression, BinaryOp, UnaryOp};

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
        let emit = match self.op {
            BinaryOp::Equal => BytecodeGenerator::emit_branch_eq::<Operand, Operand, CodeLabel, CodeLabel>,
            BinaryOp::NotEqual => BytecodeGenerator::emit_branch_ne,
            BinaryOp::GreaterThan => BytecodeGenerator::emit_branch_gt,
            BinaryOp::GreaterEqual => BytecodeGenerator::emit_branch_ge,
            BinaryOp::LessThan => BytecodeGenerator::emit_branch_lt,
            BinaryOp::LessEqual => BytecodeGenerator::emit_branch_le,

            BinaryOp::BooleanOr => {
                let start_block = generator.fork_block(true);
                let intm_target = generator.set_current_block(start_block);

                if let Some(result) = self.lhs.can_generate_as_jump(true_target, intm_target, generator) {
                    result?.unwrap();
                } else {
                    let condition = self.lhs.generate_bytecode(generator)?.unwrap();
                    generator.emit_branch_if(condition, true_target, intm_target)
                }

                generator.set_current_block(intm_target);
                if let Some(result) = self.rhs.can_generate_as_jump(true_target, false_target, generator) {
                    result?.unwrap();
                } else {
                    let condition = self.rhs.generate_bytecode(generator)?.unwrap();
                    generator.emit_branch_if(condition, true_target, false_target)
                }

                return Ok(None);
            }
            BinaryOp::BooleanAnd => {
                let start_block = generator.fork_block(true);
                let intm_target = generator.set_current_block(start_block);

                if let Some(result) = self.lhs.can_generate_as_jump(intm_target, false_target, generator) {
                    result?.unwrap();
                } else {
                    let condition = self.lhs.generate_bytecode(generator)?.unwrap();
                    generator.emit_branch_if(condition, intm_target, false_target)
                }

                generator.set_current_block(intm_target);
                if let Some(result) = self.rhs.can_generate_as_jump(true_target, false_target, generator) {
                    result?.unwrap();
                } else {
                    let condition = self.rhs.generate_bytecode(generator)?.unwrap();
                    generator.emit_branch_if(condition, true_target, false_target)
                }

                return Ok(None);
            }
            _ => unreachable!(),
        };

        let lhs = self.lhs.generate_bytecode(generator)?.unwrap();
        let rhs = self.rhs.generate_bytecode(generator)?.unwrap();

        emit(generator, lhs, rhs, true_target, false_target);

        Ok(None)
    }
}

impl<'ast> Expression<'ast> {
    fn can_generate_as_jump(
        &self,
        true_target: CodeLabel,
        false_target: CodeLabel,
        generator: &mut BytecodeGenerator
    ) -> Option<CodegenResult> {
        use BinaryOp::*;
        let Self::BinaryExpr(binary) = self else {
            return None;
        };

        match binary.op {
            Equal | NotEqual | GreaterThan |
            GreaterEqual | LessThan | LessEqual |
            BooleanOr | BooleanAnd => Some(binary.generate_as_jump(true_target, false_target, generator)),
            _ => None,
        }
    }
}

impl<'ast> GeneratorNode for BinaryExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let emit = match self.op {
            BinaryOp::Plus => BytecodeGenerator::emit_add::<Operand, Operand, Operand>,
            BinaryOp::Minus => BytecodeGenerator::emit_sub,
            BinaryOp::Mul => BytecodeGenerator::emit_mul,
            BinaryOp::Div => BytecodeGenerator::emit_div,
            BinaryOp::Mod => BytecodeGenerator::emit_mod,

            BinaryOp::ShiftLeft => BytecodeGenerator::emit_left_shift,
            BinaryOp::ShiftRight => BytecodeGenerator::emit_right_shift,

            BinaryOp::BitwiseAnd => BytecodeGenerator::emit_bitwise_and,
            BinaryOp::BitwiseOr => BytecodeGenerator::emit_bitwise_or,
            BinaryOp::BitwiseXor => BytecodeGenerator::emit_bitwise_xor,

            _ => {
                let dst = generator.allocate_reg();

                let start_block = generator.fork_block(false);
                let false_target = generator.set_current_block(start_block);
                generator.fork_block(true);
                let true_target  = generator.fork_block(true);
                let end_block = generator.set_current_block(start_block);

                self.generate_as_jump(true_target, false_target, generator)?;

                generator.set_current_block(false_target);
                generator.emit_mov(Operand::bool(false), dst);
                generator.emit_branch(end_block);

                generator.set_current_block(true_target);
                generator.emit_mov(Operand::bool(true), dst);
                generator.emit_fallthrough();

                generator.set_current_block(end_block);

                return Ok(Some(dst));
            }
        };

        let lhs = self.lhs.generate_bytecode(generator)?.unwrap();
        let rhs = self.lhs.generate_bytecode(generator)?.unwrap();

        let dst = generator.allocate_reg();
        emit(generator, lhs, rhs, dst);

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for UnaryExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let emit = match self.op {
            UnaryOp::Neg => BytecodeGenerator::emit_neg::<Operand, Operand>,
            UnaryOp::Not => BytecodeGenerator::emit_not,
            UnaryOp::Invert => BytecodeGenerator::emit_invert,
        };

        let src = self.base.generate_bytecode(generator)?.unwrap();
        let dst = generator.allocate_reg();

        emit(generator, src, dst);

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for CallExpr<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for AttributeExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let base = self.base.generate_bytecode(generator)?.unwrap();
        let dst = generator.allocate_reg();

        generator.emit_get_attribute(base, self.attr.symbol, dst);

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for SubscriptExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let base = self.base.generate_bytecode(generator)?.unwrap();
        let index = self.argument.generate_bytecode(generator)?.unwrap();
        let dst = generator.allocate_reg();

        generator.emit_get_subscript(base, index, dst);

        Ok(Some(dst))
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

