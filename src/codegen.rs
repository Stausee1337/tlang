use hashbrown::HashMap;
use hashbrown::hash_map::RawEntryMut;

use crate::lexer::Span;
use crate::memory::GCRef;
use crate::parse::{IfBranch, Break, Return, Continue, Import, ForLoop, WhileLoop, Variable, Function, AssignExpr, Literal, Ident, BinaryExpr, UnaryExpr, CallExpr, AttributeExpr, SubscriptExpr, ListExpr, ObjectExpr, Lambda, Module, Statement, LiteralKind, Expression, BinaryOp, UnaryOp, Record, RecordItem, NewExpr};

use crate::bytecode::{Operand, BytecodeGenerator, CodeLabel, RibKind};
use crate::symbol::Symbol;
use crate::tvalue::{TFunction, TString, Accessor, FunctionFlags};

#[derive(Debug)]
pub enum CodegenErr {
    SyntaxError {
        message: Option<String>,
        span: Span
    }
}

impl BytecodeGenerator {
    fn debug(&self, sym: Symbol) -> &str {
        self.vm().symbols.get(sym).as_slice()
    }
}

pub type CodegenResult = Result<Option<Operand>, CodegenErr>;

pub trait GeneratorNode {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult;
}

pub fn generate_module<'ast>(module: Module<'ast>, mut generator: BytecodeGenerator) -> Result<GCRef<TFunction>, CodegenErr> {
    generate_body(module.body, &mut generator)?;
    if !generator.is_terminated() {
        generator.emit_return(Operand::null());
    }
    Ok(generator.root_function())
}

fn generate_body<'ast>(body: &'ast [&'ast Statement], generator: &mut BytecodeGenerator) -> CodegenResult {
    for stmt in body {
        match stmt {
            Statement::Variable(ref var) if generator.find_rib(RibKind::Module, 1).is_none() => {
                for name in var.names {
                    generator.register_local(**name, var.constant)
                        .map_err(|()| CodegenErr::SyntaxError {
                            span: var.span,
                            message: Some(
                                format!("identifier `{}` has already been declared", generator.debug(name.symbol)))
                        })?;
                }
            }
            _ => ()
        }
    }

    for stmt in body {
        stmt.generate_bytecode(generator)?;
    }
    Ok(None)
}

fn generate_small_branch(src: CodeLabel, dst: CodeLabel, generator: &mut BytecodeGenerator) {
    generator.emit_branch(dst);
}

/// STATEMENTS

impl<'ast> GeneratorNode for IfBranch<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let Some(condition) = self.condition else { // handle an else { ... } branch
            debug_assert!(self.else_branch.is_none());

            return generator.with_rib(RibKind::If, |generator| generate_body(self.body, generator));
        };

        let start_block = generator.make_block();
        let true_target = generator.make_block();
        let false_target = generator.set_current_block(start_block);

        condition.generate_as_jump(true_target, false_target, generator)?;

        generator.set_current_block(true_target);
        generator.with_rib(RibKind::If, |generator| generate_body(self.body, generator))?;
        generator.set_current_block(true_target);
        let true_terminated = generator.is_terminated();

        generator.set_current_block(false_target);
        let mut false_terminated = true;
        if let Some(else_branch) = self.else_branch {
            else_branch.generate_bytecode(generator)?;
            let end_block = generator.set_current_block(false_target);
            false_terminated = generator.is_terminated();
            generator.set_current_block(end_block);
        }

        if !false_terminated {
            let mut end_block = generator.set_current_block(false_target);
            if end_block == false_target {
                generator.make_block();
                end_block = generator.set_current_block(false_target);
            }
            generate_small_branch(false_target, end_block, generator);
            generator.set_current_block(end_block);
        }

        if !true_terminated {
            let end_block = generator.set_current_block(true_target);
            generate_small_branch(true_target, end_block, generator);
            generator.set_current_block(end_block);
        }

        Ok(None)
    }
}

impl<'ast> GeneratorNode for Return<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let value = if let Some(value) = self.value {
            value.generate_bytecode(generator)?.unwrap()
        } else { Operand::null() };

        generator.emit_return(value);

        Ok(None)
    }
}

impl<'ast> GeneratorNode for Continue {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let Some(loop_rib) = generator.find_rib(RibKind::Loop, -1) else {
            return Err(CodegenErr::SyntaxError {
                message: Some("`continue` outside of loop".to_string()),
                span: self.0
            })
        };
        let (_continue, _break) = loop_rib.loop_ctx();
        generator.emit_branch(_continue);

        Ok(None)
    }
}

impl<'ast> GeneratorNode for Break {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let Some(loop_rib) = generator.find_rib(RibKind::Loop, -1) else {
            return Err(CodegenErr::SyntaxError {
                message: Some("`break` outside of loop".to_string()),
                span: self.0
            })
        };
        let (_continue, _break) = loop_rib.loop_ctx();
        generator.emit_branch(_break);

        Ok(None)
    }
}

impl GeneratorNode for Import {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

impl<'ast> GeneratorNode for ForLoop<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let start_block = generator.make_block();
        let forward_block = generator.make_block();
        let loop_body = generator.make_block();
        let end_block = generator.set_current_block(start_block);

        let iterable = self.iter.generate_bytecode(generator)?.unwrap();
        let iterator = generator.allocate_reg();
        generator.emit_get_iterator(iterator, iterable);

        generate_small_branch(start_block, forward_block, generator);
        generator.with_rib(RibKind::Loop, |generator| {
            generator.set_current_block(forward_block);
            generator.register_local(self.var, false).unwrap();
            let dst = generator.declare_local(self.var.symbol);
            generator.emit_next_iterator(dst, iterator, loop_body, end_block);

            generator.set_current_block(loop_body);

            generator.set_loop_ctx(forward_block, end_block);
            generate_body(self.body, generator)?;
            generator.emit_branch(forward_block);
            Ok(())
        })?;

        generator.set_current_block(end_block);

        Ok(None)
    }
}

impl<'ast> GeneratorNode for WhileLoop<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let start_block = generator.make_block();
        let condition_block = generator.make_block();
        let loop_body = generator.make_block();
        let end_block = generator.set_current_block(start_block);

        // TODO: optimize while (true) { ... } loop not to have a condition block
        generate_small_branch(start_block, condition_block, generator);
        generator.set_current_block(condition_block);
        self.condition.generate_as_jump(loop_body, end_block, generator)?;

        generator.set_current_block(loop_body);
        generator.with_rib(RibKind::Loop, |generator| {
            generator.set_loop_ctx(condition_block, end_block);
            generate_body(self.body, generator)?;
            generator.emit_branch(condition_block);
            Ok(())
        })?;

        generator.set_current_block(end_block);

        Ok(None)
    }
}

impl<'ast> Variable<'ast> {
    fn initialize_names<F>(&self, init: Operand, mut declare: F, generator: &mut BytecodeGenerator)
    where
        F: FnMut(Symbol, Operand, &mut BytecodeGenerator)
    {
        debug_assert!(self.names.len() > 0);
        if self.names.len() == 1 {
            declare(
                self.names[0].symbol,
                init,
                generator
            );
            return;
        }

        for (idx, name) in self.names.into_iter().enumerate() {
            let src = generator.allocate_reg();
            let idx_ = generator.make_int(idx as u64);
            generator.emit_get_subscript(src, init, idx_);
            declare(name.symbol, src, generator);
        }
    }
}

impl<'ast> GeneratorNode for Variable<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let init = if let Some(init) = self.init {
            init.generate_bytecode(generator)?.unwrap()
        } else if self.constant {
            let span = Span {
                start: self.names.first().unwrap().span.start,
                end: self.names.last().unwrap().span.end,
            };
            return Err(CodegenErr::SyntaxError {
                message: Some("missing const initializer".to_string()),
                span
            });
        } else {
            Operand::null()
        };

        if generator.find_rib(RibKind::Module, 1).is_some() { // declare (and init) global
            self.initialize_names(
                init, |symbol, init, generator| generator.emit_declare_global(symbol, init, self.constant), generator);
            return Ok(None);
        }

        self.initialize_names(
            init,
            |symbol, init, generator| {
                let dst = generator.declare_local(symbol);
                generator.emit_mov(dst, init);
            }, generator);

        Ok(None)
    }
}

impl<'ast> GeneratorNode for Function<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let (func, _scope) = generator.with_function(self.name, self.params, |generator| {
            generate_body(self.body, generator)?;
            if !generator.is_terminated() {
                generator.emit_return(Operand::null());
            }
            Ok(())
        })?;

        // FIXME: closures
        let mut module = func.module;
        if let Err(crate::vm::GlobalErr::Redeclared(..)) = module.set_global(self.name.symbol, func.into(), true) {
            generator.emit_error();
        }
        Ok(None)
    }
}

impl<'ast> GeneratorNode for Record<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        if !generator.find_rib(RibKind::Module, 1).is_some() {
            todo!("Function Local Record");
        }

        enum CGDeclaration<'ast> {
            Method(GCRef<TFunction>),
            OffsetProperty {
                init: Option<&'ast Expression<'ast>>,
                accessor: Accessor
            },
            Constant(&'ast Expression<'ast>)
        }

        let vm = generator.vm();
        let mut items: HashMap<GCRef<TString>, CGDeclaration> = HashMap::new();
        for &item in self.body {
            let (symbol, decl) = match item {
                RecordItem::Method(func) => {
                    let symbol = func.name.symbol;
                    let is_method = func.params
                        .first()
                        .map(|&ident| ident.symbol == Symbol![self])
                        .unwrap_or(false);

                    let (mut func, _scope) = generator.with_function(func.name, func.params, |generator| {
                        generate_body(func.body, generator)?;
                        if !generator.is_terminated() {
                            generator.emit_return(Operand::null());
                        }
                        Ok(())
                    })?;

                    if is_method {
                        func.flags |= FunctionFlags::METHOD;
                    }

                    (symbol, CGDeclaration::Method(func))
                }
                RecordItem::Property(prop) => {
                    let symbol = prop.name.symbol;
                    (symbol, CGDeclaration::OffsetProperty {
                        init: prop.init,
                        accessor: Accessor::GET | Accessor::SET
                    })
                }
                RecordItem::Constant(constant) => {
                    let symbol = constant.name.symbol;
                    (symbol, CGDeclaration::Constant(constant.init.unwrap()))
                }
            };

            let name = vm.symbols().get(symbol);
            let entry = items
                .raw_entry_mut()
                .from_hash(symbol.hash, |key| std::ptr::addr_eq(GCRef::as_ptr(*key), GCRef::as_ptr(name)));
            match entry {
                RawEntryMut::Occupied(mut entry) => {
                    entry.insert(decl);
                }
                RawEntryMut::Vacant(mut entry) => {
                    entry.insert_with_hasher(
                        symbol.hash, name, decl,
                        |key| vm.symbols().intern(*key).hash);
                }
            }
        }

        // FIXME: this API is horrible for generating any sort of at-runtime helper.
        // It shouldn't rely on the function or the paramters having names.
        let (func, _scope) = generator.with_function(self.name, &[&self.name], |generator| {
            let throw_away = generator.allocate_reg();
            let builder = generator.get_local_reg(self.name.symbol);

            let method_kind = generator.make_i32(0);
            let property_kind = generator.make_i32(1);
            let constant_kind = generator.make_i32(2);

            generator.grow_args_buffer(4);

            for (name, decl) in items {
                let name = generator.make_string(name);
                match decl {
                    CGDeclaration::Method(method) => {
                        // builder.declare(kind: 0, name, method)
                        let method = generator.descriptor(method.into());
                        let arguments = generator.allocate_list(vec![method_kind, name, method]);
                        generator.emit_method_call(throw_away, builder, Symbol![declare], arguments);
                    }
                    CGDeclaration::OffsetProperty { accessor, .. } => {
                        // builder.declare(kind: 1, name, access)
                        let accessor = generator.make_i32(accessor.bits() as i32);
                        let arguments = generator.allocate_list(vec![property_kind, name, accessor]);
                        generator.emit_method_call(throw_away, builder, Symbol![declare], arguments);
                    }
                    CGDeclaration::Constant(init) => {
                        // builder.declare(kind: 2, name, value)
                        let value = init.generate_bytecode(generator)?.unwrap();
                        let arguments = generator.allocate_list(vec![constant_kind, name, value]);
                        generator.emit_method_call(throw_away, builder, Symbol![declare], arguments);
                    }
                }
            }

            generator.emit_return(Operand::null());
            Ok(())
        })?;

        let base = if let Some(base) = self.base {
            base.generate_bytecode(generator)?.unwrap()
        } else {
            Operand::null()
        };

        let initializer = generator.descriptor(func.into());
        generator.emit_make_global_type(self.name.symbol, base, initializer);

        Ok(None)
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

                    generator.emit_mov(generator.get_local_reg(ident.symbol), src);
                    return Ok(Some(src));
                }

                generator.emit_set_global(ident.symbol, src);

                Ok(Some(src))
            }
            Expression::AttributeExpr(attr) => {
                let src = self.rhs.generate_bytecode(generator)?.unwrap();
                let base = attr.base.generate_bytecode(generator)?.unwrap();

                generator.emit_set_attribute(base, attr.attr.symbol, src);

                Ok(Some(src))
            }
            Expression::SubscriptExpr(subs) => {
                let src = self.rhs.generate_bytecode(generator)?.unwrap();
                let base = subs.base.generate_bytecode(generator)?.unwrap();
                let index = subs.argument.generate_bytecode(generator)?.unwrap();

                generator.emit_set_subscript(base, index, src);

                Ok(Some(src))
            }
            Expression::ListExpr(list) if list.valid_lhs() => {
                let base = self.rhs.generate_bytecode(generator)?.unwrap();

                for (idx, expr) in list.items.iter().enumerate() {
                    let Expression::Ident(ident) = expr else {
                        unreachable!()
                    };

                    let src = generator.allocate_reg();
                    let idx_ = generator.make_int(idx as u64);
                    generator.emit_get_subscript(src, base, idx_);

                    if let Some(local) = generator.find_local(ident.symbol) {
                        if local.constant {
                            generator.emit_error();
                            return Ok(Some(Operand::null()));
                        }

                        generator.emit_mov(generator.get_local_reg(ident.symbol), src);
                    } else {
                        generator.emit_set_global(ident.symbol, src);
                    }
                }

                Ok(Some(base))
            }
            _ => return Err(CodegenErr::SyntaxError {
                message: Some("invalid left hand side in assignment expression".to_string()),
                span: self.span
            })
        }
    }
}

impl GeneratorNode for Literal {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        Ok(Some(match self.kind {
            LiteralKind::Null => Operand::null(),
            LiteralKind::Integer(int) => generator.make_int(int),
            LiteralKind::Float(float) => generator.make_float(float),
            LiteralKind::Boolean(bool) => generator.make_bool(bool),
            LiteralKind::String(str) => generator.make_string(str),
        }))
    }
}

impl<'ast> GeneratorNode for Ident {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        if generator.find_local(self.symbol).is_some() {
            return Ok(Some(generator.get_local_reg(self.symbol)));
        }

        let dst = generator.allocate_reg();
        generator.emit_get_global(dst, self.symbol);
        Ok(Some(dst))
    }
}

impl<'ast> Expression<'ast> {
    fn generate_as_jump(
        &self,
        true_target: CodeLabel,
        false_target: CodeLabel,
        generator: &mut BytecodeGenerator
    ) -> CodegenResult {
        match self {
            Self::BinaryExpr(binary) if matches!(binary.op, BinaryOp::Equal | BinaryOp::NotEqual | 
                                                 BinaryOp::GreaterThan | BinaryOp::GreaterEqual | 
                                                 BinaryOp::LessThan | BinaryOp::LessEqual |
                                                 BinaryOp::BooleanOr | BinaryOp::BooleanAnd) =>
                binary.generate_bool(true_target, false_target, generator).map(|_| None),
            Self::UnaryExpr(unary) if unary.op == UnaryOp::Not =>
                unary.base.generate_as_jump(false_target, true_target, generator),
            _ => {
                let condition = self.generate_bytecode(generator)?.unwrap();
                generator.emit_branch_if(condition, true_target, false_target);
                Ok(None)
            }
        }  
    }
}

impl<'ast> BinaryExpr<'ast> {
    fn generate_bool(
        &self,
        true_target: CodeLabel,
        false_target: CodeLabel,
        generator: &mut BytecodeGenerator
    ) -> Result<(), CodegenErr> {
        let emit = match self.op {
            BinaryOp::Equal => BytecodeGenerator::emit_branch_eq,
            BinaryOp::NotEqual => BytecodeGenerator::emit_branch_ne,
            BinaryOp::GreaterThan => BytecodeGenerator::emit_branch_gt,
            BinaryOp::GreaterEqual => BytecodeGenerator::emit_branch_ge,
            BinaryOp::LessThan => BytecodeGenerator::emit_branch_lt,
            BinaryOp::LessEqual => BytecodeGenerator::emit_branch_le,

            BinaryOp::BooleanOr => {
                let start_block = generator.make_block();
                let intm_target = generator.set_current_block(start_block);

                self.lhs.generate_as_jump(true_target, intm_target, generator)?;

                generator.set_current_block(intm_target);
                self.rhs.generate_as_jump(true_target, false_target, generator)?;

                return Ok(());
            }
            BinaryOp::BooleanAnd => {
                let start_block = generator.make_block();
                let intm_target = generator.set_current_block(start_block);

                self.lhs.generate_as_jump(intm_target, false_target, generator)?;

                generator.set_current_block(intm_target);
                self.rhs.generate_as_jump(true_target, false_target, generator)?;

                return Ok(());
            }
            _ => panic!("non-bool operator in generate_bool"),
        };

        let lhs = self.lhs.generate_bytecode(generator)?.unwrap();
        let rhs = self.rhs.generate_bytecode(generator)?.unwrap();

        emit(generator, lhs, rhs, true_target, false_target);

        Ok(())
    }

    fn generate_nobool(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let emit = match self.op {
            BinaryOp::Plus => BytecodeGenerator::emit_add,
            BinaryOp::Minus => BytecodeGenerator::emit_sub,
            BinaryOp::Mul => BytecodeGenerator::emit_mul,
            BinaryOp::Div => BytecodeGenerator::emit_div,
            BinaryOp::Mod => BytecodeGenerator::emit_mod,

            BinaryOp::ShiftLeft => BytecodeGenerator::emit_left_shift,
            BinaryOp::ShiftRight => BytecodeGenerator::emit_right_shift,

            BinaryOp::BitwiseAnd => BytecodeGenerator::emit_bitwise_and,
            BinaryOp::BitwiseOr => BytecodeGenerator::emit_bitwise_or,
            BinaryOp::BitwiseXor => BytecodeGenerator::emit_bitwise_xor,

            _ => panic!("bool operator in generate_nobool")
        };

        let lhs = self.lhs.generate_bytecode(generator)?.unwrap();
        let rhs = self.rhs.generate_bytecode(generator)?.unwrap();

        let dst = generator.allocate_reg();
        emit(generator, dst, lhs, rhs);

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for BinaryExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        match self.op {
            BinaryOp::Plus | BinaryOp::Minus | BinaryOp::Mul |
            BinaryOp::Div | BinaryOp::Mod | BinaryOp::ShiftLeft |
            BinaryOp::ShiftRight | BinaryOp::BitwiseAnd | 
            BinaryOp::BitwiseOr | BinaryOp::BitwiseXor => self.generate_nobool(generator),
            _ => {
                let dst = generator.allocate_reg();

                let start_block = generator.make_block();
                let false_target = generator.make_block();
                let true_target  = generator.make_block();
                let end_block = generator.set_current_block(start_block);

                self.generate_bool(true_target, false_target, generator)?;

                generator.set_current_block(false_target);
                let false_ = generator.make_bool(false);
                generator.emit_mov(dst, false_);
                generator.emit_branch(end_block);

                generator.set_current_block(true_target);
                let true_ = generator.make_bool(true);
                generator.emit_mov(dst, true_);
                generator.emit_branch(end_block);

                generator.set_current_block(end_block);

                return Ok(Some(dst));
            }
        }
    }
}

impl<'ast> GeneratorNode for UnaryExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let emit = match self.op {
            UnaryOp::Neg => BytecodeGenerator::emit_neg,
            UnaryOp::Not => BytecodeGenerator::emit_not,
            UnaryOp::Invert => BytecodeGenerator::emit_invert,
        };

        let src = self.base.generate_bytecode(generator)?.unwrap();
        let dst = generator.allocate_reg();

        emit(generator, dst, src);

        Ok(Some(dst))
    }
}

impl<'ast> CallExpr<'ast> {
    fn generate_arguments(&self, generator: &mut BytecodeGenerator) -> Result<Vec<Operand>, CodegenErr> {
        let mut arguments = vec![];
        for arg in self.args {
            let arg = arg.generate_bytecode(generator)?.unwrap();
            arguments.push(arg);
        }
        Ok(arguments)
    }
}

impl<'ast> GeneratorNode for CallExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        match self.callable {
            Expression::AttributeExpr(AttributeExpr { attr, base, .. }) => {
                // generate method call
                let this = base.generate_bytecode(generator)?.unwrap();
                let arguments = self.generate_arguments(generator)?;
                generator.grow_args_buffer(self.args.len() + 1);

                let dst = generator.allocate_reg();
                let list = generator.allocate_list(arguments);
                generator.emit_method_call(dst, this, attr.symbol, list);
                return Ok(Some(dst));
            },
            _ => ()
        }
        let callee = self.callable.generate_bytecode(generator)?.unwrap();
        let arguments = self.generate_arguments(generator)?;
        generator.grow_args_buffer(self.args.len());

        let dst = generator.allocate_reg();
        let list = generator.allocate_list(arguments);
        generator.emit_call(dst, callee, list);
        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for NewExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let ty = self.expr.generate_bytecode(generator)?.unwrap();
        let dst = generator.allocate_reg();
        generator.emit_make_type_instance(dst, ty);

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for AttributeExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let base = self.base.generate_bytecode(generator)?.unwrap();
        let dst = generator.allocate_reg();

        generator.emit_get_attribute(dst, base, self.attr.symbol);

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for SubscriptExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let base = self.base.generate_bytecode(generator)?.unwrap();
        let index = self.argument.generate_bytecode(generator)?.unwrap();
        let dst = generator.allocate_reg();

        generator.emit_get_subscript(dst, base, index);

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for ListExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        if self.items.is_empty() {
            let dst = generator.allocate_reg();
            generator.emit_make_empty_list(dst);
            return Ok(Some(dst));
        }
        let mut ops = vec![];
        for item in self.items {
            ops.push(
                item.generate_bytecode(generator)?.unwrap()
            );
        }
        let dst = generator.allocate_reg();
        let items = generator.allocate_list(ops);
        generator.emit_make_list(dst, items);
        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for ObjectExpr<'ast> {
    fn generate_bytecode(&self, generator: &mut BytecodeGenerator) -> CodegenResult {
        let dst = generator.allocate_reg();
        generator.emit_make_anon_object(dst, self.inits.len() as u32);

        for (ident, init) in self.inits {
            let init = init.generate_bytecode(generator)?.unwrap();
            generator.emit_set_attribute(dst, ident.symbol, init);
        }

        Ok(Some(dst))
    }
}

impl<'ast> GeneratorNode for Lambda<'ast> {
    fn generate_bytecode(&self, _generator: &mut BytecodeGenerator) -> CodegenResult {
        todo!()
    }
}

