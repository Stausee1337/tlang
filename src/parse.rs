use std::{fmt::Debug, rc::Rc};

use lalrpop_util::ParseError;

use crate::{lexer::{Token, Span, SyntaxError, tokenize, self}, symbol::Symbol, memory::GCRef, tvalue::TString, vm::Eternal};
use tlang_macros::GeneratorNode;

#[derive(Debug, Copy, Clone)]
pub struct Ident {
    pub symbol: Symbol,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Module<'ast> {
    pub body: &'ast [&'ast Statement<'ast>],
    pub span: Span,
}

#[derive(Debug, Copy, Clone, GeneratorNode)]
pub enum Statement<'ast> {
    If(IfBranch<'ast>),
    Break(Break),
    Return(Return<'ast>),
    Continue(Continue),
    Import(Import),
    ForLoop(ForLoop<'ast>),
    WhileLoop(WhileLoop<'ast>),
    Variable(Variable<'ast>),
    Function(Function<'ast>),
    Expression(Expression<'ast>),
}

#[derive(Debug, Copy, Clone)]
pub struct IfBranch<'ast> {
    pub body: &'ast [&'ast Statement<'ast>],
    pub condition: Option<&'ast Expression<'ast>>,
    pub else_branch: Option<&'ast IfBranch<'ast>>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct ForLoop<'ast> {
    pub var: Ident,
    pub iter: &'ast Expression<'ast>,
    pub body: &'ast [&'ast Statement<'ast>],
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct WhileLoop<'ast> {
    pub condition: &'ast Expression<'ast>,
    pub body: &'ast [&'ast Statement<'ast>],
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Break(pub Span);

#[derive(Debug, Copy, Clone)]
pub struct Continue(pub Span);

#[derive(Debug, Copy, Clone)]
pub struct Return<'ast> {
    pub value: Option<&'ast Expression<'ast>>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Import { 
    pub alias: Option<Ident>,
    pub file: GCRef<TString>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Variable<'ast> {
    pub names: &'ast [&'ast Ident],
    pub constant: bool,
    pub init: Option<&'ast Expression<'ast>>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Function<'ast> {
    pub name: Ident,
    pub params: &'ast [&'ast Ident],
    pub body: &'ast [&'ast Statement<'ast>],
    pub span: Span
}

#[derive(Debug, Copy, Clone, GeneratorNode)]
pub enum Expression<'ast> {
    Assign(AssignExpr<'ast>),
    Literal(Literal),
    Ident(Ident),
    BinaryExpr(BinaryExpr<'ast>),
    UnaryExpr(UnaryExpr<'ast>),
    CallExpr(CallExpr<'ast>),
    AttributeExpr(AttributeExpr<'ast>),
    SubscriptExpr(SubscriptExpr<'ast>),
    ListExpr(ListExpr<'ast>),
    ObjectExpr(ObjectExpr<'ast>),
    Lambda(Lambda<'ast>),
}

#[derive(Debug, Copy, Clone)]
pub enum AssignOp {
    Equal,
}

#[derive(Debug, Copy, Clone)]
pub struct AssignExpr<'ast> {
    pub op: AssignOp,
    pub lhs: &'ast Expression<'ast>,
    pub rhs: &'ast Expression<'ast>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Literal {
    pub kind: LiteralKind,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub enum LiteralKind {
    Null,
    Integer(u64),
    Float(f64),
    Boolean(bool),
    String(GCRef<TString>),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryOp {
    Plus, Minus, Mul, Div, Mod,
    ShiftLeft, ShiftRight,
    BitwiseAnd, BitwiseOr, BitwiseXor,
    Equal, NotEqual, GreaterThan, GreaterEqual, LessThan, LessEqual,
    BooleanAnd, BooleanOr
}

#[derive(Debug, Copy, Clone)]
pub struct BinaryExpr<'ast> {
    pub op: BinaryOp,
    pub lhs: &'ast Expression<'ast>,
    pub rhs: &'ast Expression<'ast>,
    pub span: Span,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Invert, Not, Neg
}

#[derive(Debug, Copy, Clone)]
pub struct UnaryExpr<'ast> {
    pub op: UnaryOp,
    pub base: &'ast Expression<'ast>,
    pub span: Span,
}

#[derive(Debug, Copy, Clone)]
pub struct CallExpr<'ast> {
    pub callable: &'ast Expression<'ast>,
    pub args: &'ast [&'ast Expression<'ast>],
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct AttributeExpr<'ast> {
    pub attr: Ident,
    pub base: &'ast Expression<'ast>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct SubscriptExpr<'ast> {
    pub base: &'ast Expression<'ast>,
    pub argument: &'ast Expression<'ast>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct ListExpr<'ast> {
    pub items: &'ast [&'ast Expression<'ast>],
    pub span: Span
}

impl<'ast> ListExpr<'ast> {
    pub fn valid_lhs(&self) -> bool {
        for item in self.items {
            let Expression::Ident(..) = item else {
                return false;
            };
        }
        return !self.items.is_empty();
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ObjectExpr<'ast> {
    pub inits: &'ast [&'ast (Ident, Expression<'ast>)],
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Lambda<'ast> {
    pub body: &'ast [&'ast Statement<'ast>],
    pub params: &'ast [&'ast Ident],
    pub span: Span
}

pub struct ParseContext<'ast> {
    nodes: &'ast bumpalo::Bump,
    tokens: Box<[Token]>
}

impl<'ast> ParseContext<'ast> {
    pub fn new(tokens: Box<[Token]>, arena: &'ast bumpalo::Bump) -> Self {
        Self {
            tokens,
            nodes: arena
        }
    }

    pub fn parse(&self) -> Result<Module<'ast>, SyntaxError> {
        internal::ModuleParser::new()
            .parse(self, self.tokens
                .iter()
                .map(|tok| (tok.1.start, tok.0, tok.1.end)))
            .map_err(|err| SyntaxError(as_span(err)))
    }

    pub fn alloc<T>(&self, node: T) -> &'ast T {
        self.nodes.alloc(node)
    }

    pub fn slice<T>(&self, vec: Vec<T>) -> &'ast [T] {
        let slice = vec.into_boxed_slice();
        self.alloc(slice)
    }

    pub fn to_params(&self, exprs: &'ast [&'ast Expression]) -> Option<&'ast [&'ast Ident]> {
        let mut idents = Vec::new();
        for expr in exprs {
            match expr {
                Expression::Ident(ident) =>
                    idents.push(ident),
                _ => return None
            }
        }

        Some(self.slice(idents))
    }
}

fn as_span<T: Debug, E: Debug>(error: ParseError<usize, T, E>) -> Span {
    if cfg!(debug_assertions) {
        eprintln!("Syntax Error Information: {error:?}");
    }
    match error {
        ParseError::ExtraToken { token: (start, _, end) } =>
            Span { start, end },
        ParseError::InvalidToken { location } =>
            Span { start: location, end: location },
        ParseError::UnrecognizedEof { location, .. } =>
            Span { start: location, end: location },
        ParseError::UnrecognizedToken { token: (start, _, end), .. } => 
            Span { start, end },
        ParseError::User { .. } =>
            unreachable!()
    }
}

pub fn parse_from_tstring<'ast>(
    vm: Eternal<crate::vm::VM>,
    source: GCRef<TString>,
    arena: &'ast bumpalo::Bump,
) -> Result<Module<'ast>, SyntaxError> {
    let tokens = lexer::tokenize(vm, source.as_slice())?;
    let ctx = ParseContext::new(tokens, arena);
    ctx.parse()
}

mod internal {
    include!(concat!(env!("OUT_DIR"), "/grammar.rs"));
}
