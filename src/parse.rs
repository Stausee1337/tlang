
use std::{marker::PhantomData, mem::transmute};

use crate::{lexer::{Token, Span}, symbol::Symbol};

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

#[derive(Debug, Copy, Clone)]
pub enum Statement<'ast> {
    If(IfBranch<'ast>),
    Break(Span),
    Return(Return<'ast>),
    Continue(Span),
    Import(Import<'ast>),
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
pub struct Return<'ast> {
    pub value: Option<&'ast Expression<'ast>>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Import<'ast> { 
    pub alias: Option<Ident>,
    pub file: &'ast str,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Variable<'ast> {
    pub name: Ident,
    pub init: Option<&'ast Expression<'ast>>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct Function<'ast> {
    pub params: &'ast [&'ast Ident],
    pub body: &'ast [&'ast Statement<'ast>],
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub enum Expression<'ast> {
    Assign(AssignExpr<'ast>),
    Literal(Literal<'ast>),
    Ident(Ident),
    BinaryExpr(BinaryExpr<'ast>),
    UnaryExpr(UnaryExpr<'ast>),
    CallExpr(CallExpr<'ast>),
    AttributeExpr(AttributeExpr<'ast>),
    SubscriptExpr(SubscriptExpr<'ast>),
    ListExpr(ListExpr<'ast>),
    ObjectExpr(ObjectExpr<'ast>),
    TupleExpr(TupleExpr<'ast>),
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
pub struct Literal<'ast> {
    pub kind: LiteralKind<'ast>,
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub enum LiteralKind<'ast> {
    Null,
    Integer(u64),
    Float(f64),
    Boolean(bool),
    String(&'ast str),
}

#[derive(Debug, Copy, Clone)]
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

#[derive(Debug, Copy, Clone)]
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

#[derive(Debug, Copy, Clone)]
pub struct ObjectExpr<'ast> {
    pub inits: &'ast [&'ast (Ident, Expression<'ast>)],
    pub span: Span
}

#[derive(Debug, Copy, Clone)]
pub struct TupleExpr<'ast> {
    pub items: &'ast [&'ast Expression<'ast>],
    pub span: Span
}

pub struct ParseContext<'ast> {
    nodes: bumpalo::Bump,
    _phantom: PhantomData<&'ast ()>
}

impl<'ast> ParseContext<'ast> {
    fn new() -> Self {
        Self {
            nodes: bumpalo::Bump::new(),
            _phantom: PhantomData::default()
        }
    }

    pub fn alloc<T>(&self, node: T) -> &'ast T {
        unsafe {
            transmute(self.nodes.alloc(node))
        }
    }

    pub fn slice<T>(&self, vec: Vec<T>) -> &'ast [T] {
        let slice = vec.into_boxed_slice();
        self.alloc(slice)
    }
}

pub fn parse<'source>(tokens: Box<[Token<'source>]>) -> Module<'source> {
    let mut ctx = ParseContext::new();
    internal::ModuleParser::new()
        .parse(&mut ctx, tokens
            .into_iter()
            .map(|tok| (tok.1.start, tok.0, tok.1.end)))
    .unwrap()
}


mod internal {
    include!(concat!(env!("OUT_DIR"), "/grammar.rs"));
}
