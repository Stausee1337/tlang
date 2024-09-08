use std::ops::Range;

use logos::{Logos, Lexer};

use crate::{symbol::Symbol, vm::{VM, Eternal}, tvalue::TString, memory::GCRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize
}

impl From<Range<usize>> for Span {
    fn from(Range { start, end }: Range<usize>) -> Self {
        Self {
            start,
            end
        }
    }
}

#[derive(Logos, Clone, Copy, Debug)]
#[logos(skip r"[ \n\t\f]+")]
#[logos(extras = Eternal<VM>)]
pub enum TokenKind { 
    #[regex(r"//[^\n]*")]
    Comment,

    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token(";")]
    Semicolon,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LCurly,
    #[token("}")]
    RCurly,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("=")]
    Assign,
    #[token("...")]
    DotDotDot,
    #[token("=>")]
    Arrow,
    #[token("^")]
    Circumflex,
    #[token("?")]
    Question,

    #[token("&")]
    Ampersand,
    #[token("|")]
    VBar,
    #[token("~")]
    Tilde,

    #[token("<<")]
    LDoubleChevron,
    #[token(">>")]
    RDoubleChevron,

    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,

    #[token("<")]
    LChevron,
    #[token("<=")]
    LChevronEq,
    #[token(">")]
    RChevron,
    #[token(">=")]
    RChevronEq,
    #[token("==")]
    DoubleEq,
    #[token("!=")]
    BangEq,

    #[token("&&")]
    DoubleAmpersand,
    #[token("||")]
    DoubleVBar,
    #[token("!")]
    Bang,

    #[regex(r"[^\d\W]\w*", make_symbol)]
    Name(Symbol),
    #[regex(r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)", |lex| lex.slice().parse().ok())]
    Intnumber(u64),
    #[regex(r"(([0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?|\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)", |lex| lex.slice().parse().ok())]
    Floatnumber(f64),
    #[regex("\"[^\\n\"\\\\]*(?:\\\\.[^\\n\"\\\\]*)*\"", |lex| TString::from_slice(&lex.extras, lex.slice()))]
    String(GCRef<TString>),

    #[token("const")]
    Const,
    #[token("var")]
    Var,
    #[token("export")]
    Export,
    #[token("import")]
    Import,
    #[token("def")]
    Def,
    #[token("while")]
    While,
    #[token("for")]
    For,
    #[token("in")]
    In,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("record")]
    Record,
    #[token("new")]
    New,
    #[token("return")]
    Return,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,

    #[token("true")]
    True,
    #[token("false")]
    False,
    #[token("null")]
    Null
}

#[derive(Debug, Clone, Copy)]
pub struct Token(pub TokenKind, pub Span);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntaxError(pub Span);

pub fn tokenize<'source>(vm: Eternal<VM>, source: &'source str) -> Result<Box<[Token]>, SyntaxError> {
    let mut lexer = TokenKind::lexer_with_extras(source, vm);
    let mut tokens = Vec::new();

    loop {
        let Some(token) = lexer.next() else {
            break;
        };
        let span = lexer.span().into();
        let Ok(mut token) = token else {
            return Err(SyntaxError(span));
        };
        match &mut token {
            TokenKind::Comment =>
                continue,
            TokenKind::String(s) => {
                let lit = snailquote::unescape(s.as_slice())
                    .map_err(|_| SyntaxError(span))?;
                *s = TString::from_slice(&lexer.extras, &lit);
            }
            _ => ()
        }
        tokens.push(Token(token, span));
    }

    Ok(tokens.into_boxed_slice())
}

fn make_symbol(lexer: &Lexer<TokenKind>) -> Symbol {
    lexer.extras.symbols().intern_slice(lexer.slice())
}
