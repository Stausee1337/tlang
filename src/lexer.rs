use std::ops::Range;

use logos::Logos;

use crate::symbol::Symbol;

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
#[logos(skip r"[ \t\f]+")]
pub enum TokenKind<'source> { 
    #[regex(r"//[^\n]*")]
    Comment,
    #[regex(r"\n")]
    Newline,

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

    #[regex(r"[^\d\W]\w*", |lex| Symbol::intern(lex.slice()))]
    Name(Symbol),
    #[regex(r"(?:0(?:_?0)*|[1-9](?:_?[0-9])*)", |lex| lex.slice().parse().ok())]
    Intnumber(u64),
    #[regex(r"(([0-9](?:_?[0-9])*\.(?:[0-9](?:_?[0-9])*)?|\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)", |lex| lex.slice().parse().ok())]
    Floatnumber(f64),
    #[regex("\"[^\\n\"\\\\]*(?:\\\\.[^\\n\"\\\\]*)*\"", |lex| lex.slice())]
    String(&'source str),

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
pub struct Token<'source>(pub TokenKind<'source>, pub Span);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SyntaxError(pub Span);

pub fn tokenize<'source>(source: &'source str) -> Result<Box<[Token<'source>]>, SyntaxError> {
    let mut lexer = TokenKind::lexer(source);
    let mut tokens = Vec::new();
    let mut prev_ignored = true;

    loop {
        let Some(token) = lexer.next() else {
            break;
        };
        let span = lexer.span().into();
        let Ok(mut token) = token else {
            return Err(SyntaxError(span));
        };
        match &mut token {
            TokenKind::Comment => {
                prev_ignored = true;
                continue;
            }
            TokenKind::Newline => {
                if prev_ignored {
                    prev_ignored = true;
                    continue;
                }
                prev_ignored = true;
            }
            TokenKind::String(s) => {
                snailquote::unescape(s)
                    .map_err(|_| SyntaxError(span))?;
                prev_ignored = false;
            }
            _ => {
                prev_ignored = false;
            }
        }
        tokens.push(Token(token, span));
    }

    Ok(tokens.into_boxed_slice())
}

