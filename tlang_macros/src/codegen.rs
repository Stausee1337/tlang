use convert_case::{Casing, Case};
use proc_macro2::{TokenStream, Ident, Span};
use quote::{quote, ToTokens};
use syn::{Fields, spanned::Spanned, ItemEnum, punctuated::Punctuated, Token, FieldsNamed, Expr, parse::{Parse, Parser}, Visibility, token::Brace, Attribute, Meta, MacroDelimiter, LitBool, braced, PatIdent, Lifetime, Type, TypeReference, FieldValue, Member};

pub fn generate_node(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let node: ItemEnum = syn::parse2(token_stream)?;
    let node_name = node.ident.clone();
    let generics = node.generics;

    let mut match_arms = TokenStream::new();
    for variant in &node.variants {
        let variant_name = variant.ident.clone(); 
        let works = match variant.fields {
            Fields::Unnamed(ref fields) if fields.unnamed.len() == 1 => true,
            _ => false
        };

        if !works {
            return Err(syn::Error::new(variant.span(), "Expected one unnamed field per variant"));
        }

        match_arms.extend(quote! {
            #node_name::#variant_name(ref node) =>
                crate::codegen::GeneratorNode::generate_bytecode(node, generator),
        });
    }

    let implementation = quote! {
        impl #generics crate::codegen::GeneratorNode for #node_name #generics {
            fn generate_bytecode(
                &self, generator: &mut crate::bytecode::BytecodeGenerator) -> crate::codegen::CodegenResult {
                match self {
                    #match_arms
                }
            }
        }
    };

    Ok(TokenStream::from(implementation))
}

pub struct Instruction {
    serializer: Option<TokenStream>,
    formatter: Option<TokenStream>,
    terminator: bool,
    ident: Ident,
    lifetime: Option<GenericLifetime>,
    fields: Option<FieldsNamed>,
    discriminant: Option<(Token![=], Expr)>,
}

impl Parse for Instruction {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attrs = Attribute::parse_outer(input)?;
        let ident: Ident = input.parse()?;
        let lifetime: Option<GenericLifetime> = if input.peek(Token![<]) {
            Some(input.parse()?)
        } else { None };
        let fields: Option<FieldsNamed> = if input.peek(Brace) {
            Some(input.parse()?)
        } else { None };
        let assign_token: Option<Token![=]> = input.parse()?;

        let discriminant = if let Some(tok) = assign_token {
            Some((tok, input.parse()?))
        } else { None };

        let mut serializer = None;
        let mut formatter = None;
        let mut terminator = false;
        for attr in attrs {
            let Meta::List(list) = attr.meta else {
                continue;
            };
            let Some(ident) = list.path.get_ident() else {
                continue;
            };
            match ident.to_string().as_ref() {
                "serializer" => {
                    let MacroDelimiter::Paren(..) = list.delimiter else {
                        return Err(
                            syn::Error::new(
                                list.delimiter.span().span(),
                                "expected parens `(...)` for #[serializer(...)]")
                        );
                    };
                    serializer = Some(list.tokens);
                    continue;
                }
                "terminator" => {
                    let MacroDelimiter::Paren(..) = list.delimiter else {
                        return Err(
                            syn::Error::new(
                                list.delimiter.span().span(),
                                "expected parens `(...)` for #[terminator(...)]")
                        );
                    };
                    let parser = LitBool::parse;
                    let lit_bool = parser.parse2(list.tokens)?;
                    terminator = lit_bool.value;
                }
                "formatter" => {
                    let MacroDelimiter::Paren(..) = list.delimiter else {
                        return Err(
                            syn::Error::new(
                                list.delimiter.span().span(),
                                "expected parens `(...)` for #[formatter(...)]")
                        );
                    };
                    formatter = Some(list.tokens);
                }
                _ => continue,
            }
        }

        Ok(Self {
            serializer,
            formatter,
            terminator,
            ident,
            lifetime,
            fields,
            discriminant
        })
    }
}

struct GenericLifetime { 
    pub lt_token: Token![<],
    pub param: Lifetime,
    pub gt_token: Token![>],
}

impl Parse for GenericLifetime {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        Ok(Self {
            lt_token: input.parse()?,
            param: input.parse()?,
            gt_token: input.parse()?
        })
    }
}

impl ToTokens for GenericLifetime {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.lt_token.to_tokens(tokens);
        self.param.to_tokens(tokens);
        self.gt_token.to_tokens(tokens);
    }
}

fn is_slice(ty: &Type) -> bool {
    match ty {
        Type::Reference(TypeReference { mutability: None, lifetime, elem, .. })
            if matches!(elem.as_ref(), Type::Slice(..)) => true,
        _ => false 
    }
}

fn make_struct(
    vis: Visibility,
    ident: &Ident,
    opcode: &Ident,
    generics: &Option<GenericLifetime>,
    fields: &TokenStream,
    serializer: Option<&TokenStream>,
    is_terminator: bool,
    has_formatter: bool,
    structures: &mut TokenStream) {
    let serializer = serializer
        .map(|x| x.clone())
        .unwrap_or(quote!(BitSerializer));

    let debug = if !has_formatter {
        Some(quote!(Debug))
    } else {
        None
    };

    structures.extend(quote!(
            #[derive(Clone, Copy, #debug)]
            #[repr(packed)] 
            #[allow(non_camel_case_types)]
            #vis struct #ident #generics #fields));
    structures.extend(quote! {
        impl #generics Instruction for #ident #generics {
            const CODE: OpCode = OpCode::#opcode;
            const IS_TERMINATOR: bool = #is_terminator;
            type Serializer = #serializer;
        }
    });
}

pub fn generate_instructions(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let parser = Punctuated::<Instruction, Token![,]>::parse_terminated;
    let mut instructions = parser.parse2(token_stream)?;

    let mut opcodes = TokenStream::new();
    let mut structures = TokenStream::new();
    let mut codegen_impls = TokenStream::new();
    let mut block_impls = TokenStream::new();
    let mut debug_deserialize = TokenStream::new();
    for inst in &mut instructions {
        let ident = &inst.ident;
        let generics = &inst.lifetime;

        opcodes.extend(quote!(#ident));
        if let Some((_, disc)) = &inst.discriminant {
            opcodes.extend(quote!(= #disc));
        }
        opcodes.extend(quote!(,));

        let snake_case_name = format!("emit_{}", ident.to_string().to_case(Case::Snake));
        let snake_case_ident = Ident::new(&snake_case_name, Span::call_site());
        
        let mut has_slice = false;
        let mut fargs = Punctuated::<Ident, Token![,]>::new();
        let mut params = TokenStream::new();
        if let Some(fields) = &inst.fields {
            for field in fields.named.iter() { 
                let ident = field.ident.as_ref().unwrap();
                let ty = &field.ty;

                has_slice |= is_slice(&field.ty);

                fargs.push(ident.clone());
                params.extend(quote!(#ident: #ty,));
            }
        }

        let terminator_code = if inst.terminator {
            Some(quote!(self.terminated = true;))
        } else { None };

        let sident = if !has_slice {
            ident.clone()
        } else {
            Ident::new(&format!("{}_{}", ident, 1), Span::call_site())
        };

        block_impls.extend(quote! {
            pub fn #snake_case_ident #generics (&mut self, #params) {
                debug_assert!(!self.terminated);
                #terminator_code
                let instruction = crate::bytecode::instructions::#sident {
                    #fargs
                };
                Instruction::serialize(instruction, &mut self.data)
            }
        });

        codegen_impls.extend(quote! {
            pub fn #snake_case_ident #generics (&mut self, #params) {
                let func = self.current_fn_mut();
                let block = func.current_block_mut();
                block.#snake_case_ident(#fargs);
            }
        });

        debug_deserialize.extend(quote! {
            OpCode::#ident =>
                crate::bytecode::instructions::#ident::deserialize(stream).unwrap(),
        });

        let mut has_slice = false;
        let fields = if let Some(fields) = &mut inst.fields.clone() {
            for field in &mut fields.named {
                if is_slice(&field.ty) {
                    has_slice = true;
                    field.ty = syn::parse2(quote!([(); 0])).unwrap();
                    field.vis = syn::parse2(quote!( pub(super) )).unwrap();
                    continue;
                }
                field.vis = Visibility::Public(syn::token::Pub::default());
            }
            quote!(#fields)
        } else {
            quote!({})
        };

        make_struct(
            Visibility::Public(syn::token::Pub::default()),
            ident, 
            ident, 
            &None,
            &fields, 
            inst.serializer.as_ref(),
            inst.terminator,
            inst.formatter.is_some(),
            &mut structures);

        if let Some(formatter) = &inst.formatter {
            structures.extend(quote! {
                impl std::fmt::Debug for #ident {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        #formatter(self, f)
                    }
                }
            });
        }

        if has_slice {
            let fields = inst.fields.as_mut().unwrap();
            for field in &mut fields.named {
                field.vis = syn::parse2(quote!( pub(super) )).unwrap();
            }

            let fields = quote!(#fields);

            let ext_ident = Ident::new(&format!("{}_{}", ident, 1), Span::call_site());

            make_struct(
                syn::parse2(quote!( pub(super) )).unwrap(),
                &ext_ident, 
                ident,
                generics,
                &fields, 
                inst.serializer.as_ref(),
                inst.terminator,
                true,
                &mut structures);
        }
    }

    let module = quote! {
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum OpCode {
            #opcodes
        }

        impl OpCode {
            pub fn decode(op: u8) -> Self {
                unsafe { std::mem::transmute(op) }
            }

            pub fn deserialize_for_debug<'c>(self, stream: &'c mut CodeStream) -> &'c dyn std::fmt::Debug {
                match self {
                    #debug_deserialize
                }
            }
        }

        pub trait Instruction: Sized + Copy {
            const CODE: OpCode;
            const IS_TERMINATOR: bool;
            type Serializer: InstructionSerializer<Self>;

            #[inline(always)]
            fn serialize(self, vec: &mut Vec<u8>) {
                vec.push(Self::CODE as u8);
                Self::Serializer::serialize(self, vec);
            }

            #[inline(always)]
            fn deserialize<'c>(stream: &'c mut CodeStream) -> Option<&'c Self> {
                if stream.current() != Self::CODE as u8 {
                    return None;
                }
                stream.bump(1);

                let Some(inst) = Self::Serializer::deserialize(stream) else {
                    return None;
                };

                Some(inst)
            }
        }

        pub mod instructions {
            use super::*;
            #structures
        }
    };

    let implementations = quote! {
        impl BasicBlock {
            #block_impls
        }

        impl BytecodeGenerator {
            #codegen_impls
        }
    };

    Ok(quote! {
        #module
        #implementations
    })
}

struct DecodeStmt {
    decodable: Decodable,
    in_token: Token![in],
    environment: Ident
}

impl Parse for DecodeStmt {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let decodable = input.parse()?;
        let in_token = input.parse()?;
        let environment = input.parse()?;
        Ok(Self { decodable, in_token, environment })
    }
}

struct Decodable {
    ident: Ident,
    fields: DecodeFields
}

impl Parse for Decodable {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        let fields = input.parse()?;
        Ok(Self { ident, fields })
    } }

struct DecodeFields {
    brace_token: Brace,
    named: Punctuated<DecodeField, Token![,]>,
}

impl Parse for DecodeFields {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        let brace_token = braced!(content in input);

        let named = content.parse_terminated(DecodeField::parse, Token![,])?;
        Ok(Self { brace_token, named })
    }
}

struct DecodeField {
    kind: DecodeKind,
    ident: Ident
}

impl Parse for DecodeField {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let kind = input.parse()?;
        let ident = input.parse()?;
        Ok(Self { kind, ident })
    }
}

enum DecodeKind {
    Copy,
    Deref(Token![&]),
    Mutable(Token![mut]),
}

impl Parse for DecodeKind {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(Token![&]) {
            Ok(DecodeKind::Deref(input.parse()?))
        } else if input.peek(Token![mut]) {
            Ok(DecodeKind::Mutable(input.parse()?))
        } else {
            Ok(DecodeKind::Copy)
        }
    }
}

pub fn generate_decode(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let stmt: DecodeStmt = syn::parse2(token_stream)?;

    let mut fields = Punctuated::<Ident, Token![,]>::new();
    let mut locals = Punctuated::<PatIdent, Token![,]>::new();

    let ident = &stmt.decodable.ident;
    let environment = &stmt.environment;

    let mut decode_logic = TokenStream::new(); 
    for field in &stmt.decodable.fields.named {
        let ident = &field.ident;
        let local = PatIdent {
            attrs: vec![],
            ident: ident.clone(),
            by_ref: None,
            mutability: None,
            subpat: None
        };
        
        let decoding = match field.kind {
            DecodeKind::Copy => quote!(Decode::decode(&#ident, #environment)),
            DecodeKind::Deref(_) => quote!(DecodeDeref::decode_deref(&#ident, unsafe { &*codenv })),
            DecodeKind::Mutable(_) => quote!(DecodeMut::decode_mut(&#ident, #environment)),
        };

        decode_logic.extend(quote!(let #local = #decoding; ));

        fields.push(ident.clone());
        locals.push(local);
    }

    decode_logic.extend(quote!((#fields)));

    Ok(quote! {
        let (#locals) = {
            use std::ops::Deref;
            use crate::bytecode::Instruction;

            let codenv: *mut _ = &mut *#environment;

            let instruction = crate::bytecode::instructions::#ident::deserialize(
                &mut #environment.stream
            ).unwrap();


            let crate::bytecode::instructions::#ident { #fields, .. } = *instruction;

            #decode_logic
        };
    })
}

pub fn generate_ttype(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let fields_parser = Punctuated::<FieldValue, Token![,]>::parse_terminated;
    let fields = fields_parser.parse2(token_stream)?;

    let mut code = quote! {
        let mut symbols = vm.symbols;
    };
    for field in &fields {
        let ident = match &field.member {
            Member::Named(ident) => ident,
            Member::Unnamed(..) =>
                unreachable!()
        };

        let expr = &field.expr;

        code.extend(quote! {
            {
                let name = symbols.intern_slice(stringify!(#ident));
                let value = #expr;
            }
        });
    }

    let tokens = quote! {
        struct TypeBuilder<F>(F);

        impl<F: Fn(&crate::vm::VM)> TypeBuilder<F> {
            pub fn build(&self, vm: &crate::vm::VM) {
                (self.0)(vm)
            }
        }

        TypeBuilder(|vm: &crate::vm::VM| {
            #code
        })
    };
    Ok(quote!( { #tokens } ))
}
