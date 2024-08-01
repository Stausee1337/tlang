use convert_case::{Casing, Case};
use proc_macro2::{TokenStream, Ident, Span};
use quote::quote;
use syn::{Fields, spanned::Spanned, ItemEnum, punctuated::Punctuated, Token, FieldsNamed, Expr, parse::{Parse, Parser}, Visibility, token::Brace, Attribute, Meta, MacroDelimiter, LitBool,};

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
    terminator: bool,
    ident: Ident,
    fields: Option<FieldsNamed>,
    discriminant: Option<(Token![=], Expr)>,
}

impl Parse for Instruction {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attrs = Attribute::parse_outer(input)?;
        let ident: Ident = input.parse()?;
        let fields: Option<FieldsNamed> = if input.peek(Brace) {
            Some(input.parse()?)
        } else { None };
        let assign_token: Option<Token![=]> = input.parse()?;

        let discriminant = if let Some(tok) = assign_token {
            Some((tok, input.parse()?))
        } else { None };

        let mut serializer = None;
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
                _ => continue,
            }
        }

        Ok(Self {
            serializer,
            terminator,
            ident,
            fields,
            discriminant
        })
    }
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

        opcodes.extend(quote!(#ident));
        if let Some((_, disc)) = &inst.discriminant {
            opcodes.extend(quote!(= #disc));
        }
        opcodes.extend(quote!(,));

        let fields = if let Some(fields) = &mut inst.fields {
            for field in &mut fields.named {
                field.vis = Visibility::Public(syn::token::Pub::default());
            }
            quote!(#fields)
        } else {
            quote!({})
        };

        let serializer = inst.serializer
            .as_ref()
            .map(|x| x.clone())
            .unwrap_or(quote!(BitSerializer<Self>));

        let terminator = inst.terminator;

        structures.extend(quote!(#[derive(Clone, Copy, Debug)] #[repr(packed)] pub struct #ident #fields));
        structures.extend(quote! {
            impl Instruction for #ident {
                const CODE: OpCode = OpCode::#ident;
                const IS_TERMINATOR: bool = #terminator;
                type Serializer = #serializer;
            }
        });
        
        let snake_case_name = format!("emit_{}", ident.to_string().to_case(Case::Snake));
        let snake_case_ident = Ident::new(&snake_case_name, Span::call_site());
        
        let mut sargs = TokenStream::new();
        let mut fargs = Punctuated::<Ident, Token![,]>::new();
        let mut params = TokenStream::new();
        let mut gparams = TokenStream::new();
        if let Some(fields) = &inst.fields {
            for (idx, field) in fields.named.iter().enumerate() { 
                let ident = field.ident.as_ref().unwrap();
                let ty = &field.ty;

                let gident = Ident::new(&format!("__T{idx}"), Span::call_site());

                fargs.push(ident.clone());
                sargs.extend(quote!(#ident: #ident.into(),));
                params.extend(quote!(#ident: #gident,));
                gparams.extend(quote!(#gident: Into<#ty>,));
            }
        }

        let terminator_code = if !inst.terminator {
            quote!(debug_assert!(!self.terminated);)
        } else {
            quote!(self.terminated = true;)
        };

        block_impls.extend(quote! {
            pub fn #snake_case_ident<#gparams>(&mut self, #params) {
                #terminator_code
                let instruction = crate::bytecode::instructions::#ident {
                    #sargs
                };
                Instruction::serialize(instruction, &mut self.data)
            }
        });

        codegen_impls.extend(quote! {
            pub fn #snake_case_ident<#gparams>(&mut self, #params) {
                let func = self.current_fn_mut();
                let block = func.current_block_mut();
                block.#snake_case_ident(#fargs);
            }
        });

        debug_deserialize.extend(quote! {
            OpCode::#ident =>
                Box::new(crate::bytecode::instructions::#ident::deserialize(stream).unwrap()),
        });
    }

    let module = quote! {
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub enum OpCode {
            #opcodes
        }

        impl OpCode {
            pub fn decode(op: u8) -> Self {
                unsafe { std::mem::transmute(op) }
            }

            pub fn deserialize_for_debug(self, stream: &mut CodeStream) -> Box<dyn std::fmt::Debug> {
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
            fn deserialize(stream: &mut CodeStream) -> Option<Self> {
                if stream.current() != Self::CODE as u8 {
                    return None;
                }
                stream.bump(1);

                let Some((inst, size)) = Self::Serializer::deserialize(stream.code()) else {
                    return None;
                };
                stream.bump(size);

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
