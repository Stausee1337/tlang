use convert_case::{Casing, Case};
use proc_macro2::{TokenStream, Ident, Span};
use quote::quote;
use syn::{Fields, spanned::Spanned, ItemEnum, punctuated::Punctuated, Token, FieldsNamed, Expr, parse::{Parse, Parser}, Visibility, token::Brace,};

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
    ident: Ident,
    fields: Option<FieldsNamed>,
    discriminant: Option<(Token![=], Expr)>,
}

impl Parse for Instruction {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let ident: Ident = input.parse()?;
        let fields: Option<FieldsNamed> = if input.peek(Brace) {
            Some(input.parse()?)
        } else { None };
        let assign_token: Option<Token![=]> = input.parse()?;

        let discriminant = if let Some(tok) = assign_token {
            Some((tok, input.parse()?))
        } else { None };

        Ok(Self {
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

        structures.extend(quote!(#[derive(Clone, Copy)] #[repr(C)] pub struct #ident #fields));
        structures.extend(quote! {
            impl Instruction for #ident {
                const CODE: OpCode = OpCode::#ident;
            }
        });
        
        let snake_case_name = format!("emit_{}", ident.to_string().to_case(Case::Snake));
        let snake_case_ident = Ident::new(&snake_case_name, Span::call_site());
        
        let mut names = Punctuated::<Ident, Token![,]>::new();
        let mut arguments = TokenStream::new();
        if let Some(fields) = &inst.fields {
            for field in &fields.named {
                let ident = field.ident.as_ref().unwrap();
                let ty = &field.ty;
                names.push(ident.clone());
                arguments.extend(quote!(#ident: #ty,));
            }
        }

        block_impls.extend(quote! {
            pub fn #snake_case_ident(&mut self, #arguments) {
                let instruction = crate::bytecode::instructions::#ident {
                    #names
                };
                Instruction::serialize(&instruction, &mut self.data)
            }
        });

        codegen_impls.extend(quote! {
            pub fn #snake_case_ident(&mut self, #arguments) {
                let scope = self.current_scope();
                let block = scope.current_block();
                block.#snake_case_ident(#names);
            }
        });
    }

    let module = quote! {
        pub mod instructions {
            use super::*;

            #[repr(u32)]
            pub enum OpCode {
                #opcodes
            }

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
