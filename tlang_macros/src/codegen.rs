use std::collections::HashMap;

use convert_case::{Casing, Case};
use proc_macro2::{TokenStream, Ident, Span};
use quote::{quote, ToTokens, quote_spanned, TokenStreamExt};
use syn::{Fields, spanned::Spanned, ItemEnum, punctuated::Punctuated, Token, FieldsNamed, Expr, parse::{Parse, Parser}, Visibility, token::Brace, Attribute, Meta, MacroDelimiter, LitBool, braced, PatIdent, Lifetime, Type, TypeReference, FieldValue, Member, ItemMod, Item, ItemStruct, LitStr, Path, PathArguments, GenericArgument, LitInt, ExprLit, Lit, FnArg};

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

        let mut terminator = false;
        for attr in attrs {
            let Meta::List(list) = attr.meta else {
                continue;
            };
            let Some(ident) = list.path.get_ident() else {
                continue;
            };
            match ident.to_string().as_ref() {
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

fn make_struct(
    vis: Visibility,
    ident: &Ident,
    generics: Option<&GenericLifetime>,
    fields: &TokenStream,
    is_terminator: bool,
    structures: &mut TokenStream) {

    let impl_generics = if let Some(lifetime) = &generics {
        let lifetime = &lifetime.param;
        quote! { <'de: #lifetime, #lifetime> }
    } else {
        quote! { <'de> }
    };

    structures.extend(quote!(
            #[derive(Clone, Copy, Debug)]
            #[repr(packed)] 
            #[allow(non_camel_case_types)]
            #vis struct #ident #generics #fields));
    structures.extend(quote! {
        impl #impl_generics Instruction<'de> for #ident #generics {
            const CODE: OpCode = OpCode::#ident;
            const IS_TERMINATOR: bool = #is_terminator;
        }

        impl #generics Serialize for #ident #generics {
            fn serialize(&self, _serializer: &mut Serializer) {
                todo!()
            }
        }

        impl #impl_generics Deserialize<'de> for #ident #generics {
            fn deserialize(_deserializer: &mut Deserializer<'de>) -> Option<Self> {
                todo!()
            }
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
        
        let mut fargs = Punctuated::<Ident, Token![,]>::new();
        let mut params = TokenStream::new();
        if let Some(fields) = &inst.fields {
            for field in fields.named.iter() { 
                let ident = field.ident.as_ref().unwrap();
                let ty = &field.ty;

                fargs.push(ident.clone());
                params.extend(quote!(#ident: #ty,));
            }
        }

        let terminator_code = if inst.terminator {
            Some(quote!(self.terminated = true;))
        } else { None };

        block_impls.extend(quote! {
            pub fn #snake_case_ident #generics (&mut self, #params) {
                debug_assert!(!self.terminated);
                #terminator_code
                let instruction = crate::bytecode::instructions::#ident {
                    #fargs
                };
                instruction.serialize(self.serializer());
            }
        });

        codegen_impls.extend(quote! {
            pub fn #snake_case_ident #generics (&mut self, #params) {
                let func = self.current_fn_mut();
                let block = func.current_block_mut();
                block.#snake_case_ident(#fargs);
            }
        });

        /*debug_deserialize.extend(quote! {
            OpCode::#ident =>
                Box::new(crate::bytecode::instructions::#ident::deserialize(stream).unwrap()),
        });*/

        let fields = if let Some(fields) = &mut inst.fields.clone() {
            for field in &mut fields.named {
                field.vis = Visibility::Public(syn::token::Pub::default());
            }
            quote!(#fields)
        } else {
            quote!({})
        };

        make_struct(
            Visibility::Public(syn::token::Pub::default()),
            ident, 
            generics.as_ref(),
            &fields,
            inst.terminator,
            &mut structures);
    }

    let module = quote! {
        #[repr(u8)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub enum OpCode {
            #opcodes
        }

        impl OpCode {
            pub fn deserialize_for_debug<'c>(self, stream: &'c mut CodeStream) -> Box<dyn std::fmt::Debug> {
                todo!()
            }
        }

        impl<'de> Deserialize<'de> for OpCode {
            fn deserialize(_deserializer: &mut Deserializer<'de>) -> Option<Self> {
                // Some(unsafe { std::mem::transmute(op) })
                todo!();
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
    deserializer: Expr,
    comma_token1: Token![,],
    environment: Expr,
    comma_token2: Token![,],
    decodable: Decodable,
}

impl Parse for DecodeStmt {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let deserializer = input.parse()?;
        let comma_token1 = input.parse()?;
        let environment = input.parse()?;
        let comma_token2 = input.parse()?;
        let decodable = input.parse()?;
        Ok(Self { deserializer, comma_token1, environment, comma_token2, decodable })
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
    let mut outputs = Punctuated::<Ident, Token![,]>::new();
    let mut locals = Punctuated::<PatIdent, Token![,]>::new();

    let ident = &stmt.decodable.ident;
    let environment = &stmt.environment;
    let deserializer = &stmt.deserializer;

    let mut decode_logic = TokenStream::new(); 
    for field in &stmt.decodable.fields.named {
        let ident = &field.ident;
        let local = PatIdent {
            attrs: vec![],
            ident: ident.clone(),
            by_ref: None,
            mutability: if let DecodeKind::Deref(..) = field.kind {
                Some(Token![mut](Span::call_site()))
            } else { None },
            subpat: None
        };
        
        let decoding = match field.kind {
            DecodeKind::Copy => quote!(Decode::decode(&#ident, #environment)),
            DecodeKind::Deref(_) => unreachable!(),
            DecodeKind::Mutable(_) => quote!(DecodeMut::decode_mut(&#ident, #environment)),
        };

        decode_logic.extend(quote!(let #local = #decoding; ));

        fields.push(ident.clone());
        outputs.push(ident.clone());
        locals.push(local);
    }

    decode_logic.extend(quote!((#outputs,)));

    Ok(quote! {
        let (#locals,) = {
            use std::ops::Deref;
            use crate::bytecode::Instruction;

            let instruction: crate::bytecode::instructions::#ident = (#deserializer).next().unwrap();
            let crate::bytecode::instructions::#ident { #fields, .. } = instruction;

            #decode_logic
        };
    })
}

pub fn generate_tobject(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let fields_parser = Punctuated::<FieldValue, Token![,]>::parse_terminated;
    let fields = fields_parser.parse2(token_stream)?;

    let mut code = quote! {
        let mut symbols = vm.symbols;
    };

    for field in fields {
        let ident = match &field.member {
            Member::Named(ident) => ident,
            Member::Unnamed(..) =>
                unreachable!()
        };

        let expr = &field.expr;

        code.extend(quote! {
            {
                let name = symbols.intern_slice(stringify!(#ident));
                let value = VMCast::vmcast(#expr, builder.vm);
                builder.insert_attribute(name, value);
            }
        });
    }

    let tokens = quote! {
        |builder| {
            #code
        }
    };
    Ok(quote!(#tokens))
}

struct ShadowArgs(&'static str, usize, bool);

impl ToTokens for ShadowArgs {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let base_ident = Ident::new(&format!("{}", self.0), Span::call_site());
        for idx in 0..self.1 {
            if idx != 0 {
                tokens.extend(quote!(,));
            }
            let ident = if self.2 {
                Ident::new(&format!("{}{idx}", self.0), Span::call_site())
            } else {
                base_ident.clone()
            };
            tokens.extend(quote!(#ident));
        }
    }
}

pub fn generate_tfunction(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    // let fields = fields_parser.parse2(token_stream)?;
    let argsparser = Punctuated::<Expr, Token![,]>::parse_terminated;
    let arguments = argsparser.parse2(token_stream)?;

    let module = arguments.get(0)
        .ok_or_else(|| syn::Error::new(Span::call_site(), "expected `module` argument"))?;

    let function = arguments.get(1)
        .ok_or_else(|| syn::Error::new(Span::call_site(), "expected `function` argument"))?;

    let count = arguments.get(2)
        .ok_or_else(|| syn::Error::new(Span::call_site(), "expected `count` argument"))?;

    let name = arguments.get(3)
        .ok_or_else(|| syn::Error::new(Span::call_site(), "expected `name` argument"))?;

    let modulefunc = arguments.get(4)
        .ok_or_else(|| syn::Error::new(Span::call_site(), "expected `modulefunc` argument"))?;

    let count = match count {
        Expr::Lit(ExprLit { lit: Lit::Int(litint), .. }) => litint.clone(),
        _ => {
            return Err(syn::Error::new(Span::call_site(), "expected `count` to be a literal int"));
        }
    };

    let modulefunc = match modulefunc {
        Expr::Lit(ExprLit { lit: Lit::Bool(litbool), .. }) => litbool.value(),
        _ => {
            return Err(syn::Error::new(Span::call_site(), "expected `modulefunc` to be a literal bool"));
        }
    };

    let modtoken = if modulefunc {
        Some(quote!( module, ))
    } else {
        None
    };

    let count: usize = count.base10_parse()?;

    let shadow_params = ShadowArgs("_", count + (modulefunc as usize), false);
    let fntype = quote!(fn(#shadow_params) -> _);

    let mut converters = TokenStream::new();
    for idx in 0..count {
        let arg = Ident::new(&format!("arg{idx}"), Span::call_site());
        converters.extend(quote! {
            let #arg = VMDowncast::vmdowncast(args[#idx], &vm).unwrap();
        });
    }

    let shadow_args = ShadowArgs("arg", count, true);

    let tokens = quote! {
        fn inner(module: crate::memory::GCRef<crate::vm::TModule>, args: &[TValue]) -> TValue {
            let vm = module.vm();
            let function: #fntype = #function as _;
            #converters
            VMCast::vmcast(function( #modtoken #shadow_args), &vm)
        }
        crate::tvalue::TFunction::nativefunc(#name, #module, inner as _)
    };

    Ok(quote!( { #tokens } ))
}

pub fn generate_ttype(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let fields_parser = Punctuated::<FieldValue, Token![,]>::parse_terminated;
    let fields = fields_parser.parse2(token_stream)?;

    let mut name = None;
    let mut modname = None;

    let mut remaining_fields = Punctuated::<FieldValue, Token![,]>::new();

    for field in fields {
        let ident = match &field.member {
            Member::Named(ident) => ident,
            Member::Unnamed(..) =>
                unreachable!()
        };
        match ident.to_string().as_str() {
            "name" => {
                name = Some(field.expr);
            },
            "modname" => {
                modname = Some(field.expr);
            },
            _ => {
                remaining_fields.push(field)
            }
        }
    }

    if name.is_none() || modname.is_none() {
        return Err(syn::Error::new(Span::call_site(), "`name` and `modname` are required fields"));
    }
    
    let tokens = quote! {
        pub struct InternalBuilder;
        impl InternalBuilder {
            pub fn build(&self, vm: &VM) -> GCRef<TType> {
                let mut builder = crate::tvalue::TTypeBuilder::new(vm, #name, #modname);
                builder.extend(tlang_macros::tobject! {
                    #remaining_fields
                });
                builder.build()
            }
        }
        InternalBuilder
    };
    Ok(quote!( { #tokens } ))
}

pub struct MyFieldValue {
    pub member_path: Punctuated<Ident, Token![.]>,

    pub colon_token: Option<Token![:]>,

    pub expr: Expr,
}

impl Parse for MyFieldValue {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let member_path = Punctuated::<Ident, Token![.]>::parse_separated_nonempty(input)?;
        let colon_token: Option<Token![:]> = if input.peek(Token![:]) {
            Some(input.parse()?)
        } else { None };
        
        let expr = if colon_token.is_some() {
            input.parse()?
        } else {
            syn::parse2(quote! { #member_path })?
        };

        Ok(Self {
            member_path,
            colon_token,
            expr
        })
    }
}

impl ToTokens for MyFieldValue {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.member_path.to_tokens(tokens);
        tokens.extend(quote!(:));
        self.expr.to_tokens(tokens);
    }
}

fn remove_first(member_path: Punctuated<Ident, Token![.]>) -> Punctuated<Ident, Token![.]> {
    let mut path = Punctuated::<Ident, Token![.]>::new();
    for idx in 1..member_path.len() {
        path.push(member_path.get(idx).unwrap().clone());
    }
    path
}

pub fn recursive_gen_tokens(ident: TokenStream, fields: Punctuated<MyFieldValue, Token![,]>) -> Result<TokenStream, syn::Error> {
    let mut this_fields = Punctuated::<MyFieldValue, Token![,]>::new();
    let mut base_fields = Punctuated::<MyFieldValue, Token![,]>::new();

    for mut field in fields {
        if field.member_path.first().map(|path| path.eq("base")).unwrap_or(true) && field.member_path.len() > 1 {
            field.member_path = remove_first(field.member_path);
            base_fields.push(field);
            continue;
        }
        this_fields.push(field);
    }

    if base_fields.is_empty() {
        return Ok(quote! {
            #ident {
                #this_fields
            }
        });
    }

    let base_tokens = recursive_gen_tokens(
        quote!(<#ident as tlang::tvalue::TPolymorphicObject>::Base),
        base_fields)?;

    Ok(quote! {
        {
            let base = #base_tokens;
            #ident {
                base,
                #this_fields
            }
        }
    })
}

pub fn generate_struct_init(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let fields_parser = Punctuated::<MyFieldValue, Token![,]>::parse_terminated;
    let fields = fields_parser.parse2(token_stream)?;

    let self_ident = Ident::new("Self", Span::call_site());

    Ok(recursive_gen_tokens(quote!(#self_ident), fields)?)
}

mod symbol {
    use ahash::RandomState;

    const HASH0: RandomState = RandomState::with_seeds(
        0x6735611e020820df,
        0x43abdc326e8e3a2c,
        0x80a2f5f207cf871e,
        0x43d1fac44245c038
    );

    const HASH1: RandomState = RandomState::with_seeds(
        0xb15ceb7218c4c1b5,
        0xae5bd30505b9c17e,
        0xb1c5d5e97339544a,
        0xedae3f360619d8f6
    );

    const HASH2: RandomState = RandomState::with_seeds(
        0x229533896b4d1f57,
        0x82d963a13dca2bb5,
        0x51a9b15708317482,
        0x382e6f20e2020ddf
    );

    const HASH3: RandomState = RandomState::with_seeds(
        0xd6cc5b074c253c8e,
        0x591a296cf00ad299,
        0x47848ccc54e51f03,
        0x29e940477b79df10
    );

    pub fn mkhash(str: &str) -> u64 {
        HASH0.hash_one(str)
    }

    pub fn mkid(str: &str) -> u64 {
        HASH1.hash_one(str) ^ HASH2.hash_one(str) ^ HASH3.hash_one(str)
    }
}

pub fn generate_symbol(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let name: Ident = syn::parse2(token_stream)?;
    let name = name.to_string();
    let hash = symbol::mkhash(&name);
    let id = symbol::mkid(&name);

    Ok(quote! {
        tlang::symbol::Symbol {
            id: #id,
            hash: #hash
        }
    })
}
