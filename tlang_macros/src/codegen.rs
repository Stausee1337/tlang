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

enum DeclaredKind {
    Primitive,
    Record,
}

struct TypeDeclared {
    display_name: LitStr,
    item: ItemStruct,
    kind: DeclaredKind,
    impls_into: bool,
    impls_downcast: bool,
    fields: Punctuated<FieldValue, Token![,]>
}

fn generic_path_with_argument<'l>(path: &'l Path, name: &'static str) -> Option<&'l Ident> {
    let segment = path.segments.last()?;
    if segment.ident.to_string() != name {
        return None;
    }
    let args = match &segment.arguments {
        PathArguments::AngleBracketed(arguments) => arguments,
        _ => return None,
    };
    if args.args.len() != 1 {
        return None;
    }
    let first = args.args.first().unwrap();
    let ty = match first {
        GenericArgument::Type(ty) => ty,
        _ => return None,
    };
    match ty {
        Type::Path(path) => path.path.get_ident(),
        _ => None
    }
}

pub fn generate_tmodule(
    attr: TokenStream,
    token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let modname: LitStr = syn::parse2(attr)?;
    let mut module: ItemMod = syn::parse2(token_stream)?;

    let Some(content) = module.content.as_mut() else {
        return Err(syn::Error::new(Span::call_site(), "expected module with body"));
    };

    let mut types = HashMap::new();
    let mut types_names = Vec::new();

    for item in &mut content.1 {
        let (ident, strc) = match item {
            Item::Struct(s) => (s.ident.to_string(), s),
            _ => continue,
        };
        let mut display_name: Option<LitStr> = None;
        let mut kind: Option<DeclaredKind> = None;
        let mut index = 0;

        for (idx, attr) in strc.attrs.iter().enumerate() {
            index = idx;

            let Meta::List(list) = &attr.meta else {
                continue;
            };
            let Some(ident) = list.path.get_ident() else {
                continue;
            };
            kind = Some(match ident.to_string().as_str() {
                "primitive" => DeclaredKind::Primitive,
                "record" => DeclaredKind::Record,
                _ => continue
            });
            display_name = Some(syn::parse2(list.tokens.clone())?);
            break;
        }
        
        let Some((kind, display_name)) = kind.zip(display_name) else {
            continue;
        };

        strc.attrs.remove(index);
        if let DeclaredKind::Primitive = kind {
            let repr_parser = Attribute::parse_outer;
            let repr = repr_parser.parse2(quote!(#[repr(C)])).unwrap();
            strc.attrs.extend(repr);
        }

        types_names.push(ident.clone());
        types.insert(ident, TypeDeclared {
            display_name,
            kind,
            item: strc.clone(),
            impls_into: false,
            impls_downcast: false,
            fields: Punctuated::<FieldValue, Token![,]>::new()
        });
    }

    for item in &mut content.1 {
        let implem = match item {
            Item::Impl(s) => s,
            _ => continue,
        };

        let (ident, is_ref) = match implem.self_ty.as_ref() {
            Type::Path(path) => {
                if let Some(ident) = path.path.get_ident() {
                    (ident, false)
                } else if let Some(ident) = generic_path_with_argument(&path.path, "GCRef") {
                    (ident, true)
                } else {
                    continue;
                }
            },
            _ => continue
        };

        let Some(declared) = types.get_mut(&ident.to_string()) else {
            continue;
        };

        if let Some((_, path, _)) = &implem.trait_ {
            if let Some(ident) = generic_path_with_argument(&path, "Into") {
                if ident.eq("TValue") {
                    declared.impls_into = true;
                }
            } else if let Some(ident) = path.get_ident() {
                if ident.eq("VMDowncast") {
                    declared.impls_downcast = true;
                }
            }
        }

        let mut index = 0;
        let mut vmimpl = false;
        for (idx, attr) in implem.attrs.iter().enumerate() {
            index = idx;
            match &attr.meta {
                Meta::Path(path) => {
                    if let Some(ident) = path.get_ident() {
                        if ident.eq("vmimpl") {
                            vmimpl = true;
                            break;
                        }
                    }
                },
                _ => continue
            }
        }
        if !vmimpl {
            continue;
        }
        implem.attrs.remove(index);
    }

    let mut modulefuncs = TokenStream::new();

    for item in &mut content.1 {
        let func = match item {
            Item::Fn(s) => s,
            _ => continue,
        };

        let mut index = 0;
        let mut vmimpl = false;
        for (idx, attr) in func.attrs.iter().enumerate() {
            index = idx;
            match &attr.meta {
                Meta::Path(path) => {
                    if let Some(ident) = path.get_ident() {
                        if ident.eq("vmcall") {
                            vmimpl = true;
                            break;
                        }
                    }
                },
                _ => continue
            }
        }
        if !vmimpl {
            continue;
        }
        func.attrs.remove(index);

        let mut modparam = false;
        if let Some(param) = func.sig.inputs.first_mut() {
            let attrs = match param {
                FnArg::Typed(typed) => &mut typed.attrs,
                _ => unreachable!()
            };
            if let Some(Meta::Path(path)) = attrs.first().map(|attr| &attr.meta) {
                modparam = path.get_ident().map(|ident| ident.eq("module")).unwrap_or(false);
            }
            if modparam {
                attrs.remove(0);
            }
        }

        let ident = &func.sig.ident;
        let args = func.sig.inputs.len() - (modparam as usize);

        modulefuncs.extend(quote! {
            let name = vm.symbols().intern_slice(stringify!(#ident));
            let function = tlang_macros::tfunction!(module, #ident, #args, Some(stringify!(#ident)), #modparam);
            module.set_global(
                name,
                function.into(),
                true).unwrap();
        });
    }

    let mut impls = Vec::new();

    let modname_const: Item = syn::parse2(quote! {
        const MODNAME: &'static str = #modname;
    }).unwrap();
    impls.push(modname_const);

    for name in &types_names {
        let declared = types.get(name).unwrap();
        let display_name = &declared.display_name;
        let ident = &declared.item.ident;

        let typed_impl: Item = syn::parse2(quote! {
            impl crate::tvalue::Typed for #ident {
                const NAME: &'static str = #display_name;

                fn ttype(vm: &VM) -> crate::memory::GCRef<crate::tvalue::TType> {
                    vm.types().query::<Self>()
                }
            }
        }).unwrap();
        impls.push(typed_impl);

        if !declared.impls_into {
            let into_impl: Item = syn::parse2(quote! {
                impl Into<TValue> for crate::memory::GCRef<#ident> {
                    #[inline(always)]
                    fn into(self) -> TValue {
                        TValue::object(self)
                    }
                }
            }).unwrap();
            impls.push(into_impl);
        }

        if !declared.impls_downcast {
            let downcast_impl: Item = syn::parse2(quote! {
                impl VMDowncast for crate::memory::GCRef<#ident> {
                    #[inline(always)]
                    fn vmdowncast(value: TValue, vm: &VM) -> Option<Self> {
                        value.query_object::<#ident>(vm)
                    }
                }
            }).unwrap();
            impls.push(downcast_impl);
        }
        
        let initfunc_ident = Ident::new(&format!("initfunc_{ident}"), Span::call_site());
        let fields = &declared.fields;
        let size = fields.len();

        let priminit = if let DeclaredKind::Primitive = declared.kind {
            Some(quote!( (#ident, #size) ))
        } else {
            None
        };


        let tyinit = match declared.kind {
            DeclaredKind::Primitive => quote! {
                let vm = module.vm();
                let mut builder = TTypeBuilder::build_empty(
                    &vm, #ident::NAME, MODNAME, #ident::ttype(&vm));
                builder.extend(tlang_macros::tobject! { #fields });
                builder.build();
            },
            DeclaredKind::Record => quote! {
                let ttype = tlang_macros::ttype! {
                    name: #ident::NAME,
                    modname: MODNAME,
                    #fields
                }.build(&module.vm());
                module.set_rust_ttype::<#ident>(ttype);
            }
        };
        let initfunc: Item = syn::parse2(quote! {
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            #[initfunc #priminit]
            fn #initfunc_ident(module: crate::memory::GCRef<crate::vm::TModule>) {
                #tyinit
            }
        }).unwrap();

        impls.push(initfunc);
    }

    let modulefuncs_init: Item = syn::parse2(quote! {
        #[initfunc]
        fn modulefuncs_init(mut module: crate::memory::GCRef<TModule>) {
            let vm = module.vm();
            #modulefuncs
        }
    }).unwrap();
    impls.push(modulefuncs_init);

    for imp in impls {
        content.1.push(imp);
    }

    Ok(quote! {
        #[tlang_macros::tmodule_init]
        #module
    })
}

struct InitPrimitive(Path, LitInt);

struct GatheredInit {
    funcident: Ident,
    primitive: Option<InitPrimitive>
}

impl Parse for InitPrimitive {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let path = input.parse()?;
        input.parse::<Token![,]>()?;
        let int = input.parse()?;
        Ok(Self(
            path, int
        ))
    }
}

pub fn generate_tmodule_init(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let mut module: ItemMod = syn::parse2(token_stream)?;

    let Some(content) = module.content.as_mut() else {
        return Err(syn::Error::new(Span::call_site(), "expected module with body"));
    };

    let mut gathered_inits = Vec::new();

    for item in &mut content.1 {
        let itemfn = match item {
            Item::Fn(itemfn) => itemfn,
            _ => continue
        };

        let mut index = 0;
        let mut found = false;
        let mut primitive = None;

        for (idx, attr) in itemfn.attrs.iter().enumerate() {
            index = idx;
            match &attr.meta {
                Meta::Path(path) => {
                    if let Some(ident) = path.get_ident() {
                        if ident.eq("initfunc") {
                            found = true;
                            break;
                        }
                    }
                },
                Meta::List(list) => {
                    if let Some(ident) = list.path.get_ident() {
                        if ident.eq("initfunc") {
                            found = true;
                            primitive = Some(syn::parse2(list.tokens.clone())?);
                            break;
                        }
                    }
                }
                _ => continue,
            }
        }

        if !found {
            continue;
        }
        itemfn.attrs.remove(index);

        gathered_inits.push(GatheredInit {
            funcident: itemfn.sig.ident.clone(),
            primitive
        })
    }

    let mut fnpre = TokenStream::new();
    let mut fnbody = TokenStream::new();

    for init in gathered_inits {
        let funcident = &init.funcident;
        fnbody.extend(quote!{
            #funcident(module);
        });

        if let Some(primitive) = init.primitive {
            let path = primitive.0;
            let count = primitive.1;
            fnpre.extend(quote! {
                println!(stringify!(#path));
                module.set_rust_ttype::<#path>(crate::tvalue::TType::empty(&vm, #count)).unwrap();
            });
        }
    }

    let module_init_fn: Item = syn::parse2(quote! {
        pub fn module_init(mut module: GCRef<TModule>) {
            let vm = module.vm();
            #fnpre
            #fnbody
            module.set_name(MODNAME);
        }
    }).unwrap();

    content.1.push(module_init_fn);

    Ok(quote! {
        #module
    })
}
