
use convert_case::{Casing, Case};
use proc_macro2::{TokenStream, Ident, Span};
use quote::{quote, ToTokens, quote_spanned, TokenStreamExt};
use syn::{Fields, spanned::Spanned, ItemEnum, punctuated::{Punctuated, Pair}, Token, FieldsNamed, Expr, parse::{Parse, Parser}, Visibility, token::Brace, Attribute, Meta, MacroDelimiter, LitBool, braced, PatIdent, Lifetime, Type, TypeReference, FieldValue, Member, ItemMod, Item, ItemStruct, LitStr, Path, PathArguments, GenericArgument, LitInt, ExprLit, Lit, FnArg, ExprCall, PathSegment, TypePath};

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
    fields: Option<&FieldsNamed>,
    is_terminator: bool,
    structures: &mut TokenStream) {

    let mut fields_tokens = TokenStream::new();

    if let Some(fields) = fields {
        fields.to_tokens(&mut fields_tokens);
        let mut ident_fields = TokenStream::new();
        for field in &fields.named {
            let fident = &field.ident;
            ident_fields.extend(quote!(#fident,));
        }
    } else {
        fields_tokens.extend(quote!(;));
    }

    structures.extend(quote!(
            #[derive(Clone, Copy, Debug)]
            #[allow(non_camel_case_types)]
            #[repr(packed)]
            #vis struct #ident #fields_tokens));
    structures.extend(quote! {
        impl Instruction for #ident {
            const CODE: OpCode = OpCode::#ident;
            const IS_TERMINATOR: bool = #is_terminator;
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
            pub fn #snake_case_ident (&mut self, #params) {
                debug_assert!(!self.terminated);
                #terminator_code
                const SIZE: usize = std::mem::size_of::<crate::bytecode::instructions::#ident>();

                let instruction = crate::bytecode::instructions::#ident {
                    #fargs
                };
                let instruction = unsafe {
                    std::mem::transmute::<_, [u8; SIZE]>(instruction)
                };
                self.data.write(&[crate::bytecode::OpCode::#ident as u8]).unwrap();
                self.data.write(&instruction).unwrap();
            }
        });

        codegen_impls.extend(quote! {
            pub fn #snake_case_ident (&mut self, #params) {
                let func = self.current_fn_mut();
                let block = func.current_block_mut();
                block.#snake_case_ident(#fargs);
            }
        });

        debug_deserialize.extend(quote! {
            OpCode::#ident => {
                let instruction: crate::bytecode::instructions::#ident = *stream.read();
                stream.bump(std::mem::size_of::<crate::bytecode::instructions::#ident>());
                Box::new(instruction)
            },
        });

        if let Some(fields) = &mut inst.fields {
            for field in &mut fields.named {
                field.vis = Visibility::Public(syn::token::Pub::default());
            }
        }

        make_struct(
            Visibility::Public(syn::token::Pub::default()),
            ident, 
            inst.fields.as_ref(),
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
            pub fn deserialize_for_debug<'l>(self, stream: &mut CodeStream<'l>
                ) -> Box<dyn std::fmt::Debug> {
                match self {
                    #debug_deserialize
                }
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
    stream: Expr,
    _comma_token1: Token![,],
    environment: Expr,
    _comma_token2: Token![,],
    decodable: Decodable,
}

impl Parse for DecodeStmt {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let deserializer = input.parse()?;
        let comma_token1 = input.parse()?;
        let environment = input.parse()?;
        let comma_token2 = input.parse()?;
        let decodable = input.parse()?;
        Ok(Self { stream: deserializer, _comma_token1: comma_token1, environment, _comma_token2: comma_token2, decodable })
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
    _brace_token: Brace,
    named: Punctuated<DecodeField, Token![,]>,
}

impl Parse for DecodeFields {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let content;
        let brace_token = braced!(content in input);

        let named = content.parse_terminated(DecodeField::parse, Token![,])?;
        Ok(Self { _brace_token: brace_token, named })
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
    Decode(Token![&], Option<Token![mut]>),
}

impl Parse for DecodeKind {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(Token![&]) {
            let anpersand = input.parse()?;
            let mutability = if input.peek(Token![mut]) {
                Some(input.parse()?)
            } else {
                None
            };
            Ok(DecodeKind::Decode(anpersand, mutability))
        } else{
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
    let stream = &stmt.stream;

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
            DecodeKind::Copy => quote!(#ident),
            DecodeKind::Decode(_, mutability) => {
                if mutability.is_some() {
                    quote!(#environment.decode_mut(#ident))
                } else {
                    quote!(#environment.decode(#ident))
                }
            },
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

            #[repr(C, packed)]
            struct Reader(crate::bytecode::OpCode, crate::bytecode::instructions::#ident);

            let Reader(_, instruction): Reader = *(#stream).read();
            (#stream).bump(
                std::mem::size_of::<crate::bytecode::instructions::#ident>() + 
                std::mem::size_of::<crate::bytecode::OpCode>());
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

struct TCall {
    vm: Expr,
    comma_token: Token![,],
    polymorphic: ExprCall
}

impl Parse for TCall {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let vm = input.parse()?;
        let comma_token = input.parse()?;
        let polymorphic = input.parse()?;
        Ok(Self {
            vm,
            comma_token,
            polymorphic
        })
    }
}

pub fn generate_tcall(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let tcall: TCall = syn::parse2(token_stream)?;
    let vm = &tcall.vm;

    let sym;
    let ty;
    match tcall.polymorphic.func.as_ref() {
        Expr::Path(path) => {
            if path.path.segments.len() < 2 {
                return Err(
                    syn::Error::new(tcall.polymorphic.span(), "expected function path of at least 2 segments")
                );
            }
            sym = path.path.segments.last();
            let pairs = path.path.segments.pairs();

            let mut ty_path = Punctuated::<PathSegment, Token![::]>::new();
            for pair in pairs {
                match pair {
                    Pair::Punctuated(value, punct) => {
                        ty_path.push_value(value.clone());
                        ty_path.push_punct(punct.clone());
                    }
                    Pair::End(..) => {
                        ty_path.pop_punct();
                    }
                }
            }

            let path = Path {
                leading_colon: None,
                segments: ty_path
            };
            ty = TypePath {
                qself: None,
                path
            };
        },
        _ => {
            return Err(
                syn::Error::new(tcall.polymorphic.span(), "expected polymorphic function path")
            );
        }
    }

    let self_ = tcall.polymorphic.args.first()
        .ok_or_else(|| syn::Error::new(tcall.polymorphic.span(), "expected, at least, first self argument"))?;
    let self_ = self_.clone();
    
    let arguments: Punctuated<Expr, Token![,]> = Punctuated::from_iter(tcall.polymorphic.args.into_pairs().skip(1));

    Ok(quote! {
        {
            let value: #ty = #self_;
            // TODO: intern symbol #sym for messages in error resolval
            let resolved_func: tlang::interop::TPolymorphicCallable<_, _> = tlang::tvalue::resolve_by_symbol(
                #vm, tlang_macros::Symbol![#sym], value, false);
            if resolved_func.is_method() {
                resolved_func(#self_, #arguments)
            } else {
                let resolved_func2: tlang::interop::TPolymorphicCallable<_, _> = resolved_func.reencode();
                resolved_func2(#arguments)
            }
        }
    })
}

pub fn generate_tget(token_stream: TokenStream) -> Result<TokenStream, syn::Error> {
    let tcall: TCall = syn::parse2(token_stream)?;
    let vm = &tcall.vm;

    let sym;
    let ty;
    match tcall.polymorphic.func.as_ref() {
        Expr::Path(path) => {
            if path.path.segments.len() < 2 {
                return Err(
                    syn::Error::new(tcall.polymorphic.span(), "expected function path of at least 2 segments")
                );
            }
            sym = path.path.segments.last();
            let pairs = path.path.segments.pairs();

            let mut ty_path = Punctuated::<PathSegment, Token![::]>::new();
            for pair in pairs {
                match pair {
                    Pair::Punctuated(value, punct) => {
                        ty_path.push_value(value.clone());
                        ty_path.push_punct(punct.clone());
                    }
                    Pair::End(..) => {
                        ty_path.pop_punct();
                    }
                }
            }

            let path = Path {
                leading_colon: None,
                segments: ty_path
            };
            ty = TypePath {
                qself: None,
                path
            };
        },
        _ => {
            return Err(
                syn::Error::new(tcall.polymorphic.span(), "expected polymorphic function path")
            );
        }
    }

    if tcall.polymorphic.args.len() != 1 {
        return Err(syn::Error::new(tcall.polymorphic.span(), "expected, one (self) argument"));
    }

    let self_ = tcall.polymorphic.args.first().unwrap();

    let self_ = self_.clone();

    Ok(quote! {
        {
            let value: #ty = #self_;
            // TODO: intern symbol #sym for messages in error resolval
            let mut resolved_access: tlang::interop::TPropertyAccess<_> = tlang::tvalue::resolve_by_symbol(
                #vm, tlang_macros::Symbol![#sym], value, true);
            resolved_access
        }
    })
}
