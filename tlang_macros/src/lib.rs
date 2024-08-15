use proc_macro::TokenStream;

mod codegen;

#[proc_macro_derive(GeneratorNode)]
pub fn derive_code_gen(item: TokenStream) -> TokenStream {
    codegen::generate_node(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

#[proc_macro]
pub fn define_instructions(item: TokenStream) -> TokenStream {
    codegen::generate_instructions(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

#[proc_macro]
pub fn decode(item: TokenStream) -> TokenStream {
    codegen::generate_decode(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

#[proc_macro]
pub fn ttype(item: TokenStream) -> TokenStream {
    codegen::generate_ttype(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

#[proc_macro]
pub fn tobject(item: TokenStream) -> TokenStream {
    codegen::generate_tobject(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

#[proc_macro]
pub fn tfunction(item: TokenStream) -> TokenStream {
    codegen::generate_tfunction(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

#[proc_macro]
pub fn SelfWithBase(item: TokenStream) -> TokenStream {
    codegen::generate_struct_init(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

#[proc_macro]
pub fn Symbol(item: TokenStream) -> TokenStream {
    codegen::generate_symbol(item.into())
        .unwrap_or_else(|err| syn::Error::to_compile_error(&err))
        .into()
}

