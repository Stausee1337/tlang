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
