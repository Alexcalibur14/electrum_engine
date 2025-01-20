use proc_macro::TokenStream;
use syn::{DeriveInput, Type};

#[proc_macro_derive(Vertex, attributes(vertex))]
pub fn vertex_derive_macro(item: TokenStream) -> TokenStream {
    vertex_derive_macro2(item.into()).unwrap().into()
}

fn vertex_derive_macro2(
    item: proc_macro2::TokenStream,
) -> deluxe::Result<proc_macro2::TokenStream> {
    // Parse
    let ast: DeriveInput = syn::parse2(item)?;

    // generate
    Ok(impl_vertex_trait(ast))
}

fn impl_vertex_trait(mut ast: DeriveInput) -> proc_macro2::TokenStream {
    let VertexStructAttributes { binding } = deluxe::extract_attributes(&mut ast).unwrap();

    let (field_bindings, field_formats) = extract_field_attributes(&mut ast);

    // get struct identifier
    let ident = ast.ident;
    let (impl_generics, type_generics, where_clauses) = ast.generics.split_for_impl();

    // get field types
    let field_types: Vec<Type> = match ast.data {
        syn::Data::Struct(data) => data.fields.into_iter().map(|f| f.ty).collect(),
        syn::Data::Enum(_) => panic!("Only structs are supported"),
        syn::Data::Union(_) => panic!("Only structs are supported"),
    };

    let mut offset_string = quote::quote! { 0u32 };
    let offsets: Vec<proc_macro2::TokenStream> = field_types
        .iter()
        .map(|ty| {
            let parsed_string = offset_string.clone();
            let offset = quote::quote! { #parsed_string };

            offset_string = quote::quote! { #offset_string + std::mem::size_of::<#ty>() as u32 };

            offset
        })
        .collect();

    let locations: Vec<u32> = (0..field_types.len()).map(|i| i as u32).collect();

    // Generate impl
    quote::quote! {
        impl #impl_generics crate::Vertex for #ident #type_generics #where_clauses {
            fn binding_descriptions() -> Vec<ash::vk::VertexInputBindingDescription> {
                vec![
                    ash::vk::VertexInputBindingDescription {
                        binding: #binding,
                        stride: std::mem::size_of::<#ident>() as u32,
                        input_rate: ash::vk::VertexInputRate::VERTEX,
                    }
                ]
            }

            fn attribute_descriptions() -> Vec<ash::vk::VertexInputAttributeDescription> {
                vec![
                    #(
                        ash::vk::VertexInputAttributeDescription {
                            location: #locations,
                            binding: #field_bindings,
                            format: ash::vk::Format::from_raw(#field_formats),
                            offset: #offsets,
                        },
                    )*
                ]
            }
        }
    }
}

fn extract_field_attributes(ast: &mut DeriveInput) -> (Vec<u32>, Vec<i32>) {
    let mut field_bindings = Vec::new();
    let mut field_formats = Vec::new();

    if let syn::Data::Struct(s) = &mut ast.data {
        for field in s.fields.iter_mut() {
            let VertexFieldAttributes { binding, format } =
                deluxe::extract_attributes(&mut field.attrs).unwrap();
            field_bindings.push(binding);
            field_formats.push(format);
        }
    }

    (field_bindings, field_formats)
}

#[derive(deluxe::ExtractAttributes)]
#[deluxe(attributes(vertex))]
struct VertexStructAttributes {
    #[deluxe(default = 0)]
    binding: u32,
}

#[derive(deluxe::ExtractAttributes)]
#[deluxe(attributes(vertex))]
struct VertexFieldAttributes {
    #[deluxe(default = 0)]
    binding: u32,
    format: i32,
}
