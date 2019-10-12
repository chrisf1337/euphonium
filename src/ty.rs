use crate::ast::{self, TypeDecl, TypeDeclType};
use std::{collections::HashMap, rc::Rc, str::FromStr};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _EnumCase {
    pub id: String,
    pub params: Vec<Rc<_Type>>,
}

impl From<ast::EnumCase> for _EnumCase {
    fn from(case: ast::EnumCase) -> Self {
        Self {
            id: case.id.t,
            params: case.params.into_iter().map(|p| Rc::new(p.t.into())).collect(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumCase {
    pub id: String,
    pub params: Vec<TypeInfo>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _RecordField {
    pub ty: Rc<_Type>,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordField {
    pub ty: TypeInfo,
    pub index: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    ty: Rc<Type>,
    size: usize,
}

impl TypeInfo {
    pub fn new(ty: Rc<Type>, size: usize) -> Self {
        Self { ty, size }
    }

    pub fn ty(&self) -> &Rc<Type> {
        &self.ty
    }

    pub fn size(&self) -> usize {
        match self.ty.as_ref() {
            Type::Alias(_) => unreachable!("cannot get size of alias"),
            Type::Fn(..) => unreachable!("cannot get size of function"),
            _ => self.size,
        }
    }

    pub fn int() -> Self {
        Self {
            ty: Rc::new(Type::Int),
            size: std::mem::size_of::<i64>(),
        }
    }

    pub fn string() -> Self {
        Self {
            ty: Rc::new(Type::String),
            size: std::mem::size_of::<u64>(),
        }
    }

    pub fn bool() -> Self {
        Self {
            ty: Rc::new(Type::Bool),
            size: std::mem::size_of::<bool>(),
        }
    }

    pub fn unit() -> Self {
        Self {
            ty: Rc::new(Type::Unit),
            size: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum _Type {
    Int,
    String,
    Bool,
    Record(String, HashMap<String, _RecordField>),
    Array(Rc<_Type>, usize),
    Unit,
    Alias(String),
    Enum(String, HashMap<String, _EnumCase>),
    Fn(Vec<Rc<_Type>>, Rc<_Type>),
    Iterator(Rc<_Type>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Int,
    String,
    Bool,
    Record(String, HashMap<String, RecordField>),
    Array(TypeInfo, usize),
    Unit,
    Alias(String),
    Enum(String, HashMap<String, EnumCase>),
    Fn(Vec<TypeInfo>, TypeInfo),
    Iterator(TypeInfo),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct _TypeField {
    pub id: String,
    pub ty: Rc<_Type>,
}

impl From<ast::TypeField> for _TypeField {
    fn from(type_field: ast::TypeField) -> Self {
        Self {
            id: type_field.id.t,
            ty: Rc::new(type_field.ty.t.into()),
        }
    }
}

impl _Type {
    pub fn alias(&self) -> Option<&str> {
        if let _Type::Alias(alias) = self {
            Some(alias)
        } else {
            None
        }
    }

    pub fn from_type_decl(type_decl: TypeDecl) -> Self {
        match type_decl.ty.t {
            TypeDeclType::Type(ty) => _Type::from_str(&ty.t).unwrap(),
            TypeDeclType::Enum(cases) => _Type::Enum(
                type_decl.id.t.clone(),
                cases.into_iter().map(|c| (c.t.id.t.clone(), c.t.into())).collect(),
            ),
            TypeDeclType::Record(type_fields) => {
                let mut type_fields_hm = HashMap::new();
                for (i, _TypeField { id, ty }) in type_fields.into_iter().map(|tf| tf.t.into()).enumerate() {
                    type_fields_hm.insert(id, _RecordField { ty, index: i });
                }
                _Type::Record(type_decl.id.t.clone(), type_fields_hm)
            }
            TypeDeclType::Array(ty, len) => _Type::Array(Rc::new(ty.t.into()), len.t),
            TypeDeclType::Fn(param_types, return_type) => _Type::Fn(
                param_types
                    .into_iter()
                    .map(|param_type| Rc::new(param_type.t.into()))
                    .collect(),
                Rc::new(return_type.t.into()),
            ),
            TypeDeclType::Unit => _Type::Unit,
        }
    }
}

impl From<ast::Type> for _Type {
    fn from(ty: ast::Type) -> Self {
        match ty {
            ast::Type::Type(ty) => _Type::from_str(&ty.t).unwrap(),
            ast::Type::Array(ty, len) => _Type::Array(Rc::new(ty.t.into()), len.t),
            ast::Type::Fn(param_types, return_type) => _Type::Fn(
                param_types
                    .into_iter()
                    .map(|param_type| Rc::new(param_type.t.into()))
                    .collect(),
                Rc::new(return_type.t.into()),
            ),
            ast::Type::Unit => _Type::Unit,
        }
    }
}

impl FromStr for _Type {
    type Err = ();

    fn from_str(s: &str) -> std::result::Result<Self, ()> {
        Ok(match s {
            "int" => _Type::Int,
            "string" => _Type::String,
            "bool" => _Type::Bool,
            _ => _Type::Alias(s.to_owned()),
        })
    }
}
