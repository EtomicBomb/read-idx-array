use std::fs;

use nom::error::Error;
use nom::combinator::map_res;
use nom::combinator::cut;
use nom::combinator::eof;
use nom::bytes::complete::tag;
use nom::sequence::tuple;
use nom::multi::count;

use nom::number::complete::be_u8;
use nom::number::complete::be_i8;
use nom::number::complete::be_i16;
use nom::number::complete::be_i32;
use nom::number::complete::be_f32;
use nom::number::complete::be_f64;
use nom::number::complete::be_u32;

mod private {
    pub trait Sealed {}
}

impl private::Sealed for u8 {}
impl private::Sealed for i8 {}
impl private::Sealed for i16 {}
impl private::Sealed for i32 {}
impl private::Sealed for f32 {}
impl private::Sealed for f64 {}

pub trait DataFormat: private::Sealed {
    const MAGIC_BYTE: u8;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self>;
}

impl DataFormat for u8 {
    const MAGIC_BYTE: u8 = 0x08;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_u8(x)
    }
}

impl DataFormat for i8 {
    const MAGIC_BYTE: u8 = 0x09;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_i8(x)
    }
}

impl DataFormat for i16 {
    const MAGIC_BYTE: u8 = 0x0B;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_i16(x)
    }
}

impl DataFormat for i32 {
    const MAGIC_BYTE: u8 = 0x0C;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_i32(x)
    }
}

impl DataFormat for f32 {
    const MAGIC_BYTE: u8 = 0x0D;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_f32(x)
    }
}

impl DataFormat for f64 {
    const MAGIC_BYTE: u8 = 0x0E;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_f64(x)
    }
}

fn check_magic_byte<T: DataFormat>(b: u8) -> Result<(), ()> {
    if b == T::MAGIC_BYTE {
        Ok(())
    } else {
        Err(())
    }
}

// http://yann.lecun.com/exdb/mnist/

type IResult<'a, T, E = nom::Err<Error<&'a [u8]>>> = Result<(&'a [u8], T), E>;

/// Returns the the number of dimensions
fn parse_magic_number<'a, T: DataFormat>(x: &'a [u8]) -> IResult<'a, usize> {
    let (x, (_, (), num_dims)) = tuple((
        tag([0u8; 2]),
        cut(map_res(be_u8, check_magic_byte::<T>)),
        be_u8,
    ))(x)?;
    Ok((x, usize::from(num_dims)))
}

/// Gets the array of dimensions
fn parse_dims<'a>(x: &'a [u8], num_dims: usize) -> IResult<'a, Vec<usize>> {
    map_res(
        count(be_u32, num_dims),
        |dims| dims.into_iter().map(usize::try_from).collect::<Result<Vec<usize>, _>>(),
    )(x)
}

/// Gets the dimensions
fn parse_header<'a, T: DataFormat>(x: &'a [u8]) -> IResult<'a, Vec<usize>> {
    let (x, num_dims) = parse_magic_number::<T>(x)?;
    let (x, dims) = parse_dims(x, num_dims)?;
    Ok((x, dims))
}

fn parse_body<'a, T: DataFormat>(x: &'a [u8], dims: &[usize]) -> IResult<'a, Vec<T>> {
    let elements = dims.iter().try_fold(1usize, |a, &b| a.checked_mul(b)).expect("overflow");
    count(T::combinator(), elements)(x)
}

pub fn parse<T: DataFormat>(x: &[u8]) -> IResult<IdxArray<T>> {
    let (x, dims) = parse_header::<T>(x)?;
    let (x, data) = parse_body(x, &dims)?;
    let (x, _) = eof(x)?;
    Ok((x, IdxArray { dims, data }))
}

#[derive(Debug, Clone)]
pub struct IdxArray<T> {
    dims: Vec<usize>,
    data: Vec<T>,
}

impl<T: DataFormat> IdxArray<T> {
    pub fn new(input: &[u8]) -> Result<IdxArray<T>, ()> {
        match parse(input) {
            Ok((_, array)) => Ok(array),
            Err(_) => Err(()),
        }
    }

    pub fn dims_data(self) -> (Vec<usize>, Vec<T>) {
        (self.dims, self.data)
    }
}

fn main() {
    let input = fs::read("data/train-images.idx3-ubyte").expect("idx file");
    let input = IdxArray::<u8>::new(&input).expect("parse index");
    println!("{:?}", input);
}
