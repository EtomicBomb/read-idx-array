//! Reads `IDX` files as described in <http://yann.lecun.com/exdb/mnist/>

use image::GrayImage;
use nom::bytes::complete::tag;
use nom::combinator::eof;
use nom::combinator::map_res;
use nom::multi::count;
use nom::number::complete::be_f32;
use nom::number::complete::be_f64;
use nom::number::complete::be_i16;
use nom::number::complete::be_i32;
use nom::number::complete::be_i8;
use nom::number::complete::be_u32;
use nom::number::complete::be_u8;
use nom::sequence::tuple;
use std::fmt;

/// Error from parsing the `IDX` file.
#[derive(Debug, Clone)]
pub struct Error;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "parse error")
    }
}

type IResult<'a, T> = Result<(&'a [u8], T), nom::Err<nom::error::Error<&'a [u8]>>>;

mod private {
    pub trait Sealed {}
}

#[doc(hidden)]
pub trait DataFormat: private::Sealed {
    const MAGIC_BYTE: u8;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self>;
}

impl private::Sealed for u8 {}
impl DataFormat for u8 {
    const MAGIC_BYTE: u8 = 0x08;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_u8(x)
    }
}

impl private::Sealed for i8 {}
impl DataFormat for i8 {
    const MAGIC_BYTE: u8 = 0x09;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_i8(x)
    }
}

impl private::Sealed for i16 {}
impl DataFormat for i16 {
    const MAGIC_BYTE: u8 = 0x0B;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_i16(x)
    }
}

impl private::Sealed for i32 {}
impl DataFormat for i32 {
    const MAGIC_BYTE: u8 = 0x0C;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_i32(x)
    }
}

impl private::Sealed for f32 {}
impl DataFormat for f32 {
    const MAGIC_BYTE: u8 = 0x0D;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_f32(x)
    }
}

impl private::Sealed for f64 {}
impl DataFormat for f64 {
    const MAGIC_BYTE: u8 = 0x0E;
    fn combinator() -> impl for<'a> Fn(&'a [u8]) -> IResult<'a, Self> {
        |x| be_f64(x)
    }
}

fn check_magic_byte<T: DataFormat>(b: u8) -> Result<(), Error> {
    if b == T::MAGIC_BYTE {
        Ok(())
    } else {
        Err(Error)
    }
}

fn check_num_dims<const N: usize>(num_dims: u8) -> Result<usize, Error> {
    let num_dims = usize::from(num_dims);
    if num_dims == N {
        Ok(num_dims)
    } else {
        Err(Error)
    }
}

fn check_dims_dimensions<const N: usize>(dims: Vec<u32>) -> Result<([u32; N], usize), Error> {
    let dims: [u32; N] = dims.try_into().map_err(|_| Error)?;
    let elements = dims
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(usize::try_from(b).ok()?))
        .ok_or(Error)?;
    Ok((dims, elements))
}

fn parse<T: DataFormat, const N: usize>(x: &[u8]) -> IResult<'_, ([u32; N], Vec<T>)> {
    let (x, (_, (), num_dims)) = tuple((
        tag([0u8; 2]),
        map_res(be_u8, check_magic_byte::<T>),
        map_res(be_u8, check_num_dims::<N>),
    ))(x)?;
    let (x, (dims, elements)) = map_res(count(be_u32, num_dims), check_dims_dimensions)(x)?;
    let (x, data) = count(T::combinator(), elements)(x)?;
    let (x, _) = eof(x)?;
    Ok((x, (dims, data)))
}

/// The array as read from an `IDX` file.
#[derive(Debug, Clone)]
pub struct IdxArray<T, const N: usize> {
    dims: [u32; N],
    data: Vec<T>,
}

impl<T, const N: usize> IdxArray<T, N> {
    /// Returns the raw contents of the `IdxArray`.
    pub fn dims_data(self) -> ([u32; N], Vec<T>) {
        (self.dims, self.data)
    }
}

impl<T: DataFormat, const N: usize> IdxArray<T, N> {
    /// Parses `input`, which is the raw contents of an idx file.
    ///
    /// Assumes you know the type of the image before it's parsed (checks, but does not infer).
    pub fn parse(input: &[u8]) -> Result<Self, Error> {
        match parse(input) {
            Ok((_, (dims, data))) => Ok(IdxArray { dims, data }),
            Err(_) => Err(Error),
        }
    }
}

impl<T> IdxArray<T, 1> {
    /// Returns the contents of this array as a linear sequence.
    pub fn into_sequence(self) -> Vec<T> {
        self.data
    }
}

impl IdxArray<u8, 3> {
    /// Returns the sequence of greyscale images, assuming that `self` is a sequence
    /// of images over its first axis.
    pub fn as_gray_image_sequence(&self) -> Vec<GrayImage> {
        let [_, height, width] = self.dims;
        self.data
            .chunks_exact((width * height) as usize)
            .map(|buf| GrayImage::from_raw(width, height, buf.to_vec()).unwrap())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::IdxArray;
    use std::fs;

    #[test]
    fn test_t10k_labels() {
        let x = fs::read("data/t10k-labels.idx1-ubyte").expect("idx file");
        let x = IdxArray::<u8, 1>::parse(&x).expect("parse index");
        let x = x.into_sequence();
        assert_eq!(x.len(), 10_000);
        assert!(x.iter().all(|e| (0..=9).contains(e)));
        assert!((0u8..=9).all(|e| x.contains(&e)));
    }

    #[test]
    fn test_train_labels() {
        let x = fs::read("data/train-labels.idx1-ubyte").expect("idx file");
        let x = IdxArray::<u8, 1>::parse(&x).expect("parse index");
        let x = x.into_sequence();
        assert_eq!(x.len(), 60_000);
        assert!(x.iter().all(|e| (0..=9).contains(e)));
        assert!((0u8..=9).all(|e| x.contains(&e)));
    }

    #[test]
    fn test_t10k_idx3_ubyte() {
        let x = fs::read("data/t10k-images.idx3-ubyte").expect("idx file");
        let x = IdxArray::<u8, 3>::parse(&x).expect("parse index");
        let x = x.as_gray_image_sequence();
        assert_eq!(x.len(), 10_000);
    }

    #[test]
    fn test_train_idx3_ubyte() {
        let x = fs::read("data/train-images.idx3-ubyte").expect("idx file");
        let x = IdxArray::<u8, 3>::parse(&x).expect("parse index");
        let x = x.as_gray_image_sequence();
        assert_eq!(x.len(), 60_000);
    }
}
