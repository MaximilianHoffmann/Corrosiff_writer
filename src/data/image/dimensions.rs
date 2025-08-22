//! Code in this submodule deals strictly with attention to
//! image dimensions and the types of things that can go wrong
//! with `Dimensions`.
//! 

use ndarray::prelude::*;
use crate::{tiff::IFD, CorrosiffError};

pub mod macros {

// Lowest 32 bits are the tau coordinate
pub const SIFF_TAU_MASK : u64 = (1<<32) - 1;
/// Highest 16 bits are the y coordinate
pub const SIFF_YMASK : u64 = ((1<<63) | ((1<<63) - 1)) & !((1<<48)-1);
/// Bits 32-48 bits are the x coordinate
pub const SIFF_XMASK : u64 = ((1<<48)- 1) & !((1<<32)-1);

/// Parses a `u64` from a photon in a raw `.siff` read
/// to the y coordinate of the photon. If a shift is
/// provided, it will add the shift to the y coordinate.
/// 
/// Evaluates to a `usize`.
/// 
/// Warning -- need to bring SIFF_YMASK into
/// scope as well. Usually when using this macro you want
/// to use `dimensions::*;` This is bad macro hygiene and
/// I will fix it!
/// 
/// Can be called as:
/// 
/// ```rust, ignore
/// use corrosiff::data::image::dimensions::macros::*;
/// 
/// // One argument -- just a photon
/// let y = photon_to_y!(photon);
/// 
/// // Two arguments -- a photon and a y shift
/// // The resulting y coordinate is increased by
/// // the shift
/// let y = photon_to_y!(photon, shift);
/// 
/// // Three arguments -- a photon, a y shift,
/// // and a max value that wraps around. The resulting
/// // y coordinate is increased by the shift and wrapped
/// // around the max value
/// 
/// let wrapped_y = photon_to_y!(photon, shift, max);
/// ```
macro_rules! photon_to_y {
    ($photon : expr) => {
        (($photon & SIFF_YMASK) >> 48) as usize
    };
    ($photon : expr, $shift : expr) => {
        (((($photon & SIFF_YMASK) >> 48) as i32) + $shift) as usize
    };
    ($photon : expr, $shift : expr, $max : expr) => {
        ((((($photon & SIFF_YMASK) >> 48) as i32 + $shift) as usize) % ($max as usize))
    };
}

/// Parses a `u64` from a photon in a raw `.siff` read
/// to the x coordinate of the photon. If a shift is
/// provided, it will add the shift to the x coordinate.
/// 
/// Evaluates to a `usize`.
/// 
/// Warning -- need to bring SIFF_XMASK into
/// scope as well. Usually when using this macro you want
/// to use `dimensions::*;`
/// 
/// Can be called as:
/// 
/// ```rust, ignore
/// use corrosiff::data::image::dimensions::*;
/// // One argument -- just a photon
/// let x = photon_to_x!(photon);
/// 
/// // Two arguments -- a photon and an x shift
/// // The resulting x coordinate is increased by
/// // the shift
/// let x = photon_to_x!(photon, shift);
/// 
/// // Three arguments -- a photon, an x shift,
/// // and a max value that wraps around. The resulting
/// // x coordinate is increased by the shift and wrapped
/// // around the max value
/// 
/// let wrapped_x = photon_to_x!(photon, shift, max);
/// 
/// ```
macro_rules! photon_to_x {
    ($photon : expr) => {
        (($photon & SIFF_XMASK) >> 32) as usize
    };

    ($photon : expr, $shift : expr) => {
        (((($photon & SIFF_XMASK) >> 32) as i32) + $shift) as usize
    };

    ($photon : expr, $shift : expr, $max : expr) => {
        ((((($photon & SIFF_XMASK) >> 32) as i32 + $shift) as usize) % ($max as usize))
    };
}

/// Parses a `u64` from a photon in a raw `.siff` read
/// to the arrival time of the photon.
/// 
/// Evaluates to a `f64`.
macro_rules! photon_to_tau_FLOAT {
    ($photon : expr) => {
        ($photon & SIFF_TAU_MASK) as f64
    };
}

macro_rules! photon_to_tau_USIZE {
    ($photon : expr) => {
        ($photon & SIFF_TAU_MASK) as usize
    };
}

pub (crate) use photon_to_x;
pub (crate) use photon_to_y;
pub (crate) use photon_to_tau_FLOAT;
pub (crate) use photon_to_tau_USIZE;

}

/// `roll_in_place` rolls the array in place by the given
/// number of pixels in the y and x directions. The array
/// provided is populated by the rolled data, rather than returning
/// a new array. Emulates numpy's `roll` method, which seems like
/// should be in ndarray but isn't!
/// 
/// From the argument, you'd think this would be fast because it would
/// not `malloc` a whole new array, but it actually does so internally
/// (and as a result, is not any faster than `roll`).
/// 
/// ## Arguments
/// 
/// * `array` - The array to roll in place
/// 
/// * `roll` - A tuple of the number of pixels to roll in the y_shift,
/// and x_shift directions. Positive values shift the data right and down,
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use corrosiff::data::image::dimensions::roll_in_place;
/// 
/// ```
/// 
/// ## Panics
/// 
/// * If the shifts are out of bounds
/// 
/// ## See also
/// 
/// `roll` - for rolling an array and returning a new array
pub fn roll_inplace<T : Clone>(array : &mut ArrayViewMut2<T>, roll : (i32, i32)) {
    if roll == (0,0) {return ()}
    
    let copied = array.to_owned();
    
    // need this funny syntax to deal with how `s!` macro works
    match roll {
        // x only
        (0, x_shift) => {

            //let saved = array.slice(s![.., -x_shift..]).to_owned();

            array.slice_mut(s![.., x_shift..]).assign(
                //&array.slice(s![.., ..-x_shift])
                &copied.slice(s![.., ..-x_shift])
            );

            array.slice_mut(s![.., ..x_shift]).assign(
                //&saved
                &copied.slice(s![.., -x_shift..])
            );
        },

        // y only
        (y_shift, 0) => {
           
            // let saved = array.slice(s![-y_shift.., ..]).to_owned();

            array.slice_mut(s![y_shift.., ..]).assign(
                //&array.slice(s![..-y_shift, ..])
                &copied.slice(s![..-y_shift, ..])
            );

            array.slice_mut(s![..y_shift, ..]).assign(
                //&saved
                &copied.slice(s![-y_shift.., ..])
            );
        },

        // x and y
        (y_shift, x_shift) => {
            //let saved = array.slice(s![-y_shift.., -x_shift..]).to_owned();

            array.slice_mut(s![y_shift.., x_shift..]).assign(
                //&array.slice(s![..-y_shift, ..-x_shift])
                &copied.slice(s![..-y_shift, ..-x_shift])
            );

            array.slice_mut(s![..y_shift, x_shift..]).assign(
                //&array.slice(s![-y_shift.., ..-x_shift])
                &copied.slice(s![-y_shift.., ..-x_shift])
            );
            
            array.slice_mut(s![..y_shift, ..x_shift]).assign(
                //&saved
                &copied.slice(s![-y_shift.., -x_shift..])
            );

            array.slice_mut(s![y_shift.., ..x_shift]).assign(
                //&array.slice(s![..-y_shift, -x_shift..])
                &copied.slice(s![..-y_shift, -x_shift..])
            );
        }
    }
}

/// `roll` rolls the array by the given
/// number of pixels in the x and y directions.
/// 
/// ## Arguments
/// 
/// * `array` - The array to roll
/// 
/// * `roll` - A tuple of the number of pixels to roll in the y_shift,
/// and x_shift directions. Positive values shift the data right and down,
/// 
/// ## Returns
/// 
/// A new array with the rolled data
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use corrosiff::data::image::dimensions::roll;
/// 
/// let mut array = Array2::<u16>::zeros((512, 512));
/// 
/// array[[0, 12]] = 5;
/// let rolled = roll(&array, (2, 2));
/// 
/// assert_eq!(rolled[[2, 14]], 5);
/// ```
pub fn roll<T : Clone>(array : &ArrayView2<T>, roll : (i32, i32)) -> Array2<T> {
    let mut rolled = array.to_owned();
    roll_inplace(&mut rolled.view_mut(), roll);
    rolled
}

/// `Dimensions` is a simple struct that holds the dimensions
/// of a frame
/// 
/// `xdim` is the width of the frame
/// `ydim` is the height of the frame
#[derive(PartialEq, Debug, Clone)]
pub struct Dimensions {
    pub xdim : u64,
    pub ydim : u64
}

#[derive(Debug, Clone)]
pub enum DimensionsError {
    MismatchedDimensions{required : Dimensions, requested: Dimensions},
    NoConsistentDimensions,
    IncorrectFrames,
    UnknownHistogramSize,
}

impl Dimensions {
    pub fn new(xdim : u64, ydim : u64) -> Dimensions {
        Dimensions {
            xdim,
            ydim,
        }
    }

    pub fn num_el(&self) -> u64 {
        self.xdim * self.ydim
    }

    /// Creates a `Dimensions` struct from
    /// a tuple (y, x)
    pub fn from_tuple((y, x) : (usize, usize)) -> Dimensions {
        Dimensions {
            xdim : x as u64,
            ydim : y as u64,
        }
    }

    pub fn from_ifd<'a, I : IFD>(ifd : &I)-> Dimensions {
        Dimensions {
            xdim : ifd.width().unwrap().into(),
            ydim : ifd.height().unwrap().into(),
        }
    }
    
    /// Returns the dimensions as a tuple (y, x)
    pub fn to_tuple(&self) -> (usize, usize) {
        (self.ydim as usize, self.xdim as usize)
    }
}

impl std::error::Error for DimensionsError {}

impl std::fmt::Display for DimensionsError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            DimensionsError::MismatchedDimensions{required, requested} => {
                write!(f, "Mismatched dimensions. Requested: ({}, {}), Required: ({}, {})",
                    requested.xdim, requested.ydim, required.xdim, required.ydim)
            },
            DimensionsError::NoConsistentDimensions => {
                write!(f, "Requested data did not have consistent dimensions.")
            },
            DimensionsError::IncorrectFrames => {
                write!(f, "Requested frames are out of bounds.")
            },
            DimensionsError::UnknownHistogramSize => {
                write!(f, "Unknown arrival time histogram size.")
            }
        }
    }
}