//! TODO:
//! Make this actually use the `binrw` magics -- otherwise
//! why did I bother using all this `binrw` stuff to begin
//! with??

#![allow(unused_imports)]

use binrw::io::{Read, Seek};
use bytemuck::try_cast_slice;
use ndarray::prelude::*;

use std::io::{
    Error as IOError,
    ErrorKind as IOErrorKind,
};

use crate::CorrosiffError;

use crate::tiff::IFD;
use crate::tiff::{
    Tag,
    TiffTagID::{StripOffsets, StripByteCounts, Siff, },
};
use crate::data::image::{
    utils::load_array_from_siff,
    intensity::siff::{
        tiff::{
            load_array_tiff,
            load_array_tiff_registered,
        },
        registered::{
            load_array_raw_siff_registered,
            load_array_compressed_siff_registered,
            extract_mask_raw_siff_registered,
            sum_mask_raw_siff_registered,
            sum_masks_raw_siff_registered,
            extract_mask_compressed_siff_registered,
            sum_mask_compressed_siff_registered,
            sum_masks_compressed_siff_registered,
        },
        unregistered::{
            load_array_raw_siff,
            load_array_compressed_siff,
            extract_mask_raw_siff,
            extract_mask_compressed_siff,
            sum_mask_raw_siff,
            sum_mask_compressed_siff,
            sum_masks_raw_siff,
            sum_masks_compressed_siff,
        },
    }
};

mod registered;
mod unregistered;
mod tiff;
mod siff_frame;

/// For easy package-level access to the intensity loading functions
pub (crate) mod exports {
    pub (crate) use super::siff_frame::SiffFrame;
    pub (crate) use super::load_array as load_array_intensity;
    pub (crate) use super::load_array_registered as load_array_intensity_registered;
    pub (crate) use super::sum_mask as sum_intensity_mask;
    pub (crate) use super::sum_mask_registered as sum_intensity_mask_registered;
    pub (crate) use super::sum_masks as sum_intensity_masks;
    pub (crate) use super::sum_masks_registered as sum_intensity_masks_registered;
    pub (crate) use super::extract_mask as extract_intensity_mask;
    pub (crate) use super::extract_mask_registered as extract_intensity_mask_registered;
}


/// Parses a raw `.siff` format frame and returns
/// an `Intensity` struct containing the intensity data.
#[binrw::parser(reader, endian)]
fn raw_siff_parser<T : Into<u64>>(
    strip_bytes : T,
    ydim : u32,
    xdim : u32
    ) -> binrw::BinResult<Array2<u16>> {
    let mut frame = Array2::<u16>::zeros(
        (ydim as usize, xdim as usize)
    );
    load_array_raw_siff(reader, endian, (&mut frame.view_mut(), strip_bytes, ydim, xdim))?;
    Ok(frame)
}

/// Parses a compressed `.siff` format frame and returns
/// an `Intensity` struct containing the intensity data.
/// 
/// Expected to be at the data strip, so it will go backwards by the size of the
/// intensity data and read that.
#[binrw::parser(reader, endian)]
fn compressed_siff_parser(
        ydim : u32,
        xdim : u32
    ) -> binrw::BinResult<Array2<u16>> {
    
    let mut frame = Array2::<u16>::zeros(
        (ydim as usize, xdim as usize)
    );

    load_array_compressed_siff(reader, endian, (&mut frame.view_mut(), ydim, xdim))?;
    Ok(frame)
}

/// Loads an allocated array with data read directly
/// from a `.siff` file. Will NOT change the `Seek`
/// location of the reader.
/// 
/// ## Arguments
/// 
/// * `reader` - Any reader of a `.siff` file
/// 
/// * `ifd` - The IFD of the frame to load into
/// 
/// * `array` - The array to load the data into viewed as a 2d array
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// 
/// let mut array = Array2::<u16>::zeros((512, 512));
/// let mut reader = File::open("file.siff").unwrap());
/// 
/// load_array(&mut reader, &ifd, &mut array.view_mut());
/// ```
/// 
/// ## See also
/// 
/// * `load_array_registered` - for loading an array
/// and shifting the data based on registration.
pub fn load_array<'a, ReaderT, I>(
        reader : &'a mut ReaderT,
        ifd : &'a I,
        array : &'a mut ArrayViewMut2<u16>
    ) -> Result<(), CorrosiffError> where I : IFD, ReaderT : Read + Seek{
    
    load_array_from_siff!(
        reader,
        ifd,
        (
            load_array_raw_siff,
            (
                &mut array.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            load_array_compressed_siff,
            (
                &mut array.view_mut(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            load_array_tiff,
            (
                &mut array.view_mut(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}

/// Loads an allocated array with data read directly
/// from a `.siff` file. Will NOT change the `Seek`
/// location of the reader.
/// 
/// ## Arguments
/// 
/// * `reader` - Any reader of a `.siff` file
/// 
/// * `ifd` - The IFD of the frame to load into
/// the array
/// 
/// * `array` - The array to load the data into viewed as a 2d array
/// whose pixels will be filled with the intensity data
/// 
/// * `registration` - A tuple of the pixelwise shifts
/// to register the frame. The first element is the
/// shift in the y direction, and the second element
/// is the shift in the x direction. The shifts are
/// in the direct of the shift itself, i.e. a positive
/// registration in the y direction will shift the frame down.
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// 
/// let mut array = Array2::<u16>::zeros((512, 512));
/// let mut reader = File::open("file.siff").unwrap();
/// // TODO finish annotating
/// //let ifd = BigTiffIFD::new
/// // shift the frame down by 2 pixels
/// let registration = (2, 0);
/// load_array_registered(&mut reader, &ifd, &mut array.view_mut(), registration);
/// ```
/// 
/// ## See also
/// 
/// * `load_array` - for loading an array without registration
pub fn load_array_registered<'a, T, S>(
    reader : &'a mut T,
    ifd : &'a S,
    array : &'a mut ArrayViewMut2<u16>,
    registration : (i32, i32),    
) -> Result<(), CorrosiffError> where S : IFD, T : Read + Seek {
    
    load_array_from_siff!(
        reader,
        ifd,
        (
            load_array_raw_siff_registered,
            (
                &mut array.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            load_array_compressed_siff_registered,
            (
                &mut array.view_mut(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            load_array_tiff_registered,
            (
                &mut array.view_mut(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        )
    )
}


/// Loads the pixels of the `target_array` with the data
/// contained in the frame specified by `ifd` only where
/// the `mask` is `true`.
/// 
/// ## Arguments
/// * `reader` - Any reader of a `.siff` file
/// * `ifd` - The IFD of the frame to load into
/// * `target_array` - The array to load the data into viewed as a 1d array,
/// the flattened pixel map.
/// * `mask` - The mask to apply to the frame, where `true` means
/// the pixel should be loaded into the `target_array`.
pub fn extract_mask<I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    target_array : &mut ArrayViewMut1<u64>,
    mask : &ArrayView2<bool>,
    lookup_table : &ArrayView2<usize>,
) -> Result<(), IOError> {
    load_array_from_siff!(
        reader,
        ifd,
        (
            extract_mask_raw_siff,
            (
                target_array,
                mask,
                lookup_table,
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            extract_mask_compressed_siff,
            (
                target_array,
                mask,
                lookup_table,
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}


/// Loads the pixels of the `target_array` with the data
/// contained in the frame specified by `ifd` only where
/// the `mask` is `true`.
/// 
/// ## Arguments
/// * `reader` - Any reader of a `.siff` file
/// * `ifd` - The IFD of the frame to load into
/// * `target_array` - The array to load the data into viewed as a 1d array,
/// the flattened pixel map.
/// * `mask` - The mask to apply to the frame, where `true` means
/// the pixel should be loaded into the `target_array`
/// * `registration` - A tuple of the pixelwise shifts
pub fn extract_mask_registered<I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    target_array : &mut ArrayViewMut1<u64>,
    mask : &ArrayView2<bool>,
    lookup_table : &ArrayView2<usize>,
    registration : (i32, i32),
) -> Result<(), IOError> {
    
    load_array_from_siff!(
        reader,
        ifd,
        (
            extract_mask_raw_siff_registered,
            (
                target_array,
                mask,
                lookup_table,
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            extract_mask_compressed_siff_registered,
            (
                target_array,
                mask,
                lookup_table,
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        )
    )
}

/// Sums all pixels in the requested frame that are
/// masked by the `mask` array and stores the sum
/// in the `frame_sum` argument.
/// 
/// ## Arguments
/// 
/// * `reader` - Any reader of a `.siff` file
/// 
/// * `ifd` - The IFD of the frame to load into
/// 
/// * `frame_sum` - The location to store the summed
/// mask value
/// 
/// * `mask` - The mask to apply to the frame
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// 
/// let mut frame_sum = 0;
/// let mut mask = Array2::<bool>::zeros((512, 512));
/// 
/// let mut reader = File::open("file.siff").unwrap());
/// // *- snip - *  get ifd for the frame of interest //
/// 
/// sum_mask(&mut reader, &ifd, &mut frame_sum, &mask.view());
/// ```
/// 
/// ## See also
/// 
/// - `sum_mask_registered` - for summing the intensity
/// data of a frame with registration
pub fn sum_mask<I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    frame_sum : &mut u64,
    mask : &ArrayView2<bool>,
) -> Result<(), IOError> {

    load_array_from_siff!(
        reader,
        ifd,
        (
            sum_mask_raw_siff,
            (
                frame_sum,
                &mask.view(),
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            sum_mask_compressed_siff,
            (
                frame_sum,
                &mask.view(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}

pub fn sum_mask_registered<I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    frame_sum : &mut u64,
    mask : &ArrayView2<bool>,
    registration : (i32, i32),
) -> Result<(), IOError> {
    
    load_array_from_siff!(
        reader,
        ifd,
        (
            sum_mask_raw_siff_registered,
            (
                frame_sum,
                &mask.view(),
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            sum_mask_compressed_siff_registered,
            (
                frame_sum,
                &mask.view(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        )
    )
}

/// Sums all pixels in the requested frame that are
/// for each mask in the `mask` array's slow dimension
/// and stores each sum in the `frame_sums` argument.
/// 
/// ## Arguments
/// 
/// * `reader` - Any reader of a `.siff` file
/// 
/// * `ifd` - The IFD of the frame to load into
/// 
/// * `frame_sums` - An array of size `mask.dims().0`
/// 
/// * `masks` - The masks to apply to the frame
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// 
/// let mut frame_sum = Array1::<u64>::zeros(6);
/// let mut masks = Array3::<bool>::zeros((6,512, 512));
/// 
/// let mut reader = File::open("file.siff").unwrap());
/// // *- snip - *  get ifd for the frame of interest //
/// 
/// sum_masks(&mut reader, &ifd, &mut frame_sum, &mask.view());
/// ```
/// 
/// ## See also
/// 
/// - `sum_masks_registered` - for summing the intensity
/// data of a frame with registration
pub fn sum_masks<I : IFD, ReaderT: Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    frame_sums : &mut ArrayViewMut1<u64>,
    masks : &ArrayView3<bool>,
) -> Result<(), IOError> {

    load_array_from_siff!(
        reader,
        ifd,
        (
            sum_masks_raw_siff,
            (
                &mut frame_sums.view_mut(),
                &masks.view(),
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            sum_masks_compressed_siff,
            (
                &mut frame_sums.view_mut(),
                &masks.view(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}


/// Sums all pixels in the requested frame that are
/// for each mask in the `mask` array's slow dimension
/// and stores each sum in the `frame_sums` argument.
/// 
/// ## Arguments
/// 
/// * `reader` - Any reader of a `.siff` file
/// 
/// * `ifd` - The IFD of the frame to load into
/// 
/// * `frame_sums` - An array of size `mask.dims().0`
/// 
/// * `masks` - The masks to apply to the frame
/// 
/// * `registration` - A tuple of the pixelwise shifts
/// to register the frame. The first element is the
/// shift in the y direction, and the second element
/// is the shift in the x direction. The shifts are
/// in the direct of the shift itself, i.e. a positive
/// registration in the y direction will shift the frame down.
/// 
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// use std::collections::HashMap;
/// 
/// let mut frame_sum = Array1::<u64>::zeros(6);
/// let mut masks = Array3::<bool>::zeros((6,512, 512));
/// 
/// let mut reader = File::open("file.siff").unwrap());
/// // *- snip - *  get ifd for the frame of interest //
/// 
/// let registration = HashMap::<u64, (i32, i32)>::new();
/// 
/// registration.insert(ifd_idx, (2, 2));
/// 
/// sum_masks_registered(&mut reader, &ifd, &mut frame_sum, &mask.view(), registration);
/// ```
/// 
/// ## See also
/// 
/// - `sum_masks`` - for summing the intensity
/// data of a frame without registration (faster)
pub fn sum_masks_registered<I : IFD, ReaderT: Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    frame_sums : &mut ArrayViewMut1<u64>,
    masks : &ArrayView3<bool>,
    registration : (i32, i32),
) -> Result<(), IOError> {

    load_array_from_siff!(
        reader,
        ifd,
        (
            sum_masks_raw_siff_registered,
            (
                &mut frame_sums.view_mut(),
                &masks.view(),
                ifd.get_tag(StripByteCounts).unwrap().value(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            sum_masks_compressed_siff_registered,
            (
                &mut frame_sums.view_mut(),
                &masks.view(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::exports::*;
    use crate::tests::{get_test_paths, UNCOMPRESSED_FRAME_NUM, COMPRESSED_FRAME_NUM};
    use crate::tiff::BigTiffIFD;

    use crate::tiff::FileFormat;

    #[test]
    fn test_extract_intensity() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let mut f = std::fs::File::open(test_file_path).unwrap();

        let fformat = FileFormat::parse_filetype(&mut f).unwrap();
        let ifd_vec : Vec<BigTiffIFD> = fformat.get_ifd_iter(&mut f).collect();
        
        // Compressed
        assert_eq!(
            SiffFrame::from_ifd(&ifd_vec[COMPRESSED_FRAME_NUM], &mut f).unwrap().intensity.sum(),
            65426 // from SiffPy
        );

        // Uncompressed
        assert_eq!(
            SiffFrame::from_ifd(&ifd_vec[UNCOMPRESSED_FRAME_NUM], &mut f).unwrap().intensity.sum(),
            397 // from SiffPy
        );

        // Says the number of photons in this pointer is right for another frame
        assert_eq!(
            SiffFrame::from_ifd(&ifd_vec[UNCOMPRESSED_FRAME_NUM+1], &mut f).unwrap().intensity.sum(),
            ((&ifd_vec[UNCOMPRESSED_FRAME_NUM+1]).get_tag(StripByteCounts).unwrap().value() as u16) / 8
        );
    }

    #[test]
    fn frame_vs_siffreader(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let mut f = std::fs::File::open(test_file_path).unwrap();
        let fformat = FileFormat::parse_filetype(&mut f).unwrap();
        let ifd_vec : Vec<BigTiffIFD> = fformat.get_ifd_iter(&mut f).collect();

        let frame = SiffFrame::from_ifd(&ifd_vec[60], &mut f).unwrap();

        let sr = crate::open_siff(test_file_path).unwrap();
        let siff_frame = sr.get_frames_intensity(&[60], None).unwrap();

        assert_eq!(frame.intensity, siff_frame.slice(s![0,..,..]));
    }

    /// Shifts but only in the forward direction, i.e.
    /// if the shift is positive, it takes `n..end` and if
    /// the shift is negative, it takes `end-n..end`
    macro_rules! safe_slice_front{
        ($shift_y : expr, $shift_x : expr) => {
            match ($shift_y, $shift_x) {
                (0, 0) => s![.., ..],
                (0, _) => s![.., $shift_x..],
                (_, 0) => s![$shift_y.., ..],
                (_, _) => s![$shift_y.., $shift_x..]
            }
        }
    }

    /// Shifts but only in the backward direction, i.e.
    /// if the shift is positive, it takes `0..end-n` and if
    /// the shift is negative, it takes `n..end`
    macro_rules! safe_slice_back{
        ($shift_y : expr, $shift_x : expr) => {
            match ($shift_y, $shift_x) {
                (0, 0) => s![.., ..],
                (0, _) => s![.., ..-$shift_x],
                (_, 0) => s![..-$shift_y, ..],
                (_, _) => s![..-$shift_y, ..-$shift_x]
            }
        }
    }

    /// A macro to apply a load function and compare it to the registered version.
    /// Does not test the wrap-around behavior -- just up to when the shift wraps around.
    /// ```rust, ignore
    /// test_shift! (
    ///     $shift_y : expr,
    ///     $shift_x : expr,
    ///     $unregistered : expr,
    ///     $registered : expr,
    ///     $func : expr
    /// ) => { ... }
    /// ```
    /// 
    /// # Arguments
    /// 
    /// * `$shift_y` - The shift in the y direction
    /// * `$shift_x` - The shift in the x direction
    /// * `$unregistered` - The unregistered frame
    /// * `$registered` - The registered frame
    /// * `$func` - The function to call to load the registered frame
    /// 
    /// To use:
    /// 
    /// ```rust, ignore
    /// test_shift!(6, 0, frame.intensity, registered, call_load!(load_array_registered, registered));
    /// ```
    macro_rules! test_shift {
        (
            $shift_y : expr,
            $shift_x : expr,
            $unregistered : expr,
            $registered : expr,
            $func : expr
        ) => {
            let registration : (i32, i32) = ($shift_y, $shift_x);
            $func(registration).unwrap();
            assert_eq!(
                $unregistered.slice(safe_slice_back!($shift_y, $shift_x)),
                $registered.slice(safe_slice_front!($shift_y, $shift_x))
            );
        }
    }

    #[test]
    fn test_register() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let mut f = std::fs::File::open(test_file_path).unwrap();
        
        let fformat = FileFormat::parse_filetype(&mut f).unwrap();

        let ifd_vec : Vec<BigTiffIFD> = fformat.get_ifd_iter(&mut f).collect();

        // Shift down
        let frame = SiffFrame::from_ifd(&ifd_vec[UNCOMPRESSED_FRAME_NUM], &mut f).unwrap();
        let mut registered = Array2::<u16>::zeros((128, 128));

        macro_rules! call_load {
            ($func : expr, $register : expr, $ifd : expr) => {
                |x| $func(&mut f, $ifd, &mut $register.view_mut(), x)
            }
        }

        test_shift!(6, 0, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[14]));
        // Shift up
        registered.fill(0);
        test_shift!(-6, 0, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[14]));
        // Shift right
        registered.fill(0);
        test_shift!(0, 6, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[14]));

        registered.fill(0);
        test_shift!(6, -6, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[14]));

        println!("{:?}", ifd_vec[COMPRESSED_FRAME_NUM]);

        let frame = SiffFrame::from_ifd(&ifd_vec[40], &mut f).unwrap();
        let mut registered = Array2::<u16>::zeros((128, 128));

        test_shift!(0, 0, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[40]));
        
        // let registration: (i32, i32) = (0, 0);
        // registered.fill(0);
        // load_array_registered(&mut f, &ifd_vec[40], &mut registered.view_mut(), registration).unwrap();
        assert_eq!(frame.intensity, registered);

        registered.fill(0);
        test_shift!(0, 0, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[40]));

        registered.fill(0);
        test_shift!(0, -6, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[40]));

        registered.fill(0);
        test_shift!(6, 0, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[40]));

        registered.fill(0);
        test_shift!(6, -4, frame.intensity, registered, call_load!(load_array_registered, registered, &ifd_vec[40]));
    }

    #[test]
    /// Tests the photon conversion macros with fake photons
    fn test_photon_parse() {
        use crate::data::image::dimensions::macros::*;

        let y : u16 = (((1 as u64) << 16)-1) as u16;
        let x : u16 = 1;
        let arrival : u32 = 1;
        let photon : u64 = (y as u64) << 48 | (x as u64) << 32 | arrival as u64;
        assert_eq!(photon_to_y!(photon as u64), y as usize);
        assert_eq!(photon_to_x!(photon as u64), x as usize);
    }

}