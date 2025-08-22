//! Methods in this submodule deal with extracting a pixelwise or ROI-wide
//! empirical lifetime from the data stored in a frame of a `.siff` file.
use binrw::io::{Read, Seek};
use registered::*;
use unregistered::*;

use std::io::{Error as IOError, ErrorKind as IOErrorKind};
use ndarray::prelude::*;
use crate::{
    tiff::{
        IFD,
        TiffTagID::{StripOffsets, StripByteCounts, Siff},
        Tag,
    },
    CorrosiffError,
    data::image::utils::load_array_from_siff,
};

mod unregistered;
mod registered;

/// For crate-level use to access the functions in this module
pub (crate) mod exports {
    pub (crate) use super::load_flim_empirical_and_intensity_arrays;
    pub (crate) use super::load_flim_empirical_and_intensity_arrays_registered;
    pub (crate) use super::sum_lifetime_intensity_mask;
    pub (crate) use super::sum_lifetime_intensity_mask_registered;
    pub (crate) use super::sum_lifetime_intensity_masks;
    pub (crate) use super::sum_lifetime_intensity_masks_registered;
}

/// Loads an array with the pixelwise empirical lifetime
/// from the frame pointed to by the IFD. The reader
/// is returned to its original position. This method
/// is private because you almost never will want the
/// empirical lifetime without getting intensity information.
/// 
/// ## Arguments
/// 
/// * `reader` - The reader with access to the siff file
/// (implements `Read` + `Seek`)
/// 
/// * `ifd` - The IFD pointing to the frame to load the lifetime from
/// 
/// * `array` - The array to load the lifetime into (2d view for one frame)
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// 
/// let mut f = File::open("file.siff").unwrap();
/// let file_format = FileFormat::parse_filetype(&mut f).unwrap();
/// let mut array = Array2::<f64>::zeros((50, 256,256));
/// 
/// let ifds = file_format.get_ifd_vec(&mut f);
/// 
/// for (i, ifd) in ifds.iter().enumerate() {
///     load_flim_array_empirical(
///         &mut f,
///         ifd,
///         &mut array.slice_mut(s![i, ..])
///     ).unwrap();
/// }
/// ```
fn _load_flim_array_empirical<ReaderT, I>(
    reader : &mut ReaderT,
    ifd : &I,
    array : &mut ArrayViewMut2<f64>
    ) -> Result<(), CorrosiffError> where I : IFD, ReaderT : Read + Seek {

    load_array_from_siff!(
        reader,
        ifd,
        (
            _load_flim_array_empirical_uncompressed,
            (
                &mut array.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            _load_flim_array_empirical_compressed,
            (
                &mut array.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}

/// Loads intensity and empirical lifetime arrays from the frame
/// pointed to by the IFD. The reader is returned to its original position.
/// 
/// ## Arguments
/// 
/// * `reader` - The reader with access to the siff file
/// (implements `Read` + `Seek`)
/// 
/// * `ifd` - The IFD pointing to the frame to load the lifetime and intensity
/// data from
/// 
/// * `lifetime` - The array to load the lifetime into (2d view for one frame)
/// 
/// * `intensity` - The array to load the intensity into (2d view for one frame)
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// TODO: Write me!
/// ```
/// 
pub fn load_flim_empirical_and_intensity_arrays<I: IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    lifetime : &mut ArrayViewMut2<f64>,
    intensity : &mut ArrayViewMut2<u16>,
    ) -> Result<(), CorrosiffError> {

    load_array_from_siff!(
        reader,
        ifd,
        (
            _load_flim_intensity_empirical_uncompressed,
            (
                &mut lifetime.view_mut(),
                &mut intensity.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            _load_flim_intensity_empirical_compressed,
            (
                &mut lifetime.view_mut(),
                &mut intensity.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}

pub fn load_flim_empirical_and_intensity_arrays_registered
<I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    lifetime : &mut ArrayViewMut2<f64>,
    intensity : &mut ArrayViewMut2<u16>,
    registration : (i32, i32),
    ) -> Result<(), CorrosiffError> {
    
    load_array_from_siff!(
        reader,
        ifd,
        (
            _load_flim_intensity_empirical_uncompressed_registered,
            (
                &mut lifetime.view_mut(),
                &mut intensity.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            _load_flim_intensity_empirical_compressed_registered,
            (
                &mut lifetime.view_mut(),
                &mut intensity.view_mut(),
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        )
    )
}

/// Applies a mask to the frame of interest and computes the empirical
/// lifetime across all pixels in the mask and the total intensity
/// within the mask, loading the arguments provided in place.
/// 
/// ## Arguments
/// 
/// * `reader` - The reader with access to the siff file (implements
/// `Read` + `Seek`)
/// 
/// * `ifd` - The IFD pointing to the frame to load the lifetime and intensity
/// data from
/// 
/// * `lifetime` - The value to load the computed lifetime into
/// 
/// * `intensity` - The value to load the computed intensity into
/// 
/// * `roi` - The mask to apply to the frame
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// TODO:
/// ```
/// 
/// 
pub fn sum_lifetime_intensity_mask< I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    lifetime : &mut f64,
    intensity : &mut u64,
    roi : &ArrayView2<bool>,
) -> Result<(), CorrosiffError>{
    load_array_from_siff!(
        reader,
        ifd,
        (
            _sum_mask_empirical_intensity_raw,
            (   
                &roi,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            _sum_mask_empirical_intensity_compressed,
            (
                &roi,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}

pub fn sum_lifetime_intensity_mask_registered< I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    lifetime : &mut f64,
    intensity : &mut u64,
    roi : &ArrayView2<bool>,
    registration : (i32, i32),
) -> Result<(), CorrosiffError>{
    load_array_from_siff!(
        reader,
        ifd,
        (
            _sum_mask_empirical_intensity_raw_registered,
            (   
                &roi,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            _sum_mask_empirical_intensity_compressed_registered,
            (
                &roi,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        )
    )
}

pub fn sum_lifetime_intensity_masks< I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    lifetime : &mut ArrayViewMut1<f64>,
    intensity : &mut ArrayViewMut1<u64>,
    rois : &ArrayView3<bool>,
) -> Result<(), CorrosiffError>{
    load_array_from_siff!(
        reader,
        ifd,
        (
            _sum_masks_empirical_intensity_raw,
            (   
                &rois,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        ),
        (
            _sum_masks_empirical_intensity_compressed,
            (
                &rois,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32
            )
        )
    )
}


pub fn sum_lifetime_intensity_masks_registered< I : IFD, ReaderT : Read + Seek>(
    reader : &mut ReaderT,
    ifd : &I,
    lifetime : &mut ArrayViewMut1<f64>,
    intensity : &mut ArrayViewMut1<u64>,
    rois : &ArrayView3<bool>,
    registration : (i32, i32),
) -> Result<(), CorrosiffError>{
    load_array_from_siff!(
        reader,
        ifd,
        (
            _sum_masks_empirical_intensity_raw_registered,
            (   
                &rois,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        ),
        (
            _sum_masks_empirical_intensity_compressed_registered,
            (
                &rois,
                lifetime,
                intensity,
                ifd.get_tag(StripByteCounts).unwrap().value().into(),
                ifd.height().unwrap().into() as u32,
                ifd.width().unwrap().into() as u32,
                registration
            )
        )
    )
}

/// Internal structure for testing and validating
/// reading flim data.
#[allow(dead_code)]
struct FlimArrayEmpirical<D> {
    intensity : ndarray::Array<u16,D>,
    empirical_lifetime : ndarray::Array<f64, D>,
    confidence : Option<ndarray::Array<f64, D>>,
} 

#[allow(dead_code)]
impl<D> FlimArrayEmpirical<D> {
    
    /// Single frame
    pub fn from_ifd<I : IFD>(_ifd : &I) -> Result<FlimArrayEmpirical<Dim<[usize ; 2]>>, CorrosiffError> {
        Err(CorrosiffError::NotImplementedError)
    }

    /// Volume, with requested shape produced with a `reshape` method
    pub fn from_ifds<I : IFD>(_ifds : &[&I], _shape : Option<D>) -> Result<FlimArrayEmpirical<D>, CorrosiffError> {
        Err(CorrosiffError::NotImplementedError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{get_test_paths, COMPRESSED_FRAME_NUM, UNCOMPRESSED_FRAME_NUM};
    use crate::data::image::intensity::siff::load_array as load_array_intensity;

    #[test]
    fn load_compressed_arrival_only() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let TEST_FILE_PATH = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let mut f = std::fs::File::open(TEST_FILE_PATH).unwrap();
        let file_format = crate::tiff::FileFormat::parse_filetype(&mut f).unwrap();

        let ifds = file_format.get_ifd_vec(&mut f);
        // let shape = (ifds[COMPRESSED_FRAME_NUM].height().unwrap().into() as usize,
        //     ifds[COMPRESSED_FRAME_NUM].width().unwrap().into() as usize);
        let shape = (128,128);
        let mut array = Array2::<f64>::zeros(shape);

        _load_flim_array_empirical(&mut f, &ifds[COMPRESSED_FRAME_NUM], &mut array.view_mut()).unwrap();
    }

    #[test]
    fn load_uncompressed_arrival_only() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let TEST_FILE_PATH = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let mut f = std::fs::File::open(TEST_FILE_PATH).unwrap();
        let file_format = crate::tiff::FileFormat::parse_filetype(&mut f).unwrap();

        let ifds = file_format.get_ifd_vec(&mut f);
        // let shape = (ifds[UNCOMPRESSED_FRAME_NUM].height().unwrap().into() as usize,
        //     ifds[UNCOMPRESSED_FRAME_NUM].width().unwrap().into() as usize);
        let shape = (128,128);
        let mut array = Array2::<f64>::zeros(shape);

        _load_flim_array_empirical(&mut f, &ifds[UNCOMPRESSED_FRAME_NUM], &mut array.view_mut()).unwrap();
    }

    #[test]
    fn load_intensity_and_flim_together_test(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let TEST_FILE_PATH = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let mut f = std::fs::File::open(TEST_FILE_PATH).unwrap();
        let file_format = crate::tiff::FileFormat::parse_filetype(&mut f).unwrap();

        let ifds = file_format.get_ifd_vec(&mut f);
        let shape = (128,128);
        let mut lifetime = Array2::<f64>::zeros(shape);
        let mut intensity = Array2::<u16>::zeros(shape);

        load_flim_empirical_and_intensity_arrays(&mut f, &ifds[UNCOMPRESSED_FRAME_NUM], &mut lifetime.view_mut(), &mut intensity.view_mut()).unwrap();

        // Now check that they're the same as the arrival-only tests

        let mut lifetime_arrival = Array2::<f64>::zeros(shape);
        let mut intensity_alone = Array2::<u16>::zeros(shape);

        _load_flim_array_empirical(&mut f, &ifds[UNCOMPRESSED_FRAME_NUM], &mut lifetime_arrival.view_mut()).unwrap();
        load_array_intensity(&mut f, &ifds[UNCOMPRESSED_FRAME_NUM], &mut intensity_alone.view_mut()).unwrap();
        
        lifetime.iter().zip(lifetime_arrival.iter()).for_each(|(&x,&y)| {
            if (!x.is_nan()) | (!y.is_nan()) {assert_eq!(x,y);}
        });

        intensity.iter().zip(intensity_alone.iter()).for_each(|(&x,&y)| {
            assert_eq!(x,y);
        });

        // Now again for the compressed frame

        let mut lifetime = Array2::<f64>::zeros(shape);
        let mut intensity = Array2::<u16>::zeros(shape);

        load_flim_empirical_and_intensity_arrays(&mut f, &ifds[COMPRESSED_FRAME_NUM], &mut lifetime.view_mut(), &mut intensity.view_mut()).unwrap();

        // Now check that they're the same as the arrival-only tests

        let mut lifetime_arrival = Array2::<f64>::zeros(shape);
        let mut intensity_alone = Array2::<u16>::zeros(shape);

        _load_flim_array_empirical(&mut f, &ifds[COMPRESSED_FRAME_NUM], &mut lifetime_arrival.view_mut()).unwrap();
        load_array_intensity(&mut f, &ifds[COMPRESSED_FRAME_NUM], &mut intensity_alone.view_mut()).unwrap();

        lifetime.iter().zip(lifetime_arrival.iter()).for_each(|(&x,&y)| {
            if (!x.is_nan()) | (!y.is_nan()) {assert_eq!(x,y);}
        });

        intensity.iter().zip(intensity_alone.iter()).for_each(|(&x,&y)| {
            assert_eq!(x,y);
        });

    }
}
