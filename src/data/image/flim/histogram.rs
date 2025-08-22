//! TODO: Masked histogram methods!

use binrw::io::{Read, Seek};
use bytemuck::try_cast_slice;
use ndarray::prelude::*;
use itertools::izip;

use std::io::{
    Error as IOError,
    ErrorKind as IOErrorKind,
};

use crate::{
    data::image::utils::photonwise_op, tiff::{
    Tag, TiffTagID::{Siff, StripByteCounts, StripOffsets}, IFD
    }, CorrosiffError
};
use crate::data::image::{
    utils::load_array_from_siff,
    dimensions::{roll,macros::*}
};

/// Crate level exports
pub (crate) mod exports {
    pub (crate) use super::load_histogram;
    pub (crate) use super::load_histogram_mask;
    pub (crate) use super::load_histogram_mask_registered;
}

/// Reads the data pointed to by the IFD and uses it to
/// increment the counts of the histogram. Presumes
/// the reader already points to the start of the main data.
fn _load_histogram_compressed<I, ReaderT>(
    ifd : &I,
    reader : &mut ReaderT,
    histogram : &mut ArrayViewMut1<u64>
    ) -> Result<(), IOError> 
    where I : IFD, ReaderT : Read + Seek {

    let strip_byte_counts = ifd.get_tag(StripByteCounts).unwrap().value();
    
    let mut data: Vec<u8> = vec![0; strip_byte_counts.into() as usize];
    reader.read_exact(&mut data)?;

    let hlen = histogram.len();
    try_cast_slice::<u8, u16>(&data).map_err(
        |err| IOError::new(IOErrorKind::InvalidData, err)
    )?.iter().for_each(|&x| {histogram[x as usize % hlen] += 1});

    Ok(())
}

/// Presumes the reader is already at the start of the data
fn _load_histogram_uncompressed<I, ReaderT>(
    ifd : &I,
    reader : &mut ReaderT,
    histogram : &mut ArrayViewMut1<u64>
    ) -> Result<(), IOError> 
    where ReaderT : Read + Seek, I : IFD{

    let strip_byte_counts = ifd.get_tag(StripByteCounts).unwrap().value();
    let mut data : Vec<u8> = vec![0; strip_byte_counts.into() as usize];
    reader.read_exact(&mut data)?;

    let hlen = histogram.len();

    unsafe {
        let (_, data, _) = data.align_to::<u64>();
        data.iter().for_each(|&x| {
            let tau = photon_to_tau_USIZE!(x);
            histogram[tau % hlen] += 1;
        });
    }

    Ok(())
}

/// Takes an existing array viewed in 1 dimension (presumed to be the tau dimension)
/// and loads the data from the frame pointed to by the current IFD.
/// 
/// Will NOT change the position of the reader.
/// 
/// ## Arguments
/// 
/// * `ifd` - The IFD pointing to the frame to load the histogram from
/// 
/// * `reader` - The reader with access to the data
/// 
/// * `histogram` - The array to load the histogram into (1d)
/// 
/// ## Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::fs::File;
/// 
/// let mut f = File::open("file.siff").unwrap();
/// let file_format = FileFormat::parse_filetype(&mut f).unwrap();
/// let mut array = Array2::<u64>::zeros((50, file_format.num_flim_tau_bins().unwrap()));
/// let ifds = file_format.get_ifd_vec(&mut f);
/// 
/// for (i, ifd) in ifds.iter().enumerate() {
///    load_histogram(ifd, &mut f, &mut array.slice_mut(s![i, ..])).unwrap();
/// }
/// ```
pub fn load_histogram<I, ReaderT>(
    ifd: &I, reader: &mut ReaderT, histogram : &mut ArrayViewMut1<u64>
    )-> Result<(), IOError> where I : IFD, ReaderT : Read + Seek {

    let curr_pos = reader.stream_position()?;
    reader.seek(
        std::io::SeekFrom::Start(
            ifd.get_tag(StripOffsets)
            .ok_or(IOError::new(IOErrorKind::InvalidData,
            "Strip offset not found")
            )?.value().into()
        )  
    )?;
    match ifd.get_tag(Siff).unwrap().value().into() {
        0 => {
            _load_histogram_uncompressed(ifd, reader, histogram)?;
        },
        1 => {
            _load_histogram_compressed(ifd, reader, histogram)?;
        },
        _ => {
            Err(IOError::new(IOErrorKind::InvalidData,
                "Invalid Siff tag value"))?;
        }
    }
    let _ = reader.seek(std::io::SeekFrom::Start(curr_pos));
    Ok(())
}

#[binrw::parser(reader)]
fn _load_histogram_mask_uncompressed<I : IFD>(
    ifd: &I,
    mask : &ArrayView2<bool>,
    histogram : &mut ArrayViewMut1<u64>,
    ) -> Result<(), IOError> {

    let xdim = ifd.width().unwrap().into() as u32;
    let ydim = ifd.height().unwrap().into() as u32;
    let hlen = histogram.len();
    let strip_bytes = ifd.get_tag(StripByteCounts).unwrap().value();
    photonwise_op!(
        reader,
        strip_bytes,
        |x : &u64| {
            histogram[photon_to_tau_USIZE!(x) % hlen] += mask[
                [photon_to_y!(*x, 0, ydim), photon_to_x!(*x, 0, xdim)]
            ] as u64;
        }
    );

    Ok(())
}

#[binrw::parser(reader)]
fn _load_histogram_mask_compressed<I : IFD>(
    ifd: &I,
    mask : &ArrayView2<bool>,
    histogram : &mut ArrayViewMut1<u64>
    ) -> Result<(), IOError> {
    
    let xdim = ifd.width().unwrap().into() as u32;
    let ydim = ifd.height().unwrap().into() as u32;

    reader.seek(std::io::SeekFrom::Current(
        -((ydim * xdim * std::mem::size_of::<u16>() as u32) as i64)
    ))?;

    let mut data : Vec<u8> = vec![0;
        ydim as usize * xdim as usize * std::mem::size_of::<u16>()
    ];

    reader.read_exact(&mut data)?;

    let intensity_data = try_cast_slice::<u8, u16>(&data).map_err(
        |err| IOError::new(IOErrorKind::InvalidData, err)
    )?;

    let strip_byte_counts = ifd.get_tag(StripByteCounts).unwrap().value();
    
    let mut data: Vec<u8> = vec![0; strip_byte_counts.into() as usize];
    reader.read_exact(&mut data)?;

    let hlen = histogram.len();

    let arrival_times = try_cast_slice::<u8, u16>(&data).map_err(
        |err| IOError::new(IOErrorKind::InvalidData, err)
    )?;

    let mut arrival_time_pointer = 0;
    izip!(
        intensity_data.iter(),
        mask.iter(),
    ).for_each(|(&intensity, &m)| {
            arrival_times[arrival_time_pointer..arrival_time_pointer + intensity as usize].iter().for_each(|&tau| {
                histogram[tau as usize % hlen] += m as u64;
            });
            arrival_time_pointer += intensity as usize;
    });

    // let strip_byte_counts = ifd.get_tag(StripByteCounts).unwrap().value();
    // let mut data: Vec<u8> = vec![0; strip_byte_counts.into() as usize];
    // reader.read_exact(&mut data)?;
    
    // // confusing that the `if` statement is needed!!
    // // come back to this! Maybe there's a mistake in how
    // // the data is being saved??? Or maybe sometimes the laser
    // // sync is missed, like one in every several thousand pulses?
    // try_cast_slice::<u8, u16>(&data).map_err(
    //     |err| IOError::new(IOErrorKind::InvalidData, err)
    // )?.iter().zip(mask.iter()).for_each(|(&x, &m)| if m != 0 && x < histogram.len() as u16 {histogram[x as usize] += 1});

    Ok(())
}

/// Reads the data pointed to by the IFD and uses it to
/// increment the counts of the histogram provided by all
/// pixels within the mask.
pub fn load_histogram_mask<I : IFD, ReaderT : Read + Seek>(
    reader: &mut ReaderT,
    ifd: &I,
    histogram : &mut ArrayViewMut1<u64>,
    mask : &ArrayView2<bool>,
    )-> Result<(), IOError> {

    let curr_pos = reader.stream_position()?;
    reader.seek(
        std::io::SeekFrom::Start(
            ifd.get_tag(StripOffsets)
            .ok_or(IOError::new(IOErrorKind::InvalidData,
            "Strip offset not found")
            )?.value().into()
        )  
    )?;
    match ifd.get_tag(Siff).unwrap().value().into() {
        0 => {
            _load_histogram_mask_uncompressed(reader, binrw::Endian::Little, (ifd, mask, histogram))
        },
        1 => {
            _load_histogram_mask_compressed(reader, binrw::Endian::Little, (ifd, mask, histogram))
        },
        _ => {
            Err(IOError::new(IOErrorKind::InvalidData,
                "Invalid Siff tag value"))
        }
    }.map_err(|err| {
        let _ = reader.seek(std::io::SeekFrom::Start(curr_pos));
        err
    })?;
    let _ = reader.seek(std::io::SeekFrom::Start(curr_pos));
    Ok(())
}


/// Reads the data pointed to by the IFD and uses it to
/// increment the counts of the histogram provided by all
/// pixels within the mask, adjusting for the registration of the frame.
pub fn load_histogram_mask_registered<I : IFD, ReaderT : Read + Seek>(
    reader: &mut ReaderT,
    ifd : &I,
    histogram : &mut ArrayViewMut1<u64>,
    mask : &ArrayView2<bool>,
    _registration : (i32, i32)
    )-> Result<(), IOError> {

    let rolled_mask = roll(mask, _registration);
    load_histogram_mask(reader, ifd, histogram, &rolled_mask.view())?;
    Ok(())
}

/// Probably will contain more info at
/// some point...
#[allow(dead_code)]
struct FlimHistogram {
    data : Array1<u64>,
}

#[allow(dead_code)]
impl FlimHistogram {

    /// Create a new FlimHistogram from a given IFD
    /// 
    /// ## Arguments
    /// 
    /// * `ifd` - The IFD for the frame to create the histogram from
    ///
    /// * `reader` - The reader with access to the data
    ///
    /// ## Returns
    /// 
    /// A new FlimHistogram 
    fn from_ifd<'a, 'b, I, ReaderT>(ifd : &'a I, reader : &'b mut ReaderT, n_bins : u32)
    -> Result<Self, IOError> where I : IFD, ReaderT : Read + Seek {
        let curr_pos = reader.stream_position()?;

        reader.seek(
            std::io::SeekFrom::Start(
                ifd.get_tag(StripOffsets)
                .ok_or(IOError::new(IOErrorKind::InvalidData,
                "Strip offset not found")
                )?.value().into()
            )  
        )?;

        let mut hist = FlimHistogram {
            data : Array1::zeros(Dim(n_bins as usize)),
        };

        match ifd.get_tag(Siff).unwrap().value().into() {
            0 => {
                _load_histogram_uncompressed(ifd, reader, &mut hist.data.view_mut())?;
            },
            1 => {
                _load_histogram_compressed(ifd, reader, &mut hist.data.view_mut())?;
            },
            _ => {
                let _ = reader.seek(std::io::SeekFrom::Start(curr_pos));
                Err(IOError::new(IOErrorKind::InvalidData,
                    "Invalid Siff tag value"))?;
            }
        }
        
        let _ = reader.seek(std::io::SeekFrom::Start(curr_pos));
        Ok(hist)
    }
}

/// This is an image with an extra axis corresponding to
/// the arrival time of the photons in each pixel. The
/// fastest axis is the "tau" axis, corresponding to the
/// number of photons arriving in bin `tau` in that pixel
/// in the frame. For most FLIM data, there are ~1000 arrival
/// time bins, so these data RAPIDLY become gigantic.
#[allow(dead_code)]
struct ImageHistogram<D> {
    data : ndarray::Array<u64, D>
}

#[allow(dead_code)]
impl<D> ImageHistogram<D> {

    pub fn new_from_ifds<I : IFD>(_ifds : &[&I]) -> Result<Self, CorrosiffError> {
        Err(CorrosiffError::NotImplementedError)
    }
}


#[cfg(test)]
mod tests{
    use super::*;
    use crate::tests::{
        get_test_paths,
        UNCOMPRESSED_FRAME_NUM,
        COMPRESSED_FRAME_NUM
    };
    use crate::tiff::FileFormat;
    use crate::data::image::intensity::siff::exports::SiffFrame;

    #[test]
    fn single_frame_histograms() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let TEST_FILE_PATH = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let mut f = std::fs::File::open(TEST_FILE_PATH).unwrap();

        let file_format = FileFormat::parse_filetype(&mut f).unwrap();
        let ifd_vec = file_format.get_ifd_vec(&mut f);

        let _hist = FlimHistogram::from_ifd(
            &ifd_vec[COMPRESSED_FRAME_NUM], 
            &mut f, 
            file_format.num_flim_tau_bins().unwrap()
        ).unwrap();

        let _frame = SiffFrame::from_ifd(&ifd_vec[COMPRESSED_FRAME_NUM], &mut f).unwrap();

        // Should have the same number of photons
        // assert_eq!(hist.data.sum(), frame.intensity.fold(0, |running_sum, &x| running_sum + (x as u64)));

        let hist = FlimHistogram::from_ifd(
            &ifd_vec[UNCOMPRESSED_FRAME_NUM], 
            &mut f, 
            file_format.num_flim_tau_bins().unwrap()
        ).unwrap();

        let frame = SiffFrame::from_ifd(&ifd_vec[UNCOMPRESSED_FRAME_NUM], &mut f).unwrap();
        // Should have the same number of photons
        assert_eq!(hist.data.sum(), frame.intensity.fold(0, |running_sum, &x| running_sum + (x as u64)));
    }

    #[test]
    fn image_histogram_tests(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let TEST_FILE_PATH = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        // TODO implement when I actually start using these.
        let mut f = std::fs::File::open(TEST_FILE_PATH).unwrap();
        let file_format = FileFormat::parse_filetype(&mut f).unwrap();
        let _ifd_vec = file_format.get_ifd_vec(&mut f);

        let mut _hist = ImageHistogram {
            data : ArrayD::<u64>::zeros(IxDyn(&[file_format.num_flim_tau_bins().unwrap() as usize, 512, 512]))
        };

        let mut _reader = std::io::BufReader::new(f);
        // for (_i, ifd) in ifd_vec.iter().enumerate() {
        //     let curr_pos = reader.stream_position().unwrap();
        //     reader.seek(
        //         std::io::SeekFrom::Start(
        //             ifd.get_tag(StripOffsets)
        //             .unwrap().value().into()
        //         )
        //     ).unwrap();
        //     // TODO finish test
        //     //assert!(false);
        //     // match ifd.get_tag(Siff).unwrap().value().into() {
        //     //     0 => {
        //     //         _load_histogram_uncompressed(ifd, &mut reader, &mut hist.data.index_axis_mut(Axis(0), i)).unwrap();
        //     //     },
        //     //     1 => {
        //     //         _load_histogram_compressed(ifd, &mut reader, &mut hist.data.index_axis_mut(Axis(0), i)).unwrap();
        //     //     },
        //     //     _ => {
        //     //         panic!("Invalid Siff tag value");
        //     //     }
        //     // }
        //     reader.seek(std::io::SeekFrom::Start(curr_pos)).unwrap();
        // }
    }
}

