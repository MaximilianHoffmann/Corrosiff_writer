use bytemuck::try_cast_slice;
use ndarray::prelude::*;
use itertools::izip;

use std::io::{
    Error as IOError,
    ErrorKind as IOErrorKind,
};

use crate::data::image::{
    utils::photonwise_op,
    dimensions::macros::*
};

/// Loads an allocated array with data read from a raw
/// `.siff` format frame (presumes the `reader` argument already
/// points to the frame) by ADDING data!
/// 
/// # Arguments
/// 
/// * `array` - The array to load the data into viewed as a 2d array
/// * `strip_bytes` - The number of bytes in the strip
/// * `ydim` - The height of the frame
/// * `xdim` - The width of the frame
/// 
/// # Example
/// 
/// ```rust, ignore
/// use ndarray::prelude::*;
/// use std::io::BufReader;
/// 
/// let mut array = Array2::<u16>::zeros((512, 512));
/// let mut reader = BufReader::new(std::fs::File::open("file.siff").unwrap());
/// reader.seek(std::io::SeekFrom::Start(34238)).unwrap();
/// load_array_raw_s&iff(mut array, 512*512*2, 512, 512);
/// ```
/// 
/// # See also
/// 
/// `load_array_raw_siff_registered` - for loading an array
/// and shifting the data based on registration.
#[binrw::parser(reader)]
pub fn load_array_raw_siff<T : Into<u64>>(
        array : &mut ArrayViewMut2<u16>,
        strip_bytes : T,
        ydim : u32,
        xdim : u32,
    ) -> binrw::BinResult<()> {

    photonwise_op!(
        reader,
        strip_bytes,
        |photon : &u64| {
            array[
                [
                    photon_to_y!(photon, 0, ydim),
                    photon_to_x!(photon, 0, xdim),
                ]
            ] += 1;
        }
    );
    
    Ok(())
}

#[binrw::parser(reader)]
pub fn extract_mask_raw_siff<T : Into<u64>>(
    target_array : &mut ArrayViewMut1<u64>,
    mask : &ArrayView2<bool>,
    lookup_table : &ArrayView2<usize>,
    strip_bytes : T,
    ydim : u32,
    xdim : u32,
) -> binrw::BinResult<()> {
    photonwise_op!(
        reader,
        strip_bytes,
        |siffphoton| {
            let (y, x) = (
                photon_to_y!(siffphoton, 0, ydim),
                photon_to_x!(siffphoton, 0, xdim)
            );
            target_array[lookup_table[[y, x]]] += mask[[y, x]] as u64;
        }
    );
    Ok(())
}

/// Computes the sum of the intensity data in a raw frame
/// masked by a bool array and stores it by changing the
/// `frame_sum` argument.
#[binrw::parser(reader)]
pub fn sum_mask_raw_siff<T : Into<u64>>(
    frame_sum : &mut u64,
    mask : &ArrayView2<bool>,
    strip_bytes : T,
    ydim : u32,
    xdim : u32,
) -> binrw::BinResult<()> {
    
    photonwise_op!(
        reader,
        strip_bytes,
        |siffphoton : &u64| {
            *frame_sum += mask[
                (photon_to_y!(siffphoton, 0 , ydim),
                photon_to_x!(siffphoton, 0, xdim),)
            ] as u64
        }
    );

    Ok(())
}

/// Iterates over all the masks for each pixel
#[binrw::parser(reader)]
pub fn sum_masks_raw_siff<T : Into<u64>>(
    frame_sums : &mut ArrayViewMut1<u64>,
    masks : &ArrayView3<bool>,
    strip_bytes : T,
    ydim : u32,
    xdim : u32,
) -> binrw::BinResult<()> {

    photonwise_op!(
        reader,
        strip_bytes,
        |siffphoton : &u64| {
            masks.axis_iter(Axis(0)).zip(frame_sums.iter_mut())
            .for_each(|(mask, frame_sum)| {
                *frame_sum += mask[
                    (photon_to_y!(siffphoton, 0 , ydim),
                    photon_to_x!(siffphoton, 0, xdim),)
                ] as u64
            });
        } 
    );

    Ok(())
}

/// Parses a compressed `.siff` format frame and returns
/// an `Intensity` struct containing the intensity data.
/// 
/// Expected to be at the data strip, so it will go backwards by the size of the
/// intensity data and read that.
#[binrw::parser(reader)]
pub fn load_array_compressed_siff(
        array : &mut ArrayViewMut2<u16>,
        ydim : u32,
        xdim : u32
    ) -> binrw::BinResult<()> {
    
    reader.seek(std::io::SeekFrom::Current(
        -(ydim as i64 * xdim as i64 * std::mem::size_of::<u16>() as i64)
    ))?;
    
    let mut data : Vec<u8> = vec![0; 
        ydim as usize * xdim as usize * std::mem::size_of::<u16>()
    ];
    reader.read_exact(&mut data)?;

    let data = try_cast_slice::<u8, u16>(&data).map_err(|err| binrw::Error::Io(
        IOError::new(IOErrorKind::InvalidData, err))
    )?;

    array.iter_mut().zip(data.iter()).for_each(|(a, &v)| *a = v);

    Ok(())
}

#[binrw::parser(reader, endian)]
pub fn extract_mask_compressed_siff(
    target_array : &mut ArrayViewMut1<u64>,
    mask : &ArrayView2<bool>,
    lookup_table : &ArrayView2<usize>,
    ydim : u32,
    xdim : u32,
) -> binrw::BinResult<()> {
    
    let mut frame_array = Array2::<u16>::from_shape_vec(
        (ydim as usize, xdim as usize),
        vec![0; ydim as usize * xdim as usize]
    ).unwrap();
    load_array_compressed_siff(reader, endian,
        (&mut frame_array.view_mut(), ydim, xdim))?;
    
    for (&mask_px, &frame_px, &lookup_px) in izip!(
        mask.iter(), frame_array.iter(), lookup_table.iter()
    ) {
       target_array[lookup_px] += mask_px as u64 * frame_px as u64;
    }

    Ok(())
}

/// Computes the sum of the intensity data in a compressed frame
/// masked by a bool array and stores it by changing the
/// `frame_sum` argument.
#[binrw::parser(reader)]
pub fn sum_mask_compressed_siff(
    frame_sum : &mut u64,
    mask : &ArrayView2<bool>,
    ydim : u32,
    xdim : u32
) -> binrw::BinResult<()> {
        
        reader.seek(std::io::SeekFrom::Current(
            -(ydim as i64 * xdim as i64 * std::mem::size_of::<u16>() as i64)
        ))?;
        
        let mut data : Vec<u8> = vec![0; 
            ydim as usize * xdim as usize * std::mem::size_of::<u16>()
        ];
        reader.read_exact(&mut data)?;
    
        let data = try_cast_slice::<u8, u16>(&data).map_err(|err| binrw::Error::Io(
            IOError::new(IOErrorKind::InvalidData, err))
        )?;

        // data.iter().zip(mask.iter()).for_each(|(&d, m)| {
        //     if *m {*frame_sum += d as u64}
        // });
    
        data.iter().zip(mask.iter()).for_each(|(&d, m)| {
            *frame_sum += (d as u64) * (*m as u64);
        });
    
        Ok(())
}

/// Iterates over all the masks for each pixel
#[binrw::parser(reader)]
pub fn sum_masks_compressed_siff(
    frame_sums : &mut ArrayViewMut1<u64>,
    masks : &ArrayView3<bool>,
    ydim : u32,
    xdim : u32,
) -> binrw::BinResult<()> {

    reader.seek(std::io::SeekFrom::Current(
        -(ydim as i64 * xdim as i64 * std::mem::size_of::<u16>() as i64)
    ))?;
    
    let mut data : Vec<u8> = vec![0; 
        ydim as usize * xdim as usize * std::mem::size_of::<u16>()
    ];
    reader.read_exact(&mut data)?;

    let data = try_cast_slice::<u8, u16>(&data).map_err(|err| binrw::Error::Io(
        IOError::new(IOErrorKind::InvalidData, err))
    )?;

    // Seems bad to iterate over data N times!!
    masks.axis_iter(Axis(0)).zip(frame_sums.iter_mut()).for_each(
        |(mask, mask_sum)| {
            data.iter().zip(mask.iter()).for_each(
                |(&d, mask_px)| {
                    *mask_sum += (d as u64) * (*mask_px as u64);
                }
            )
        }
    );

    // .zip(data.iter()).for_each(
    //     |(masks_pxs, &d)|
    //     masks_pxs.zip(frame_sums.iter_mut()).for_each(|(mask_pixel, maskwise_sum)| {
    //         *maskwise_sum += (d as u64) * (*mask_pixel as u64);
    //     })
    // );

    Ok(())
}