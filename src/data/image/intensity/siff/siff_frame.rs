use ndarray::prelude::*;
use binrw::io::{Read, Seek, Write};
use binrw::BinWrite;

use std::io::{
    Error as IOError,
    ErrorKind as IOErrorKind,
};

use crate::data::image::Image;
use crate::{
    tiff::{
        IFD,
        TiffTagID::*,
        Tag,
    },
    data::image::
    intensity::siff::{
        raw_siff_parser,
        compressed_siff_parser,
    },
};

/// A local struct for reading directly.
/// Only used internally for testing.
#[allow(dead_code)]
pub struct SiffFrame{
    pub intensity : ndarray::Array2<u16>,
}

impl SiffFrame {
    /// Parses a frame from a `.siff` file being viewed by
    /// `reader` using the metadata in the `ifd` argument
    /// to return a `SiffFrame` struct containing the intensity.
    /// 
    /// Does not move the `Seek` position of the reader because it
    /// is restored to its original position after reading the frame.
    /// 
    /// ## Arguments
    /// 
    /// * `ifd` - The IFD of the frame to load
    /// 
    /// * `reader` - The reader of the `.siff` file
    /// 
    /// ## Returns
    /// 
    /// * `Result<SiffFrame, IOError>` - A `SiffFrame` struct containing the intensity data
    /// for the requested frame.
    /// 
    /// ## Errors
    /// 
    /// * `IOError` - If the frame cannot be read for any reason
    /// this will throw an `IOError`
    #[allow(dead_code)]
    pub fn from_ifd<'a, 'b, I, ReaderT>(ifd : &'a I, reader : &'b mut ReaderT) 
    -> Result<Self, IOError> where I : IFD, ReaderT : Read + Seek {
        let cur_pos = reader.stream_position()?;

        reader.seek(
        std::io::SeekFrom::Start(
                ifd.get_tag(StripOffsets)
                .ok_or(
                    IOError::new(IOErrorKind::InvalidData, "Strip offset not found")
                )?.value().into()
            )
        ).or_else(|e| {reader.seek(std::io::SeekFrom::Start(cur_pos)).unwrap(); Err(e)})?;

        let parsed = match ifd.get_tag(Siff).unwrap().value().into() {
            0 => {
                raw_siff_parser(reader, binrw::Endian::Little,
                (
                    ifd.get_tag(StripByteCounts).unwrap().value(),
                    ifd.height().unwrap().into() as u32,
                    ifd.width().unwrap().into() as u32,
                )
            )},
            1 => {
                compressed_siff_parser(reader, binrw::Endian::Little, 
                (
                    ifd.height().unwrap().into() as u32,
                    ifd.width().unwrap().into() as u32,
                )
            )},
            _ => {Err(
                binrw::error::Error::Io(IOError::new(
                    IOErrorKind::InvalidData, "Invalid Siff tag")
                ))
            }
        }
        .map_err(|err| {
            reader.seek(std::io::SeekFrom::Start(cur_pos)).unwrap_or(0);
            IOError::new(IOErrorKind::InvalidData, err)
        })?;

        reader.seek(std::io::SeekFrom::Start(cur_pos)).unwrap_or(0);

        Ok(SiffFrame {
            intensity : parsed
        })
    }

    /// Write the intensity data pointed to by an IFD into a file.
    /// First writes the head of the IFD, then drops the data in place
    /// as a flat array of u16s.
    pub fn write_frame_as_tiff
    <ReaderT : Read + Seek, WriterT : Write + Seek, I : IFD>
    (reader : &mut ReaderT, writer : &mut WriterT, ifd : &I)
    -> binrw::io::Result<()> {

        let start_of_write_ifd = writer.stream_position()?;
        // Parse the array before anything -- error if
        // there's a problem.
        let frame = SiffFrame::from_ifd(ifd, reader)?;

        let num_tags = if ifd.get_tag(Siff).is_none() {
            ifd.num_tags()
        }
        else {
            ifd.num_tags() - 1
        };

        writer.write(&((num_tags as u64).to_le_bytes()))?;

        let image_size : u64 = ifd.width().unwrap().into() * ifd.height().unwrap().into() * std::mem::size_of::<u16>() as u64;
        let _bytes_per_tag = ifd.size_of_tag() as u64;

        let end_of_ifd = 
            start_of_write_ifd +
            std::mem::size_of::<u64>() as u64 // num_tags size
            + (num_tags as u64)*20 // so is this what's wrong?
            + std::mem::size_of::<I::PointerSize>() as u64; // next_ifd size

        let description_length = match ifd.get_tag(Siff) {
            None => {
                Ok(
                    ifd.get_tag(StripOffsets).unwrap().value().into()
                    - ifd.get_tag(ImageDescription).unwrap().value().into() as u64
                )
            }
            Some(tag) => {
                match tag.value().into() {
                    0 => {
                        Ok(ifd.get_tag(StripOffsets).unwrap().value().into() as u64
                        - ifd.get_tag(ImageDescription).unwrap().value().into() as u64
                        )
                    }
                    1 => {
                        Ok(ifd.get_tag(StripOffsets).unwrap().value().into() as u64
                        - ifd.get_tag(ImageDescription).unwrap().value().into() as u64
                        - image_size
                        )
                    },
                    _ => {
                        Err(IOError::new(IOErrorKind::InvalidData, "Invalid Siff tag"))
                    }
                }
            }
        }?;

        // Most tags can be copied, but not all of them.
        // This code is a mess and it's all my fault.
        for tag in ifd.tags() {
            match tag.tag() {
                Siff => {},
                ImageDescription => {
                    writer.write(&((Into::<u16>::into(tag.tag()).to_le_bytes()))).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                    writer.write(&(end_of_ifd.to_le_bytes())).unwrap();
                },
                StripOffsets => {
                    writer.write(&(Into::<u16>::into(tag.tag()).to_le_bytes())).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                    writer.write(&((end_of_ifd + description_length).to_le_bytes())).unwrap();
                },
                StripByteCounts => {
                    writer.write(&((Into::<u16>::into(tag.tag()).to_le_bytes()))).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                    writer.write(&(image_size.to_le_bytes())).unwrap(); 
                },
                BitsPerSample => {
                    writer.write(&(Into::<u16>::into(tag.tag())).to_le_bytes()).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype())).to_le_bytes()).unwrap();
                    writer.write(&1u64.to_le_bytes()).unwrap();
                    writer.write(&16u64.to_le_bytes()).unwrap();
                },
                SampleFormat => {
                    writer.write(&(Into::<u16>::into(tag.tag())).to_le_bytes()).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                },
                SamplesPerPixel => {
                    writer.write(&(Into::<u16>::into(tag.tag()).to_le_bytes())).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                },
                Compression => {
                    writer.write(&(Into::<u16>::into(tag.tag()).to_le_bytes())).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                },
                PhotometricInterpretation => {
                    writer.write(&(Into::<u16>::into(tag.tag()).to_le_bytes())).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                    writer.write(&(1u64.to_le_bytes())).unwrap();
                },
                _ => {
                    writer.write(&(Into::<u16>::into(tag.tag()).to_le_bytes())).unwrap();
                    writer.write(&(Into::<u16>::into(tag.tag_dtype()).to_le_bytes())).unwrap();
                    writer.write(&(tag.num_values().into().to_le_bytes())).unwrap();
                    writer.write(&(tag.value().into().to_le_bytes())).unwrap();
                }
            }
        };
        let pos = writer.stream_position()?;
        // write the next IFD location
        let mut next_ifd : u64 = pos + image_size + description_length + std::mem::size_of::<I::PointerSize>() as u64;
        if ifd.next_ifd().is_none() || (ifd.next_ifd().unwrap().into() == 0) {
            next_ifd = 0;
        }
        writer.write_all(&next_ifd.to_le_bytes()).unwrap();

        debug_assert_eq!(writer.stream_position()?, end_of_ifd);

        // write the image description
        reader.seek(
            std::io::SeekFrom::Start(
                ifd.get_tag(ImageDescription).unwrap().value().into()
            )
        )?;
        let mut description = vec![0u8; description_length as usize];
        reader.read_exact(&mut description)?;

        writer.write_all(&description)?;

        unsafe{
            writer.write_all(std::slice::from_raw_parts(
                frame.intensity.as_ptr() as *const u8,
                2*frame.intensity.len()
            ))?;
        };

        // debug_assert_eq!(amount_written, image_size as usize);
        debug_assert_eq!(writer.stream_position()?, next_ifd);

        Ok(())
    }
}

/// Arbitrary dimensional siff intensity data
/// Not implemented for now...
#[allow(dead_code)]
pub struct SiffArray<D> {
    pub array : Array<u16, D>,
}