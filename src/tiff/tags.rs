//! `Tags`
//! 
//! This submodule implements information pertaining to
//! the individual types of Tags, which are either stored
//! as vectors in the tag itself or as pointers to the
//! data in the file.

use std::convert::TryFrom;
use std::fmt::Display;
use binrw::BinRead;

/// The ValueType field can either
/// be of size u32 or u64, depending
/// on the file format.
// pub enum ValueType {
//     U32,
//     U64,
// }

/// The `unwrap_tag_as!` macro is a convenience macro
/// for unwrapping a tag and converting it to a specific
/// type. Use: `unwrap_tag_as!(ifd, tag, into)`
/// 
/// ## Arguments
/// 
/// * `ifd` - The IFD to get the tag from
/// * `tag` - The tag to get
/// * `into` - The type to convert the tag to
#[macro_export]
macro_rules! unwrap_tag_as {
    ($ifd:expr, $tag:expr, $into : tt) => {
        $ifd.get_tag($tag).unwrap().value().into() as $into
    };
}

/// The `Tag` trait is implemented by the `TiffTag` and
/// `BigTag` struct types, which contain the same types of
/// fields but different data sizes.
pub trait Tag {
    type ValueType : Into<u64>; // Either a u32, u64, or a pointer
    fn tag(&self) -> TiffTagID;
    fn tag_dtype(&self) -> TiffTagType;
    fn num_values(&self) -> Self::ValueType;
    fn value(&self) -> Self::ValueType;
    /// Parses the tag data from its raw reads into
    /// real data. For many data types, that's simply
    /// returning the value, but for some it may be
    /// a transformation of the data.
    fn parse(&self);
    fn to_string(&self) -> String {
        format!("{}", self.tag())
    }
    fn sizeof() -> usize;
}

#[derive(BinRead, Debug)]
#[br(little)]
pub struct TiffTag {
    #[br(map  = |x: u16| TiffTagID::try_from(x).unwrap())]
    //#[bw(map = |x : &TiffTagID| x.into())]
    pub tag : TiffTagID,
    #[br(map = |x: u16| TiffTagType::try_from(x).unwrap())]
    //#[bw(map = |x : &TiffTagType| x.into())]
    pub tag_dtype : TiffTagType,
    pub num_values : u32,
    pub value : u32,
}

impl Tag for TiffTag {
    type ValueType = u32;

    fn tag(&self) -> TiffTagID {
        self.tag
    }

    fn tag_dtype(&self) -> TiffTagType {
        self.tag_dtype
    }

    fn num_values(&self) -> u32 {
        self.num_values
    }

    fn value(&self) -> Self::ValueType {
        self.value.clone()
    }

    fn parse(&self) {
    }

    fn sizeof() -> usize {
        std::mem::size_of::<u16>() 
        + std::mem::size_of::<u16>()
        + std::mem::size_of::<u32>()
        + std::mem::size_of::<u32>()
    }
}

#[derive(BinRead, Debug)]
#[br(little)]
pub struct BigTag {
    #[br(map  = |x: u16| TiffTagID::try_from(x).unwrap())]
    //#[bw(map = |x : &TiffTagID| x.into())]
    pub tag : TiffTagID,
    #[br(map = |x: u16| TiffTagType::try_from(x).unwrap())]
    //#[bw(map = |x : &TiffTagType| x.into())]
    pub tag_dtype : TiffTagType,
    //bytes_per_value : u64,
    pub num_values : u64,
    pub value : u64,
}

impl Tag for BigTag {
    type ValueType = u64;

    fn tag(&self) -> TiffTagID {
        self.tag
    }

    fn tag_dtype(&self) -> TiffTagType {
        self.tag_dtype
    }

    fn num_values(&self) -> u64 {
        self.num_values
    }

    fn value(&self) -> Self::ValueType {
        self.value
    }

    fn parse(&self) {
        // Do nothing for now
    }

    fn sizeof() -> usize {
        std::mem::size_of::<u16>() 
        + std::mem::size_of::<u16>()
        + std::mem::size_of::<u64>()
        + std::mem::size_of::<u64>()
    }
}

/// The `TiffTagID` enum contains the possible tag
/// identifiers for the Tiff file format.
/// 
/// ## Variants
/// 
/// * `ImageWidth` - The width of the image in pixels (x dim).
/// * `ImageLength` - The length of the image in pixels (y dim).
/// * `BitsPerSample` - The number of bits per sample.
/// * `Compression` - The type of compression used.
/// * `PhotometricInterpretation` - The color space of the image.
/// * `ImageDescription` - Actually points to the end of the IFD. Absolute terms, not relative.
/// * `StripOffsets` - The address of the data strip (for `siff`,
/// this is the beginning of the photon count data). In absolute terms, not relative
/// * `Orientation` - The orientation of the image.
/// * `SamplesPerPixel` - The number of samples per pixel (meaningless for .siff).
/// * `RowsPerStrip` - The number of rows per strip.
/// * `StripByteCounts` - The number of bytes in each strip.
/// * `XResolution` - The x resolution of the image (in resolution unit).
/// * `YResolution` - The y resolution of the image (in resolution unit).
/// * `PlanarConfiguration` - The planar configuration of the image.
/// * `ResolutionUnit` - The resolution unit of the image.
/// * `Software` - The address of the NVFD for the file
/// * `DateTime` - The date and time the file was created?
/// * `Artist` - The address of the ROI data.
/// * `Predictor` - The predictor used for the image.
/// * `ExtraSamples` - The number of extra samples.
/// * `SampleFormat` - The format of the samples.
/// * `Siff` - SiffCompressed form if 1, straight photon counts if 0.
/// 
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TiffTagID {
    ImageWidth, // x
    ImageLength, // y
    BitsPerSample,
    Compression,
    PhotometricInterpretation,
    ImageDescription, // End of IFD
    StripOffsets, // Data strip address
    Orientation,
    SamplesPerPixel,
    RowsPerStrip,
    StripByteCounts,
    XResolution,
    YResolution,
    PlanarConfiguration,
    ResolutionUnit,
    Software, // NVFD address
    DateTime,
    Artist, // ROI address
    Predictor,
    ExtraSamples,
    SampleFormat,
    Siff, // SiffCompress if 1
}

impl From<TiffTagID> for u16 {
    fn from(tag: TiffTagID) -> Self {
        match tag {
            TiffTagID::ImageWidth => 256,
            TiffTagID::ImageLength => 257,
            TiffTagID::BitsPerSample => 258,
            TiffTagID::Compression => 259,
            TiffTagID::PhotometricInterpretation => 262,
            TiffTagID::ImageDescription => 270,
            TiffTagID::StripOffsets => 273,
            TiffTagID::Orientation => 274,
            TiffTagID::SamplesPerPixel => 277,
            TiffTagID::RowsPerStrip => 278,
            TiffTagID::StripByteCounts => 279,
            TiffTagID::XResolution => 282,
            TiffTagID::YResolution => 283,
            TiffTagID::PlanarConfiguration => 284,
            TiffTagID::ResolutionUnit => 296,
            TiffTagID::Software => 305,
            TiffTagID::DateTime => 306,
            TiffTagID::Artist => 315,
            TiffTagID::Predictor => 317,
            TiffTagID::ExtraSamples => 338,
            TiffTagID::SampleFormat => 339,
            TiffTagID::Siff => 907,
        }
    }
}

// impl Into<u16> for TiffTagID {
//     fn into(self) -> u16 {
//         match self {
//             TiffTagID::ImageWidth => 256,
//             TiffTagID::ImageLength => 257,
//             TiffTagID::BitsPerSample => 258,
//             TiffTagID::Compression => 259,
//             TiffTagID::PhotometricInterpretation => 262,
//             TiffTagID::ImageDescription => 270,
//             TiffTagID::StripOffsets => 273,
//             TiffTagID::Orientation => 274,
//             TiffTagID::SamplesPerPixel => 277,
//             TiffTagID::RowsPerStrip => 278,
//             TiffTagID::StripByteCounts => 279,
//             TiffTagID::XResolution => 282,
//             TiffTagID::YResolution => 283,
//             TiffTagID::PlanarConfiguration => 284,
//             TiffTagID::ResolutionUnit => 296,
//             TiffTagID::Software => 305,
//             TiffTagID::DateTime => 306,
//             TiffTagID::Artist => 315,
//             TiffTagID::Predictor => 317,
//             TiffTagID::ExtraSamples => 338,
//             TiffTagID::SampleFormat => 339,
//             TiffTagID::Siff => 907,
//         }
//     }
// }

impl TryFrom<u16> for TiffTagID {
    type Error = ();

    fn try_from(v: u16) -> Result<Self, Self::Error> {
        match v {
            256 => Ok(TiffTagID::ImageWidth),
            257 => Ok(TiffTagID::ImageLength),
            258 => Ok(TiffTagID::BitsPerSample),
            259 => Ok(TiffTagID::Compression),
            262 => Ok(TiffTagID::PhotometricInterpretation),
            270 => Ok(TiffTagID::ImageDescription),
            273 => Ok(TiffTagID::StripOffsets),
            274 => Ok(TiffTagID::Orientation),
            277 => Ok(TiffTagID::SamplesPerPixel),
            278 => Ok(TiffTagID::RowsPerStrip),
            279 => Ok(TiffTagID::StripByteCounts),
            282 => Ok(TiffTagID::XResolution),
            283 => Ok(TiffTagID::YResolution),
            284 => Ok(TiffTagID::PlanarConfiguration),
            296 => Ok(TiffTagID::ResolutionUnit),
            305 => Ok(TiffTagID::Software),
            306 => Ok(TiffTagID::DateTime),
            315 => Ok(TiffTagID::Artist),
            317 => Ok(TiffTagID::Predictor),
            338 => Ok(TiffTagID::ExtraSamples),
            339 => Ok(TiffTagID::SampleFormat),
            907 => Ok(TiffTagID::Siff),
            _ => Err(()),
        }
    }
}

impl Display for TiffTagID {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TiffTagType {
    Byte,
    Ascii,
    Short,
    Long,
    Rational,
    SByte,
    Undefined,
    SShort,
    SLong,
    SRational,
    Float,
    Double,
    Long8, // BigTiff only
    SLong8, // BigTiff only
    IFD8, // BigTiff only
}

// impl Into<u16> for TiffTagType {
//     fn into(self) -> u16 {
//         match self {
//             TiffTagType::Byte => 1,
//             TiffTagType::Ascii => 2,
//             TiffTagType::Short => 3,
//             TiffTagType::Long => 4,
//             TiffTagType::Rational => 5,
//             TiffTagType::SByte => 6,
//             TiffTagType::Undefined => 7,
//             TiffTagType::SShort => 8,
//             TiffTagType::SLong => 9,
//             TiffTagType::SRational => 10,
//             TiffTagType::Float => 11,
//             TiffTagType::Double => 12,
//             TiffTagType::Long8 => 16,
//             TiffTagType::SLong8 => 17,
//             TiffTagType::IFD8 => 18,
//         }
//     }
// }

impl From<TiffTagType> for u16 {
    fn from(tag: TiffTagType) -> Self {
        match tag {
            TiffTagType::Byte => 1,
            TiffTagType::Ascii => 2,
            TiffTagType::Short => 3,
            TiffTagType::Long => 4,
            TiffTagType::Rational => 5,
            TiffTagType::SByte => 6,
            TiffTagType::Undefined => 7,
            TiffTagType::SShort => 8,
            TiffTagType::SLong => 9,
            TiffTagType::SRational => 10,
            TiffTagType::Float => 11,
            TiffTagType::Double => 12,
            TiffTagType::Long8 => 16,
            TiffTagType::SLong8 => 17,
            TiffTagType::IFD8 => 18,
        }
    }
}


impl TryFrom<u16> for TiffTagType {
    type Error = ();

    fn try_from(v: u16) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(TiffTagType::Byte),
            2 => Ok(TiffTagType::Ascii),
            3 => Ok(TiffTagType::Short),
            4 => Ok(TiffTagType::Long),
            5 => Ok(TiffTagType::Rational),
            6 => Ok(TiffTagType::SByte),
            7 => Ok(TiffTagType::Undefined),
            8 => Ok(TiffTagType::SShort),
            9 => Ok(TiffTagType::SLong),
            10 => Ok(TiffTagType::SRational),
            11 => Ok(TiffTagType::Float),
            12 => Ok(TiffTagType::Double),
            16 => Ok(TiffTagType::Long8),
            17 => Ok(TiffTagType::SLong8),
            18 => Ok(TiffTagType::IFD8),
            _ => Err(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use binrw::io::Cursor;
    #[test]
    fn read_tiff_tags() {
        let data = [
            0x00, 0x01, // Tag ID
            0x01, 0x00, // Tag Type
            0x01, 0x00, 0x00, 0x00, // Num Values
            0x01, 0x00, 0x00, 0x00, // Value
        ];
        let mut cursor = Cursor::new(&data);
        let tag = TiffTag::read(&mut cursor).unwrap();
        assert_eq!(tag.tag(), TiffTagID::ImageWidth);
        assert_eq!(tag.tag_dtype(), TiffTagType::Byte);
        assert_eq!(tag.num_values(), 1);
        assert_eq!(tag.value(), 1);
    }

    #[test]
    fn read_bigtiff_tags(){
        let data = [
            0x00, 0x01, // Tag ID
            0x01, 0x00, // Tag Type
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // Num Values
            0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,// Value
        ];
        let mut cursor = Cursor::new(&data);
        let tag = BigTag::read(&mut cursor).unwrap();
        assert_eq!(tag.tag(), TiffTagID::ImageWidth);
        assert_eq!(tag.tag_dtype(), TiffTagType::Byte);
        assert_eq!(tag.num_values(), 1);
        assert_eq!(tag.value(), (1<<32) + 1);
    }
}