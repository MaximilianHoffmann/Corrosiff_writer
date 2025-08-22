//! The `Metadata` struct parses the microscope acquisition
//! parameters and metadata from the various strings they are
//! stored in scattered throughout the file. Most of the data
//! is read from the `nvfd` -- non-varying-frame-data -- string,
//! but information about frame times comes from data just before
//! each individual IFD (for example).
use std::io::{Read,Seek};
use crate::{
    tiff::{
        IFD, Tag,
        TiffTagID::*,
    },
    unwrap_tag_as,
    CorrosiffError,
};

/// Parses the metadata string for a given field.
/// TODO: Learn procedural macros, and do it with
/// one of these!
/// 
/// `metadata_string_field` parses the metadata string,
/// extracts the component corresponding to the requested
/// field, and returns it as the requested type. If the
/// requested type is `Option<T>`, then it will return an
/// `Option` that is `None` if the field is not found.
/// 
/// If an `Option` is not used for the requested type, then
/// the function will panic if the field is not found.
/// 
/// ## Panics
/// 
/// - If the field is not found and the requested type is not
/// `Option<T>`, the function will panic.
macro_rules! metadata_string_field {
    ($field:ident, $string:expr, Option<$cast_to:ty>) => {{
        let start = match $field.find($string) {
            Some(start) => start + $string.len(),
            None => return None,
        };
        let end = match $field[start..].find("\n") {
            Some(end) => end + start,
            None => return None,
        };
        $field[start..end].trim().parse::<$cast_to>().ok()
    }};

    ($field:ident, $string:expr, $cast_to: ty) => {{
        let start = $field.find($string).unwrap() 
            + $string.len();
        let end = $field[start..].find("\n").unwrap() + start;
        $field[start..end].trim().parse::<$cast_to>().unwrap()
    }};

    ($field : ident, $target_start : expr, $target_end : expr, Option<$cast_to : ty>) => {{
        let start = match $field.find($target_start) {
            Some(start) => start + $target_start.len(),
            None => return None,
        };
        let end = match $field[start..].find($target_end) {
            Some(end) => end + start,
            None => return None,
        };
        $field[start..end].trim().parse::<$cast_to>().ok()
    }};

    ($field : ident, $target_start : expr, $target_end : expr, $cast_to : ty) => {{
        let start = $field.find($target_start).unwrap() 
        + $target_start.len();
        let end = $field[start..].find($target_end).unwrap() + start;
        $field[start..end].trim().parse::<$cast_to>().unwrap()
    }};
}

/// The `Metadata` struct is a `Rust`
/// object that holds important or relevant
/// metadata in a human-interpretable format.
/// 
/// It reads the tiff IFD data and also extracts
/// a string containing more frame-specific metadata
/// (e.g. frame timestamps). This struct is mostly
/// for debugging -- it's a relatively inefficient way
/// of extracting this information.
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    pub width : u64,
    pub height : u64,
    pub bits_per_sample : u16,
    pub compression : u16,
    pub photometric_interpretation : u16,
    pub end_of_ifd : u64,
    pub data_offset : u64,
    pub orientation : u16,
    pub samples_per_pixel : u16,
    pub rows_per_strip : u64,
    pub strip_byte_counts : u64,
    pub x_resolution : u64,
    pub y_resolution : u64,
    pub resolution_unit : u16,
    pub nvfd_address : u64,
    pub roi_address : u64,
    pub sample_format : u16,
    pub siff_compress : Option<u16>,
    pub metadata_string : String,
}

impl FrameMetadata {
    /// Creates a new `FrameMetadata` object from an `IFD`.
    /// Because the `IFD` alone does not contain the long metadata
    /// string, this is not a complete metadata object. But its
    /// usable enough for most purposes.
    /// 
    /// ## Arguments
    /// 
    /// * `ifd` - The IFD to create the metadata from
    /// 
    /// ## Returns
    /// 
    /// A new `FrameMetadata` object
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// use crate::tiff::FileFormat;
    /// use crate::metadata::FrameMetadata;
    /// // `f` is a `File` object
    /// let file_format = FileFormat::parse_filetype(&mut f).unwrap();
    /// /* - snip - */
    /// let ifds = file_format.get_ifd_vec(&mut f);
    /// /* - snip - */
    /// let metadata = FrameMetadata::from_ifd(&ifds[15]);
    pub fn from_ifd<I : IFD>(ifd : &I) -> Self {
        FrameMetadata{
            width : unwrap_tag_as!(ifd, ImageWidth, u64),
            height : unwrap_tag_as!(ifd, ImageLength, u64),
            bits_per_sample : unwrap_tag_as!(ifd, BitsPerSample, u16),
            compression : unwrap_tag_as!(ifd, Compression, u16),
            photometric_interpretation : unwrap_tag_as!(ifd, PhotometricInterpretation, u16), 
            end_of_ifd  : unwrap_tag_as!(ifd, ImageDescription, u64),
            data_offset : unwrap_tag_as!(ifd, StripOffsets, u64),
            orientation : unwrap_tag_as!(ifd, Orientation, u16),
            samples_per_pixel : unwrap_tag_as!(ifd, SamplesPerPixel, u16),
            rows_per_strip : unwrap_tag_as!(ifd, RowsPerStrip, u64),
            strip_byte_counts : unwrap_tag_as!(ifd, StripByteCounts, u64),
            x_resolution : unwrap_tag_as!(ifd, XResolution, u64),
            y_resolution : unwrap_tag_as!(ifd, YResolution, u64),
            resolution_unit : unwrap_tag_as!(ifd, ResolutionUnit, u16),
            nvfd_address : unwrap_tag_as!(ifd, Software, u64),
            roi_address : unwrap_tag_as!(ifd, Artist, u64),
            sample_format : unwrap_tag_as!(ifd, SampleFormat, u16),
            siff_compress : ifd.get_tag(Siff).map(|x| x.value().into() as u16),
            metadata_string : String::new(),
        }
    }

    /// Fully-parsed metadata, needs a file to read the metadata string from.
    /// Returns the reader to its original position if successful.
    /// 
    /// ## Arguments
    /// 
    /// * `ifd` - The IFD to create the metadata from
    /// 
    /// * `reader` - The reader to read the metadata string from
    /// 
    /// ## Returns
    /// 
    /// A new `FrameMetadata` object with a populated metadata string
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::IOError` - If there is an error reading the metadata string
    /// due to an invalid `Siff` tag.
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// use std::fs::File;
    /// use std::io::BufReader;
    /// use crate::tiff::FileFormat;
    /// 
    /// let f = File::open("file.siff").unwrap();
    /// let reader = BufReader::new(f);
    /// let file_format = FileFormat::parse_filetype(&mut reader).unwrap();
    /// let ifds = file_format.get_ifd_vec(&mut reader);
    /// let metadata = FrameMetadata::from_ifd_and_file(&ifds[15], &reader).unwrap();
    /// 
    /// println!("{}", metadata.metadata_string);
    /// ```
    pub fn from_ifd_and_file<I : IFD, ReaderT : Read + Seek>(ifd : &I, reader : &mut ReaderT)
        -> Result<Self, CorrosiffError> {
        let mut metadata = FrameMetadata::from_ifd(ifd);
        metadata.metadata_string = Self::metadata_string(ifd, reader);
        Ok(metadata)
    }

    /// Reads a metadata string from the file without needing to
    /// create a `FrameMetadata` object. Returns
    /// the reader to its original position if successful.
    /// 
    /// ## Arguments
    /// 
    /// * `ifd` - The IFD to read the metadata string from
    /// 
    /// * `reader` - The reader to read the metadata string from
    /// 
    /// ## Returns
    /// 
    /// A `String` containing the metadata string
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// use std::fs::File;
    /// use std::io::BufReader;
    /// use crate::tiff::FileFormat;
    /// 
    /// let f = File::open("file.siff").unwrap();
    /// let reader = BufReader::new(f);
    /// let file_format = FileFormat::parse_filetype(&mut reader).unwrap();
    /// let ifds = file_format.get_ifd_vec(&mut reader);
    /// let metadata_string = FrameMetadata::metadata_string(&ifds[15], &reader);
    /// println!("{}", metadata_string);
    /// ```
    pub fn metadata_string<I : IFD, ReaderT : Read + Seek>(ifd : &I, reader : &mut ReaderT)->String {
        let string_length : u64;
        if ifd.get_tag(Siff).is_none() {
            string_length = unwrap_tag_as!(ifd, StripOffsets, u64)
            - unwrap_tag_as!(ifd, ImageDescription, u64);
        }
        else{
            string_length = match unwrap_tag_as!(ifd, Siff, u16) {
                0 => {
                    unwrap_tag_as!(ifd, StripOffsets, u64)
                    - unwrap_tag_as!(ifd, ImageDescription, u64)
                },
                1 => {
                    unwrap_tag_as!(ifd, StripOffsets, u64)
                    - unwrap_tag_as!(ifd, ImageDescription, u64)
                    - unwrap_tag_as!(ifd, ImageWidth, u64)
                    * unwrap_tag_as!(ifd, ImageLength, u64)
                    * std::mem::size_of::<u16>() as u64
                },
                _ => return "Invalid Siff compression value".to_string(),
            };
        };
        let curr_pos = reader.stream_position().unwrap();
        reader.seek(std::io::SeekFrom::Start(unwrap_tag_as!(ifd, ImageDescription, u64))).unwrap();
        let mut metadata_string = vec![0u8; string_length as usize];
        reader.read_exact(&mut metadata_string).unwrap();
        reader.seek(std::io::SeekFrom::Start(curr_pos)).unwrap();
        String::from_utf8(metadata_string).unwrap()
    }

    /// Public function for extracting the frame number from the metadata string
    /// without constructing an entire `FrameMetadata` object.
    pub fn frame_number_from_metadata_str(string : &str) -> u64 {
        metadata_string_field!(string, "frameNumbers = ", u64)
    }

    /// Public function for extracting the frame time from the metadata string
    /// without constructing an entire `FrameMetadata` object.
    pub fn frame_time_experiment_from_metadata_str(string : &str) -> f64 {
        metadata_string_field!(string, "frameTimestamps_sec = ", f64)
    }

    /// Public function for extracting the frame epoch from the metadata string
    /// without constructing an entire `FrameMetadata` object.
    pub fn frame_time_epoch_from_metadata_str(string : &str) -> u64 {
        metadata_string_field!(string, "\nepoch = ", u64)
    }

    /// Public function for extracting the most recent system time from the metadata string
    /// without constructing an entire `FrameMetadata` object.
    pub fn most_recent_system_time_from_metadata_str(string : &str) -> Option<u64> {
        metadata_string_field!(string, "mostRecentSystemTimestamp_epoch = ", Option<u64>)
    }

    /// Public function for extracting the sync stamps from the metadata string
    /// without constructing an entire `FrameMetadata` object.
    pub fn sync_stamps_from_metadata_str(string : &str) -> u64 {
        metadata_string_field!(string, "sync Stamps = ", u64)
    }

    /// Public function for extracting the appended text from the metadata string
    /// without constructing an entire `FrameMetadata` object.
    /// 
    /// ## Returns
    /// 
    /// A tuple containing the full appended text, and the timestamp (if one is present).
    /// Is `None` for any string in which there is no appended text.
    pub fn appended_text_from_metadata_str(string : &str)->Option<(String, Option<f64>)> {
        let full_string = metadata_string_field!(string, "\nAppended text = ", Option<String>)?;
        let timestamp = metadata_string_field!(string, "\nText timestamp = ", Option<f64>);
        Some((full_string, timestamp))
    }
}

/// Returns the timestamps for a set of frames in _experiment time_,
/// which is seconds since the image acquisition began.
/// 
/// ## Arguments
/// 
/// * `ifds` - A slice of `IFD` objects corresponding to the
/// frames to get timestamps from
/// 
/// * `reader` - An object with read access to the file
/// 
/// ## Returns
/// 
/// A `Vec<f64>` containing the timestamps for each frame
/// in seconds since the image acquisition began, in the order
/// of the requested `ifd` slice.
pub fn get_experiment_timestamps<I : IFD, ReaderT : Read + Seek>(
    ifds: &[&I],
    reader : &mut ReaderT
    ) -> Vec<f64> {
    ifds.iter().map(|&ifd| 
        FrameMetadata::frame_time_experiment_from_metadata_str(
            &FrameMetadata::metadata_string(ifd, reader)
        )
    ).collect()
}

/// Returns the timestamps for a set of frames in _epoch time_,
/// which is nanoseconds since the Unix epoch. This is estimated
/// using the number of laser pulses at the moment of the frame
/// trigger, and the estimated pulse rate of the laser. This should
/// be extremely regular, but the pulse rate estimate may be slightly
/// off compared to the system clock and so this is _low jitter_ and
/// _high drift_.
/// 
/// ## Arguments
/// 
/// * `ifds` - A slice of `IFD` objects corresponding to the
/// frames to get timestamps from
/// 
/// * `reader` - An object with read access to the file
/// 
/// ## Returns
/// 
/// A `Vec<u64>` containing the timestamps for each frame
/// in nanoseconds since the Unix epoch, in the order of the
/// requested `ifd` slice.
pub fn get_epoch_timestamps_laser<I : IFD, ReaderT : Read + Seek>(
    ifds: &[&I],
    reader : &mut ReaderT
    ) -> Vec<u64> {
    ifds.iter().map(|&ifd| 
        FrameMetadata::frame_time_epoch_from_metadata_str(
            &FrameMetadata::metadata_string(ifd, reader)
        )
    ).collect()
}

/// Returns the most recent system epoch timestamp call for a set of frames.
/// The system clock is called about once a second (or may be different
/// depending on the system configuration) so this is a _high jitter_
/// but _no drift_ timestamp, as long as the system clock is synchronized
/// to a master clock.
/// 
/// ## Arguments
/// 
/// * `ifds` - A slice of `IFD` objects corresponding to the
/// frames to get timestamps from
/// 
/// * `reader` - An object with read access to the file
/// 
/// ## Returns
/// 
/// A `Vec<Option<u64>>` containing the timestamps for each frame
/// in nanoseconds since the Unix epoch, in the order of the
/// requested `ifd` slice. If the system timestamp is not available
/// for a frame, the value will be `None`.
/// 
/// ## Errors
/// 
/// * `CorrosiffError::NoSystemTimestamps` - If the system timestamp
/// is not available for a frame, this will Error.
pub fn get_epoch_timestamps_system<I : IFD, ReaderT : Read + Seek>(
    ifds: &[&I],
    reader : &mut ReaderT
    ) -> Result<Vec<Option<u64>>, CorrosiffError> {
        // Error if any of these return None.
        let vec : Vec<_> = ifds.iter().map(|&ifd| 
            FrameMetadata::most_recent_system_time_from_metadata_str(
                &FrameMetadata::metadata_string(ifd, reader)
            )
        ).collect();

        if vec.iter().any(|&x| x.is_none()) {
            Err(CorrosiffError::NoSystemTimestamps)
        } else {
            Ok(vec)
        }
}

/// Returns the appended text for each frame in a set of frames.
/// This is text that is appended to the metadata string for each
/// frame, and can be used to store additional information about
/// the frame. This is often used to store information about the
/// frame that is not easily stored in the metadata tags -- timestamped
/// events, notes, etc.
/// 
/// ## Arguments
/// 
/// * `ifds` - A slice of `IFD` objects corresponding to the
/// frames to get timestamps from
/// 
/// * `reader` - An object with read access to the file
/// 
/// ## Returns
/// 
/// A `Vec<(u64, String, Option<f64>)>` containing the frame number
/// (in terms of the IFDs passed in -- NOT the actual frame number!
/// it is the responsibility of the USER to correspond these values
/// to the actual frame number), the appended text, and the timestamp
/// (if one is present) for each frame in the order of the IFDs
/// passed. The only entries in the vector are frames that
/// actually have appended text.
pub fn get_appended_text<I : IFD, ReaderT : Read + Seek>(
    ifds: &[&I],
    reader : &mut ReaderT
    ) -> Vec<(u64, String, Option<f64>)> {
    ifds.iter().enumerate().filter_map(|(idx, &ifd)| 
        FrameMetadata::appended_text_from_metadata_str(
            &FrameMetadata::metadata_string(ifd, reader)
        ).map(|(text, timestamp)| (idx as u64, text, timestamp))    
    ).collect()
}

/// Returns both system and laser epoch timestamps for each frame,
/// providing one *high jitter but low drift* and one *low jitter but
/// high drift* timestamp for each frame. The two are self-correcting,
/// and can be used to far more accurately estimate the timestamp for
/// every frame.
/// 
/// ## Arguments
/// 
/// * `ifds` - A slice of `IFD` objects corresponding to the
/// frames to get timestamps from
/// 
/// * `reader` - An object with read access to the file
/// 
/// ## Returns
/// 
/// A `Vec<(u64, u64)>` containing the timestamps for each frame
/// in nanoseconds since the Unix epoch, in the order of the
/// requested `ifd` slice. The first value in the tuple is the
/// laser timestamp, and the second is the system timestamp.
/// 
/// ## Panics
/// 
/// - If the system timestamp is not available for a frame,
/// this will PANIC! TODO: make this an error like in
/// `get_epoch_timestamps_system`.
/// 
/// ## Errors
/// 
/// * `CorrosiffError::NoSystemTimestamps` - If the system timestamp
/// is not available for a frame, this will Error. Actually doesn't!
/// TODO: make this error!
pub fn get_epoch_timestamps_both<I : IFD, ReaderT : Read + Seek>(
    ifds: &[&I],
    reader : &mut ReaderT
    ) -> Result<Vec<(u64, u64)>, CorrosiffError> {
    let vec_out : Vec<_> =
        ifds.iter().map(|&ifd| {
        let metadata = FrameMetadata::metadata_string(ifd, reader);
        (
            FrameMetadata::frame_time_epoch_from_metadata_str(&metadata),
            FrameMetadata::most_recent_system_time_from_metadata_str(&metadata).unwrap()
        )
    }).collect();
    Ok(vec_out)
}

pub mod getters{
    pub use super::{
        get_experiment_timestamps,
        get_epoch_timestamps_laser,
        get_epoch_timestamps_system,
        get_appended_text,
        get_epoch_timestamps_both,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::get_test_paths;
    use crate::tiff::FileFormat;
    use std::fs::File;
    use std::io::BufReader;

    #[test]
    fn test_metadata() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let appended_text_file = test_paths.get("APPENDED_TEXT_FILE").expect("APPENDED_TEXT_FILE not found");
        let tiff_siff = test_paths.get("TIFF_SIFF").expect("TIFF_SIFF not found");
        let f = File::open(test_file_path).unwrap();
        let mut reader = BufReader::new(f);
        let file_format = FileFormat::parse_filetype(&mut reader).unwrap();
        let ifds = file_format.get_ifd_vec(&mut reader);
        let metadata = FrameMetadata::from_ifd_and_file(&ifds[0], &mut reader).unwrap();
        println!("{:?}", metadata);

        // Appended text needs a file that actually has text in it!
        reader = BufReader::new(File::open(appended_text_file).unwrap());
        let file_format = FileFormat::parse_filetype(&mut reader).unwrap();

        let ifds = file_format.get_ifd_vec(&mut reader);

        println!(
            "Appended text: {:?}",
            FrameMetadata::appended_text_from_metadata_str(
                &FrameMetadata::metadata_string(&ifds[5726], &mut reader)
            )
        );

        let ifds_ref : Vec<_>= ifds.iter().map(|x| x).collect();
        println!(
            "all appended text: {:?}",
            get_appended_text(&ifds_ref, &mut reader)
        );

        reader = BufReader::new(File::open(tiff_siff).unwrap());

        let file_format = FileFormat::parse_filetype(&mut reader).unwrap();
        let ifds = file_format.get_ifd_vec(&mut reader);
        let metadata = FrameMetadata::from_ifd_and_file(&ifds[10], &mut reader).unwrap();
        println!("{:?}", metadata);
    }
}