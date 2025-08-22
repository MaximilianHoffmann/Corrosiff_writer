//! # FileFormat
//! 
//! Contains all the `.siff` and `.tiff` file
//! format information that is not specific to
//! parsing individual frames or data.

use std::fmt::{Display, Debug};
use std::vec;

use binrw::{
    BinRead,
    BinWrite,
    io::{
        Read,
        Seek,
        Write,
    },
};

use crate::{
    tiff::ifd::{
        BigTiffIFD,
        IFDIterator,
        IFD,
    },
    data::image::Dimensions,
};

use super::ifd::SeekRead;


/// A struct that holds the file format
/// information, which determines information
/// such as how to read individual frames,
/// what the origin of the data is, etc.
/// 
/// ## Fields
/// 
/// * `siff_header` - The header information
/// that is specific to the `.siff` and `.tiff` 
/// file format.
/// 
/// * `nvfd` - The non-varying-frame-data, which
/// is a string that contains information about
/// the image acquisition, such as hardware configuration
/// and scan parameters.
/// 
/// * `roi_string` - The data for MROI fields, if
/// present (empty for Free ScanImage but otherwise
/// contains the MROI data).
/// 
/// * `tiff_type` - The type of tiff file, which
/// is either `Tiff` or `BigTiff`.
pub struct FileFormat {
    siff_header : SiffHeader,
    pub nvfd : String,
    pub roi_string : String,
    _tiff_type : TiffType,
}

enum TiffType {
    Tiff,
    BigTiff,
}

/// Contains primary tiff specification
/// data -- nothing ScanImage or siff specific.
#[derive(BinRead, BinWrite)]
enum TiffHeader {
    #[brw(magic(42u16))]
    Default {
        first_ifd : u32,
    },
    #[brw(magic(43u16))]
    #[br(assert(_bytes_per_pointer == 8))]
    BigTiff {
        #[brw(pad_after(2))]
        _bytes_per_pointer : u16,
        first_ifd : u64,
    }
}

/// Contains the header information that dictates
/// the .tiff parameters to read the file, including:
/// * endian
/// * pointer size
/// * whether this is a ScanImage tiff
/// * the first IFD offset
/// * the length of the non-varying-frame-data
/// * the length of the ROI string
/// 
/// These data are just read directly from binary
/// into the structure with no parsing.
/// 
/// The actual NVFD is at the end of the header, and the
/// ROI string is after the NVFD.
#[derive(BinRead, BinWrite)]
#[br(
    little,
    assert(endian == [73, 73]),
    assert(tiff_magic == 117637889),
    assert((si_version == 4) || (si_version == 3)),
)]
#[bw(little)]
struct SiffHeader {
    endian : [u8; 2],
    
    tiffheader : TiffHeader,
    
    #[allow(dead_code)]
    tiff_magic : u32, // 0x4949 0x002A
    
    #[allow(dead_code)]
    si_version : u32, // 3 if 2016, 4 if 2019
    
    nvfd_length : u32,
    
    roi_string_length : u32,
}

/// Checks if the dimensions of the IFDs are all consistent
/// with each other. If they are, returns the dimensions.
/// Otherwise, returns `None`. If the IFD vector is empty,
/// it also returns `None`.
/// 
/// ## Arguments
/// 
/// * `ifds` - A vector of IFDs to check
/// 
/// ## Returns
/// 
/// * `Option<Dimensions>` - The dimensions of the IFDs if they are consistent
/// 
/// ## Example
/// 
/// ```rust, ignore
/// let reader = BufReader::new(File::open("file.siff").unwrap());
/// let file_format = FileFormat::parse_filetype(&mut reader).unwrap();
/// let ifds = file_format.get_ifd_vec(&mut reader);
/// let dims = dimensions_consistent(&ifds);
/// assert_eq!(dims, Some(Dimensions::new(512, 512)));
/// ```
pub fn dimensions_consistent<IFDType : IFD>(ifds : &Vec<IFDType>)->Option<Dimensions>{
    if ifds.len() == 0 {
        return None;
    }
    let template_dims = ifds[0].dimensions();
    if ifds.iter().any(|ifd| ifd.dimensions() != template_dims){
        None
    } else {
        template_dims
    }
}

impl FileFormat{

    /// Returns whether the file being read
    /// uses the BigTiff format vs. a standard
    /// 32 bit Tiff specification.
    pub fn is_bigtiff(&self) -> bool {
        match self._tiff_type {
            TiffType::BigTiff => true,
            TiffType::Tiff => false,
        }
    }

    /// Checks a file against all the
    /// criteria to determine the enum
    /// type of file contained.
    /// 
    /// ## Arguments
    /// 
    /// * `buffer` - A `BufReader`` pointing to
    /// the _start_ of a file. The file format
    /// data is all at the header, so a buffer
    /// is the perfect way to read it without any
    /// system calls.
    pub fn parse_filetype<'a, 'b, T>(buffer : &'a mut T) -> Result<Self, String>
        where T : Read + Seek {
        let siff_header = SiffHeader::read(buffer)
            .map_err(
                |err| format!("Error reading header: {}", err)
            )?;

        let mut nvfd = vec![0u8; siff_header.nvfd_length as usize];
        buffer.read_exact(&mut nvfd)
            .map_err(
                |err| format!("Error reading NVFD: {}", err)
            )?;

        let mut roi_string = vec![0u8; siff_header.roi_string_length as usize];
        buffer.read_exact(&mut roi_string)
            .map_err(
                |err| format!("Error reading ROI string: {}", err)
            )?;

        Ok(FileFormat {
            _tiff_type : match &siff_header.tiffheader {
                TiffHeader::Default {..} => TiffType::Tiff,
                TiffHeader::BigTiff {..} => TiffType::BigTiff,
            },
            siff_header,
            nvfd : nvfd.iter().map(|x| *x as char).collect::<String>(),
            roi_string : roi_string.iter().map(|x| *x as char).collect::<String>(),
        })
    }

    /// Just reads and stores enough to determine how to find IFDs.
    pub fn minimal_filetype<'a, 'b, T>(buffer : &'a mut T) -> Result<Self, String>
        where T : Read + Seek {
            let siff_header = SiffHeader::read(buffer)
            .map_err(
                |err| format!("Error reading header: {}", err)
            )?;
            Ok(
                FileFormat {
                    _tiff_type : match &siff_header.tiffheader {
                        TiffHeader::Default {..} => TiffType::Tiff,
                        TiffHeader::BigTiff {..} => TiffType::BigTiff,
                    },
                    siff_header,
                    nvfd : String::new(),
                    roi_string : String::new(),
                }
            ) 
        }

    /// Returns the location of the first IFD
    /// in the file (from start).
    /// 
    /// ## Returns
    /// 
    /// * `u64` or `u32` - The location of the first IFD from the beginning of the file
    pub fn first_ifd_val(&self) -> u64 {
        match self.siff_header.tiffheader {
            TiffHeader::Default {first_ifd} => first_ifd as u64,
            TiffHeader::BigTiff {first_ifd, ..} => first_ifd,
        }
    }

    /// Returns an `IFD` object from the pointer to its start. Mostly
    /// for debugging.
    #[allow(dead_code)]
    pub fn read_ifd<ReaderT : SeekRead>(buffer : &mut ReaderT, offset : u64)
        -> Result<BigTiffIFD, std::io::Error> {
        buffer.seek(std::io::SeekFrom::Start(offset))?;
        BigTiffIFD::read(buffer).map_err(
            |err| std::io::Error::new(std::io::ErrorKind::InvalidData, err)
        )
    }

    /// Returns an iterator over all IFDs in the file.
    pub fn get_ifd_iter<'reader, ReaderT>(&self,buff: &'reader mut ReaderT) 
    -> IFDIterator<'reader, ReaderT, BigTiffIFD>
    where ReaderT : SeekRead + Sized, {
        IFDIterator {
            reader : buff,
            to_next : self.first_ifd_val(),
        }
    }

    #[allow(dead_code)]
    pub fn get_ifd_vec<'reader, ReaderT>(&self, buff : &'reader mut ReaderT) -> Vec<BigTiffIFD>
    where ReaderT : SeekRead + Sized{
        self.get_ifd_iter(buff).collect()
    }

    /// Parses the non-varying-frame-data to get
    /// the number of FLIM tau bins -- if that data
    /// is present in the header.
    pub fn num_flim_tau_bins(&self) -> Option<u32> {
        let needle = "Tau_bins = ";
        let tau_bin_ptr = self.nvfd.find(needle)?;
        let tau_bin_ptr = tau_bin_ptr + needle.len();
        let tau_bin_len = self.nvfd[tau_bin_ptr..].find("\n")?;
        let tau_bin = self.nvfd[tau_bin_ptr..tau_bin_ptr+tau_bin_len].trim();
        //tau_bin.parse::<u32>().ok()
        tau_bin.parse::<u32>().map(|x| x + 2).ok()
    }

    /// Returns the size of a single bin in picoseconds
    #[allow(dead_code)]
    pub fn flim_tau_bin_size_picoseconds(&self) -> Option<u32> {
        let needle = "binResolution = ";
        let tau_bin_ptr = self.nvfd.find(needle)?;
        let tau_bin_ptr = tau_bin_ptr + needle.len();
        let tau_bin_len = self.nvfd[tau_bin_ptr..].find("\n")?;
        let tau_bin = self.nvfd[tau_bin_ptr..tau_bin_ptr+tau_bin_len].trim();
        let bin_res = tau_bin.parse::<u32>().ok()?;
        Some(5 * u32::pow(2,bin_res))
    }

    /// Writes the file-format data that is common to all tiff/siff files
    /// into the position the writer is currently at.
    /// 
    /// ## Arguments
    /// 
    /// * `writer` - A writer that can write and seek pointing to the
    /// location in the file where the tiff header data should be written
    /// 
    /// ## Returns
    /// 
    /// * `Result<(), binrw::io::Error>` - An error if the write fails
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// 
    /// let writer = BufWriter::new(File::create("file.tiff").unwrap());
    /// file_format.write(&writer);
    /// 
    /// ```
    /// 
    pub fn write<T: Write + Seek>(&self, writer : &mut T) -> binrw::io::Result<()> {
        self.siff_header.write(writer)
        .map_err(|e| binrw::io::Error::new(binrw::io::ErrorKind::Other, e))?;
        writer.write_all(&mut self.nvfd.as_bytes())?;
        writer.write_all(&mut self.roi_string.as_bytes())?;
        debug_assert_eq!(
            writer.stream_position()?,
            self.first_ifd_val()
        );
        Ok(())
    }

}

impl Display for FileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "FileFormat: {:?}",
            self.siff_header,
        )
    }
}

impl Debug for FileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "FileFormat: {:?}\nNVFD: {}\n ROI String: {}",
            self.siff_header,
            self.nvfd,
            self.roi_string,
    )
    }
}

impl Debug for SiffHeader {
    fn fmt(&self, f : &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "Endian: {:?}\nNVFD Length: {}\nROI String Length: {}",
            self.endian.iter().map(|x| *x as char).collect::<Vec<char>>(),
            self.nvfd_length,
            self.roi_string_length,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use binrw::io::BufReader;
    use crate::tests::get_test_paths;

    #[test]
    fn test_parse_filetype() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let f = File::open(test_file_path)
            .expect("Could not open test file");
        let mut buffer = BufReader::new(&f);
        let file_format = FileFormat::parse_filetype(&mut buffer).unwrap();
        println!("{:?}", file_format);
        assert_eq!(file_format.num_flim_tau_bins(), Some(631));
    }

    #[test]
    fn test_read_ifd() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let f = File::open(test_file_path)
            .expect("Could not open test file");
        let mut buffer = BufReader::new(&f);
        let file_format = FileFormat::parse_filetype(&mut buffer).unwrap();
        println!("First IFD is {}" , file_format.first_ifd_val());
        let ifd = FileFormat::read_ifd(&mut buffer, file_format.first_ifd_val()).unwrap();
        println!("{:?}", ifd);
    }

    #[test]
    fn test_get_ifd_iter() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let f = File::open(test_file_path)
            .expect("Could not open test file");
        let mut buffer = BufReader::new(&f);
        let file_format = FileFormat::parse_filetype(&mut buffer).unwrap();
        let mut ifd_iter = file_format.get_ifd_iter(&mut buffer);
        while let Some(ifd) = ifd_iter.next() {
            println!("{:?}", ifd);
        }
    }
}