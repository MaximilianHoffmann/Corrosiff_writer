//! The primary `SiffReader` object, which
//! parses files and extracts interesting
//! information and/or data.
//! 
//! The style here is quite bad -- lots of copy-paste code
//! because I kept realizing that I would need a new feature
//! and just sort of tacked it on in the same style. I look
//! forward to thinking about how to make this more natural,
//! elegant, and (frankly) readable.
use std::{
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::Write,
    io::Seek,
    path::{Path, PathBuf}
};

use binrw::io::BufReader;
use itertools::izip;
use ndarray::prelude::*;
use rayon::prelude::*;
use num_complex::Complex;

// my module structure is probably
// too complex. Or I should hide some of
// these deeper in the module?
use crate::{
    data::image::{
        load::*, Dimensions, DimensionsError
    }, metadata::{getters::*, FrameMetadata}, tiff::{
        dimensions_consistent, BigTiffIFD, FileFormat, IFD
    }, utils::{parallelize_op,FramesError,}, ClockBase, CorrosiffError, TiffMode
};

pub type RegistrationDict = HashMap<u64, (i32, i32)>;

// Boilerplate frame checking code.

/// Iterates through all frames and check that there's a corresponding
/// IFD for each frame.
fn _check_frames_in_bounds(frames : &[u64], ifds : &Vec<BigTiffIFD>)
    -> Result<(), DimensionsError> {
    frames.iter().all(|&x| x < ifds.len() as u64)
        .then(||()).ok_or(DimensionsError::IncorrectFrames)
}

/// Checks whether all requested frames share a shape. If not
/// returns `None`, otherwise returns the shared shape.
fn _check_shared_shape(frames : &[u64], ifds : &Vec<BigTiffIFD>)
    -> Option<Dimensions> {
    let array_dims = ifds[frames[0] as usize].dimensions().unwrap();
    frames.iter().all(
        |&x| {
            ifds[x as usize].dimensions().unwrap() == array_dims
        })
    .then(||array_dims)
}

/// Returns `Ok` if every element of `frames` is a key of `registration`.
/// If `registration` is totally empty, converts it to `None` and returns `Ok`.
/// If it's only partially populated, returns an error.
fn _check_registration(registration : &mut Option<&RegistrationDict>, frames : &[u64])
    -> Result<(), FramesError> {
    if let Some(reg) = registration {
        if reg.is_empty() {
            *registration = None;
            Ok(())
        } else {
            frames.iter().all(|k| reg.contains_key(k))
            .then(||()).ok_or(FramesError::RegistrationFramesMissing)
        }
    } else {
        Ok(())
    }
}


/// A struct for reading a `.siff` file
/// or a ScanImage-Flim `.tiff` file.
/// Has methods which return arrays of
/// image or FLIM
pub struct SiffReader {
    _file : File,
    _filename : PathBuf,
    file_format : FileFormat,
    _ifds : Vec<BigTiffIFD>,
    _image_dims : Option<Dimensions>,
}

impl SiffReader{
    
    /// Opens a file and returns a `SiffReader` object
    /// for interacting with the data if successful.
    /// 
    /// ## Arguments
    /// 
    /// * `filename` - A string slice that holds the name of the file to open
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `std::io::Error` - If there is an error opening the file,
    /// it will be returned directly.
    /// 
    pub fn open<P : AsRef<Path>>(filename : P) -> Result<Self, CorrosiffError> {

        // Open the file and parse its formatting info
        let file = File::open(&filename)?;
        let mut buff = BufReader::new(&file);
        let file_format = {
            FileFormat::parse_filetype(&mut buff)
            .map_err(|_| CorrosiffError::FileFormatError)
        }?;

        // A small buffer for reading the IFDs which are quite small, using a smaller
        // buffer makes this run much faster
        let mut wee_buff = BufReader::with_capacity(400, &file);
        let _ifds = file_format.get_ifd_iter(&mut wee_buff).collect::<Vec<_>>();
        Ok(
            SiffReader {
            _filename : filename.as_ref().to_path_buf(),
            _image_dims : dimensions_consistent(&_ifds),
            _ifds,
            file_format,
            _file : file,
            }
        )
    }

    /// Returns number of frames in the file
    /// (including flyback etc).
    pub fn num_frames(&self) -> usize {
        self._ifds.len()
    }

    pub fn frames_vec(&self) -> Vec<u64> {
        (0..self.num_frames() as u64).collect()
    }

    pub fn image_dims(&self) -> Option<Dimensions> {
        self._image_dims.clone()
    }

    /// Get value of the filename
    /// 
    /// # Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// println!("{}", reader.filename());
    /// ```
    pub fn filename(&self) -> String {
        self._filename.to_str().unwrap().to_string()
    }

    /// Return the non-varying frame data, which
    /// contains all of the instrument settings
    pub fn nvfd(&self) -> String {
        self.file_format.nvfd.clone()
    }

    /// Return the mROI information as a string
    pub fn roi_string(&self) -> String {
        self.file_format.roi_string.clone()
    }

    /// Returns whether the file uses the BigTIFF
    /// format or the standard 32-bit TIFF format.
    pub fn is_bigtiff(&self) -> bool {
        self.file_format.is_bigtiff()
    }

    /// Returns whether the file being read is a
    /// .siff (and contains lifetime information)
    /// or just a regular tiff file.
    /// 
    /// For now, this is implemented is a very
    /// unsophisticated manner -- it simply
    /// checks whether the file ends in `.siff`!
    /// 
    /// TODO: Better!
    pub fn is_siff(&self) -> bool {
        self._filename.to_str().unwrap().ends_with(".siff")
    }

    /// Size of the FLIM arrival time histogram
    /// in bins.
    /// 
    /// ## Returns
    /// 
    /// Number of actual bins (irrespective of what the bin
    /// size corresponds to in real time units)
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the histogram size
    /// is unknown or was unable to be parsed (or if the file is not a `.siff`)
    pub fn num_flim_bins(&self) -> Result<u32, CorrosiffError> {
        self.file_format.num_flim_tau_bins()
        .ok_or(CorrosiffError::DimensionsError(
            DimensionsError::UnknownHistogramSize
        ))
    }

    /// Return the metadata objects corresponding to
    /// each of the requested frames.
    pub fn get_frame_metadata(&self, frames : &[u64]) 
        -> Result<Vec<FrameMetadata>, CorrosiffError> {

            _check_frames_in_bounds(frames, &self._ifds).map_err(
            |err| FramesError::DimensionsError(err)
        )?;

        let mut metadata = Vec::with_capacity(frames.len());
        let mut f = File::open(&self._filename).unwrap();
        for frame in frames {
            metadata.push(FrameMetadata::from_ifd_and_file(
                &self._ifds[*frame as usize],
                &mut f
            )?);
        }
        Ok(metadata)
    }

    /// Return an array of timestamps corresponding to the
    /// experiment time of each frame requested (seconds since
    /// the onset of the acquisition).
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array1<f64>, CorrosiffError>` - An array of `f64` values
    /// corresponding to the timestamps of the frames requested (in units
    /// of seconds since beginning of the acquisition).
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let timestamps = reader.get_experiment_timestamps(&[0, 1, 2]);
    /// ```
    pub fn get_experiment_timestamps(
        &self, frames : &[u64]
    ) -> Result<Array1<f64>, CorrosiffError> {

        _check_frames_in_bounds(frames, &self._ifds)?;
        let mut array = Array1::<f64>::zeros(frames.len());

        parallelize_op!(
            array, 
            5000, 
            frames, 
            self._filename.to_str().unwrap(),
            | filename : &str |{BufReader::with_capacity(800, File::open(filename).unwrap())},
            |frames : &[u64], chunk : &mut ArrayBase<_, Ix1>, reader : &mut BufReader<File>| {
                let ifds = frames.iter().map(|&x| &self._ifds[x as usize]).collect::<Vec<_>>();
                get_experiment_timestamps(&ifds, reader)
                    .iter().zip(chunk.iter_mut())
                    .for_each(|(&x, y)| *y = x);
                Ok(())
            }
        );
        Ok(array)    
    }

    /// Return an array of timestamps corresponding to the
    /// epoch time of each frame requested computed using
    /// the number of laser pulses into the acquisition at the
    /// time of the frame trigger (and the estimated pulse rate).
    /// 
    /// This measurement has extremely low jitter but a fixed
    /// rate of drift from "true" epoch (as the system clock might
    /// read it).
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array1<u64>, CorrosiffError>` - An array of `u64` values
    /// corresponding to the epoch time of each frame
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let timestamps = reader.get_epoch_timestamps(&[0, 1, 2]);
    /// ```
    pub fn get_epoch_timestamps_laser(
        &self, frames : &[u64]
    ) -> Result<Array1<u64>, CorrosiffError> {
        _check_frames_in_bounds(frames, &self._ifds)?;

        let mut array = Array1::<u64>::zeros(frames.len());

        parallelize_op!(
            array, 
            5000, 
            frames, 
            self._filename.to_str().unwrap(),
            | filename : &str |{BufReader::with_capacity(800, File::open(filename).unwrap())},
            |frames : &[u64], chunk : &mut ArrayBase<_, Ix1>, reader : &mut BufReader<File>| {
                let ifds = frames.iter().map(|&x| &self._ifds[x as usize]).collect::<Vec<_>>();
                get_epoch_timestamps_laser(&ifds, reader)
                    .iter().zip(chunk.iter_mut())
                    .for_each(|(&x, y)| *y = x);
                Ok(())
            }
        );
        Ok(array)
    }

    /// Return an array of timestamps corresponding to the
    /// most recent epoch timestamped system call at the time
    /// of the frame trigger. This is the most accurate measure
    /// of system time because it does not drift, but the system
    /// is only queried about once a second, so there is high
    /// _apparent jitter_, with many consecutive frames sharing
    /// a value.
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array1<u64>, CorrosiffError>` - An array of `u64` values
    /// corresponding to the system time of each frame
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// reader.get_epoch_timestamps_system(&[0, 1, 2]);
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::NoSystemTimestamps` - If the system timestamps
    /// are not present in the file, this error is returned.
    pub fn get_epoch_timestamps_system(
        &self, frames : &[u64]
    ) -> Result<Array1<u64>, CorrosiffError> {
        _check_frames_in_bounds(frames, &self._ifds)?;

        let mut array = Array1::<u64>::zeros(frames.len());

        let op = 
        | frames : &[u64], chunk : &mut ArrayViewMut1<u64>, reader : &mut BufReader<File> |
        -> Result<(), CorrosiffError> {
            let ifds = frames.iter().map(|&x| &self._ifds[x as usize]).collect::<Vec<_>>();
            get_epoch_timestamps_system(&ifds, reader)?
                .iter().zip(chunk.iter_mut())
                .for_each(|(&x, y)| *y = x.unwrap());
            Ok(())
        };

        parallelize_op!(
            array, 
            5000, 
            frames, 
            self._filename.to_str().unwrap(),
            | filename : &str | { BufReader::with_capacity(800, File::open(filename).unwrap()) },
            op
        );

        Ok(array)
    }

    /// Return an array of timestamps corresponding to the
    /// two measurements of epoch time in the data: the
    /// laser clock synched one (*low jitter, some drift*)
    /// and the system call one (*high jitter, no drift*).
    /// 
    /// The two can be combined to allow much more reliable
    /// estimation of the timestamp of every frame trigger
    /// in absolute epoch time determined by the PTP system.
    /// These data are in nanoseconds since epoch.
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array2<u64>, CorrosiffError>` - A 2D array of `u64` values
    /// corresponding to the two epoch timestamps of each frame. The first
    /// row is `laser_clock` values, the second row is `system_clock` values.
    /// The `system_clock` changes only once a second or so, but this is much
    /// faster than the drift of the laser clock.
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// reader.get_epoch_timestamps_both(&[0, 1, 2]);
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::NoSystemTimestamps` - If the system timestamps
    /// are not present in the file, this error is returned.
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds (the underlying `DimensionsError` is attached to this error)
    pub fn get_epoch_timestamps_both(
        &self, frames : &[u64]
    ) -> Result<Array2<u64>, CorrosiffError> {
        _check_frames_in_bounds(frames, &self._ifds)?;

        let mut array = Array2::<u64>::zeros((2, frames.len()));

        let chunk_size = 5000;
        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = array.axis_chunks_iter_mut(Axis(1), chunk_size).collect();

        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, mut chunk)| -> Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = BufReader::with_capacity(800, 
                File::open(&self._filename).unwrap()
            );

            let ifds = local_frames.iter().map(|&x| &self._ifds[x as usize]).collect::<Vec<_>>();
            get_epoch_timestamps_both(&ifds, &mut local_f)?
                .iter().zip(chunk.axis_iter_mut(Axis(1)))
                .for_each(|(x, mut y)|{
                y[0] = x.0; y[1] = x.1}
            );
            Ok(())
            }
        )?;
        Ok(array)
    }

    /// Returns a vector of all frames containing appended text.
    /// Each element of the vector is a tuple containing the frame
    /// number, the text itself, and the timestamp of the frame (if present).
    /// 
    /// ## Returns
    /// 
    /// (`frame_number`, `text`, Option<`timestamp`>)
    pub fn get_appended_text(&self, frames : &[u64]) -> Vec<(u64, String, Option<f64>)> {
        let mut f = File::open(&self._filename).unwrap();
        let ifd_by_ref = frames.iter().map(|&x| &self._ifds[x as usize]).collect::<Vec<_>>();
        get_appended_text(&ifd_by_ref, &mut f)
        .iter().map(
            |(idx, this_str, this_timestamp)|
            (frames[*idx as usize], this_str.clone(), *this_timestamp)
        ).collect()
    }

    /******************************
     * 
     * Frame data methods
     * 
     * ***************************
     */

    /// Return an array corresponding to the intensity
    /// data of the frames requested. The returned array
    /// is a 3D array, with the first dimension corresponding
    /// to the frame number, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frame.
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let frames = reader.get_frames_intensity(
    ///     &[0, 1, 2],
    ///     None
    /// );
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `FramesError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds or do not all share the same shape (the underlying
    /// `DimensionsError` is attached to this error)
    /// 
    /// * `FramesError::IOError(_)` - If there is an error reading the file (with
    /// the underlying error attached)
    /// 
    /// * `FramesError::RegistrationFramesMissing` - If registration is used, and
    /// the registration values are missing for some frames
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array3<u16>, FramesError>` - A 3D array of `u16` values
    /// corresponding to the intensity data of the frames requested.
    pub fn get_frames_intensity(
        &self,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<Array3<u16>, CorrosiffError> { 
        
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds).map_err(
                FramesError::DimensionsError)?;
        
        // Check that the frames share a shape
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        let mut registration = registration;
        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        // Create the array
        let mut array = Array3::<u16>::zeros((frames.len(), array_dims.ydim as usize, array_dims.xdim as usize));

        let op = | frames : &[u64], chunk : &mut ArrayViewMut3<u16>, reader : &mut File |
        -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            load_array_intensity_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                *reg.get(&this_frame).unwrap(),
                            )
                        })?;
                },
                None => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            load_array_intensity(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                            )
                        })?;
                },
            }
            Ok(())
        };
        
        parallelize_op!(
            array, 
            2500, 
            frames, 
            self._filename,
            op
        );
        
        Ok(array)
    }

    /// Return two arrays: the intensity (photon counts) and
    /// the empirical lifetime (in arrival time bins) of each
    /// pixel of the frames requested.
    /// corresponding to the `y` and `x` dimensions of the frame.
    /// The lifetime array is a 3D `f64` array with the first dimension
    /// corresponding to the frame number, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frame.
    /// The intensity array is a 3D `u16` array with the same shape.
    /// If `registration` is `None`, the frames are read unregistered,
    /// otherwise they are registered in place.
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array3<f64>, Array3<u16>), CorrosiffError>` - A tuple
    /// containing the lifetime and intensity arrays of the frames requested
    /// (in that order). The lifetime array is the empirical lifetime in
    /// units of arrival time bins of the MultiHarp -- meaning for every frame,
    /// each pixel contains the average arrival time of all photons in that pixel
    /// during that frame.
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let (lifetime, intensity) = reader.get_frames_flim(
    ///    &[0, 1, 2],
    ///   None
    /// );
    /// let intensity_alone = reader.get_frames(
    ///   &[0, 1, 2],
    ///  None
    /// );
    /// assert_eq!(intensity, intensity_alone);
    /// // intensity is a 3D array of u16 values
    /// // lifetime is a 3D array of f64 values in units of
    /// // arrival time bins of the MultiHarp.
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds or do not all share the same shape (the underlying
    /// `DimensionsError` is attached to this error)
    /// 
    /// * `CorrosiffError::FramesError(FramesError::IOError(_))` - If there is an error reading the file (with
    /// the underlying error attached)
    /// 
    /// * `CorrosiffError::FramesError(FramesError::RegistrationFramesMissing)` - If registration is used, and
    /// the registration values are missing for some frames
    /// 
    /// ## See also
    /// 
    /// - `get_frames_intensity` - for just the intensity data
    /// - `get_histogram` to pool all photons for a frame into a histogram
    /// 
    /// ## Panics
    /// 
    /// ???
    pub fn get_frames_flim(
        &self,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<(Array3<f64>, Array3<u16>), CorrosiffError> {
        
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let (mut lifetime, mut intensity) = (
            Array3::<f64>::zeros((frames.len(), array_dims.ydim as usize, array_dims.xdim as usize)),
            Array3::<u16>::zeros((frames.len(), array_dims.ydim as usize, array_dims.xdim as usize))
        ); 

        let op = 
        |
            frames : &[u64],
            chunk_intensity : &mut ArrayViewMut3<u16>,
            chunk_lifetime : &mut ArrayViewMut3<f64>,
            reader : &mut File
        | -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    izip!(
                        frames,
                        chunk_lifetime.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0))
                    ).try_for_each(
                        
                        |(&this_frame, mut this_chunk_l, mut this_chunk_i)|
                        -> Result<(), CorrosiffError> {
                        load_flim_empirical_and_intensity_arrays_registered(
                            reader,
                            &self._ifds[this_frame as usize],
                            &mut this_chunk_l,
                            &mut this_chunk_i,
                            *reg.get(&this_frame).unwrap(),
                        )
                        }

                    )?;
                },
                None => {
                    izip!(
                        frames,
                        chunk_lifetime.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0))
                    ).try_for_each(

                        |(&this_frame, mut this_chunk_l, mut this_chunk_i)|
                        -> Result<(), CorrosiffError> {
                        load_flim_empirical_and_intensity_arrays(
                            reader,
                            &self._ifds[this_frame as usize],
                            &mut this_chunk_l,
                            &mut this_chunk_i,
                        )
                        }

                    )?;
                },
            }
            Ok(())
        };
        
        parallelize_op!(
            (intensity, lifetime), 
            2500, 
            frames, 
            self._filename,
            op
        );
        Ok((lifetime, intensity))
    }

    /// Return two arrays: the intensity (photon counts) and
    /// the phasor representation of the pixelwise photon histogram.
    /// 
    /// TODO: Implement!
    /// 
    pub fn get_frames_phasor(
        &self,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<(Array3<Complex<f64>>, Array3<u16>), CorrosiffError> {
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let (mut phasor, mut intensity) = (
            Array3::<Complex<f64>>::zeros((frames.len(), array_dims.ydim as usize, array_dims.xdim as usize)),
            Array3::<u16>::zeros((frames.len(), array_dims.ydim as usize, array_dims.xdim as usize))
        ); 

        let num_tau = self.file_format.num_flim_tau_bins().unwrap();
        let cos_lookup = Array1::from_iter(
            (0..num_tau)
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).cos())
        );

        let sin_lookup = Array1::from_iter(
            (0..self.file_format.num_flim_tau_bins().unwrap())
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).sin())
        );

        let op = 
        |
            frames : &[u64],
            chunk_intensity : &mut ArrayViewMut3<u16>,
            chunk_phasor : &mut ArrayViewMut3<Complex<f64>>,
            reader : &mut File
        | -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    izip!(
                        frames,
                        chunk_phasor.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0))
                    ).try_for_each(
                        
                        |(&this_frame, mut this_chunk_p, mut this_chunk_i)|
                        -> Result<(), CorrosiffError> {
                        load_flim_phasor_and_intensity_arrays_registered(
                            reader,
                            &self._ifds[this_frame as usize],
                            &mut this_chunk_p,
                            &mut this_chunk_i,
                            &cos_lookup.view(),
                            &sin_lookup.view(),
                            *reg.get(&this_frame).unwrap(),
                        )
                        }

                    )?;
                },
                None => {
                    izip!(
                        frames,
                        chunk_phasor.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0))
                    ).try_for_each(

                        |(&this_frame, mut this_chunk_p, mut this_chunk_i)|
                        -> Result<(), CorrosiffError> {
                        load_flim_phasor_and_intensity_arrays(
                            reader,
                            &self._ifds[this_frame as usize],
                            &mut this_chunk_p,
                            &mut this_chunk_i,
                            &cos_lookup.view(),
                            &sin_lookup.view(),
                        )
                        }

                    )?;
                },
            }
            Ok(())
        };
        
        parallelize_op!(
            (intensity, phasor), 
            2500, 
            frames, 
            self._filename,
            op
        );
        Ok((phasor, intensity))
    }

    /// Returns a 1D array of `u64` values corresponding to the
    /// photon stream of the frames requested, rather than converting
    /// the data into an array of intensity (or arrival) values.
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// 
    /// ## Panics
    /// 
    /// *It's not implemented yet!*
    // pub fn get_photon_stream(
    //     &self,
    //     frames : &[u64],
    //     registration : Option<&RegistrationDict>,
    // ) -> Result<Array1<u64>, CorrosiffError> {
    //     unimplemented!()
    // }

    /// Returns a 4d array of `u16` values corresponding to the
    /// number of photons per pixel per arrival time bin for each
    /// frame. The first dimension corresponds to the frame number,
    /// the second and third dimensions correspond to the `y` and `x`
    /// dimensions of the frame, and the fourth dimension corresponds
    /// to the arrival time bin. Warning: this data structure will generally
    /// be ~600x as large as the corresponding image structure!
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// 
    pub fn get_frames_tau_d(
        &self,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<Array4<u16>, CorrosiffError> {

        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut array = Array4::<u16>::zeros(
            (
                frames.len(),
                array_dims.ydim as usize,
                array_dims.xdim as usize,
                self.file_format.num_flim_tau_bins().unwrap() as usize
            )
        );

        let op = | frames: &[u64], chunk : &mut ArrayViewMut4::<u16>, reader : &mut File | -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            load_array_tau_d_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                *reg.get(&this_frame).unwrap(),
                            )
                        })?;
                },
                None => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            load_array_tau_d(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                            )
                        })?;
                },
            }
            Ok(())
        };

        parallelize_op!(
            array, 
            2500, 
            frames, 
            self._filename,
            op
        );

        Ok(array)
    }

    /***************
     * 
     * ROI-like methods
     * 
     * ************
     */

    /// Returns a 2D array of `u64` values corresponding to the
    /// number of photons in each bin of the arrival time histogram
    /// summing across ALL photons in the frames requested (not masked!).
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// ## Returns
    /// 
    /// * `histogram` - An `ndarray::Array2<u64>` with the first dimension
    /// equal to the number of frames and the second dimension equal to the
    /// number of arrival time bins of the histogram (read from the `.siff`
    /// metadata).
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let hist = reader.get_histogram(&[0, 1, 2]);
    /// 
    /// assert_eq!(reader.metadata().picoseconds_per_bin(), 20)
    /// 
    /// // 629 bins * 20 picoseconds per bin = 12.58 nanoseconds, ~80 MHz
    /// assert_eq!(hist.shape(), &[3, 629]);
    /// ```
    /// 
    pub fn get_histogram(&self, frames : &[u64]) -> Result<Array2<u64>, CorrosiffError> {

        _check_frames_in_bounds(frames, &self._ifds).map_err(|err| FramesError::DimensionsError(err))?;

        let mut array = Array2::<u64>::zeros(
            (
                frames.len(),
                self.file_format.num_flim_tau_bins()
                .ok_or(FramesError::FormatError("Could not compute tau bins for file".to_string()))? as usize
            )
        );

        let op = |frames : &[u64], chunk : &mut ArrayViewMut2<u64>, reader : &mut File|
        -> Result<(), CorrosiffError> {
            frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                .try_for_each(
                    |(&this_frame, mut this_chunk)|
                    -> Result<(), std::io::Error> {
                    load_histogram(
                        &self._ifds[this_frame as usize],
                        reader,
                        &mut this_chunk,
                    )
                })?;
            Ok(())
        };

        parallelize_op!(
            array, 
            5000, 
            frames, 
            self._filename,
            op
        );

        Ok(array)
    }

    /// Extracts a framewise photon arrival time histogram
    /// restricted only to the pixels in the region of interest.
    /// 
    /// Returns an array of `u64` values corresponding to the
    /// number of photons in each bin of the arrival time histogram
    /// summing across all photons within the ROI for each frame.
    /// 
    /// ## Arguments
    /// 
    /// * `frames` - The frames to apply the mask to
    /// 
    /// * `mask` - A 2D boolean array with the same shape as the frames'
    /// `y` and `x` dimensions. The ROI is true for the pixels that should
    /// be pooled to produce the histogram
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array2<u64>, CorrosiffError>` - An array of `u64` values
    /// corresponding to the number of photons in each bin of the arrival
    /// time histogram for each frame requested. Shape is `(frames.len(), num_bins)`.
    pub fn get_histogram_mask(&self, frames : &[u64], mask : &ArrayView2<bool>, registration : Option<&RegistrationDict>)
    -> Result<Array2<u64>, CorrosiffError> {
        _check_frames_in_bounds(&frames, &self._ifds)?;
            
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != mask.dim() {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple(mask.dim()),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut array = Array2::<u64>::zeros((frames.len(), self.file_format.num_flim_tau_bins().unwrap() as usize));

        let op = 
        | frames : &[u64], chunk : &mut ArrayViewMut2<u64>, reader : &mut File | 
        -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            load_histogram_mask_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                &mask.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                        })?;
                },
                None => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            load_histogram_mask(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                &mask.view(),
                            )?; Ok(())
                        })?;
                }
            }

            Ok(())
        };

        parallelize_op!(
            array, 
            5000, 
            frames, 
            self._filename,
            op
        );

        Ok(array)
    }

    /// Extracts a framewise photon arrival time histogram
    /// restricted only to the pixels in the region of interest.
    /// Uses a volume mask which cycles through the z-dimension
    /// alongside the frames
    /// 
    pub fn get_histogram_mask_volume(
        &self, frames : &[u64], mask : &ArrayView3<bool>, registration : Option<&RegistrationDict>
    ) -> Result<Array2<u64>, CorrosiffError> {
        _check_frames_in_bounds(&frames, &self._ifds)?;

        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (mask.dim().1, mask.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((mask.dim().1, mask.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut array = Array2::<u64>::zeros((frames.len(), self.file_format.num_flim_tau_bins().unwrap() as usize));

        // Another disappointing lack of macro magic
        let chunk_size = 2500;

        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;

        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = array.axis_chunks_iter_mut(Axis(0), chunk_size).collect();

        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, mut chunk)| -> Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(&self._filename).unwrap();

            let roi_cycle = mask.axis_iter(Axis(0)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the mask
            let roi_cycle = roi_cycle.skip(start % mask.dim().0);

            match registration {
                Some(reg) => {
                    izip!(local_frames.iter(), chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_chunk, roi_plane)|
                            -> Result<(), CorrosiffError> {
                            load_histogram_mask_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                &roi_plane,
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                        })?;
                },
                None => {
                    izip!(local_frames.iter(), chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_chunk, roi_cycle)|
                            -> Result<(), CorrosiffError> {
                            load_histogram_mask(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                &roi_cycle,
                            )?; Ok(())
                        })?;
                }
            }
            Ok(())
            }
        )?;
        Ok(array)
    }

    /// Returns a timeseries of a 1D-ified mask applied to
    /// each frame in the specified range. This allows you
    /// to read only a much smaller array into memory than the
    /// whole image series but still retain each of the individual
    /// pixels. Should be equivalent to `get_frames_intensity`
    /// followed by masking.
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 2D boolean array with the same shape as the
    /// frames' `y` and `x` dimensions. The ROI is a mask which
    /// will be used to select the pixels of the returned array.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array2<u64>, CorrosiffError>` - A 2D array of `u64` values
    /// corresponding to the flattened intensity of the frames requested
    /// within the ROI specified. Shape is `(frames.len(), num_pixels_in_roi)`.
    /// 
    /// ## Example
    /// 
    /// ```rust
    /// use corrosiff::tests::get_test_files;
    /// use corrosiff::SiffReader;
    /// use ndarray::prelude::*;
    /// let test_files = get_test_files();
    /// let filename = test_files.get("TEST_PATH").expect("TEST_PATH not found");
    /// let reader = SiffReader::open(filename);
    /// let roi = Array2::<bool>::from_elem((512, 512), true);
    /// // Set the ROI to false in the middle
    /// //roi.slice(s![200..300, 200..300]).fill(false);
    /// 
    ///
    /// ```
    ///
    /// ## See also
    /// 
    /// - `get_roi_volume` - for a 3D ROI mask
    pub fn get_roi_flat(
        &self,
        roi : &ArrayView2<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<Array2<u64>, CorrosiffError> {
        
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != roi.dim() {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple(roi.dim()),
                }
            ).into());
        }

        let mut registration = registration;
        _check_registration(&mut registration, &frames)?;

        // The ROI array has the same shape as the number of True values
        // in the mask.
        let mut array = Array2::<u64>::zeros(
            (frames.len(),
            roi.iter().filter(|&&x| x).count())
        );

        let mut lookup_table = Array2::<usize>::zeros(roi.dim());

        let mut target_px = 0;

        for (&roi_px, lookup_idx) in izip!(roi.iter(), lookup_table.iter_mut()) {
            if roi_px {
                *lookup_idx = target_px;
                target_px += 1;
            }
        }

        let op = | frames : &[u64], chunk : &mut ArrayViewMut2<u64>, reader : &mut File |
        -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            extract_intensity_mask_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                &roi.view(),
                                &lookup_table.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                        })?;
                },
                None => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                        .try_for_each(
                            |(&this_frame, mut this_chunk)|
                            -> Result<(), CorrosiffError> {
                            extract_intensity_mask(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_chunk,
                                &roi.view(),
                                &lookup_table.view(),
                            )?; Ok(())
                        })?;
                },
            }
            Ok(())
        };

        parallelize_op!(
            array, 
            2500, 
            frames, 
            self._filename,
            op
        );

        Ok(array)
    }

    /// Returns a timeseries of a 1D-ified mask applied to
    /// the volumes in the specified range. This allows you
    /// to read only a much smaller array into memory than the
    /// whole image series but still retain each of the individual
    /// pixels. Should be equivalent to `get_frames_intensity`
    /// followed by masking.
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 3D boolean array with the same shape as the
    /// frames' `y` and `x` dimensions. The ROI is a mask which
    /// will be used to sum the intensity of the frames requested.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve. Iterates through these frames
    /// in the z-dimension of the ROI.
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array2<u64>, CorrosiffError>` - A 2D array of `u64` values
    /// corresponding to the flattened intensity of the frames requested
    /// within the ROI specified. Shape is `(frames.len() / num_slices, num_pixels_in_roi)`.
    /// 
    /// ## Example
    /// 
    /// ```rust
    /// use corrosiff::tests::get_test_files;
    /// use corrosiff::SiffReader;
    /// use ndarray::prelude::*;
    /// let test_files = get_test_files();
    /// let filename = test_files.get("TEST_PATH").expect("TEST_PATH not found");
    /// let reader = SiffReader::open(filename);
    /// ///TODO FINISH THIS EXAMPLE
    /// 
    ///
    /// ```
    ///
    /// ## See also
    /// 
    /// - `get_roi_flat` - for a 2D ROI mask
    pub fn get_roi_volume(
        &self,
        roi : &ArrayView3<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<Array2<u64>, CorrosiffError> {
        _check_frames_in_bounds(&frames, &self._ifds)?;

        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (roi.dim().1, roi.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((roi.dim().1, roi.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;
        _check_registration(&mut registration, &frames)?;


        let num_slices = roi.dim().0;

        // The ROI array has the same shape as the number of True values
        // in the mask.
        let mut array = Array2::<u64>::zeros(
            (frames.len() / num_slices, roi.iter().filter(|&&x| x).count())
        );

        let mut lookup_table = Array3::<usize>::zeros(roi.dim());

        let mut target_px = 0;

        for (&roi_px, lookup_idx) in izip!(roi.iter(), lookup_table.iter_mut()) {
            if roi_px {
                *lookup_idx = target_px;
                target_px += 1;
            }
        }

        let chunk_size = 2500;
        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = array.axis_chunks_iter_mut(Axis(0), chunk_size).collect();
        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, mut chunk)| -> Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(&self._filename).unwrap();

            let roi_cycle = roi.axis_iter(Axis(0)).cycle();
            let lookup_cycle = lookup_table.axis_iter(Axis(0)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the mask
            let roi_cycle = roi_cycle.skip(start % roi.dim().0);
            let lookup_cycle = lookup_cycle.skip(start % roi.dim().0);

            // This has to work differently from the other versions because
            // each tick of the 0th axis corresponds to a volume, not a frame,
            // so it has to be repeated num_slices times
            match registration {
                Some(reg) => {
                    izip!(
                        local_frames.iter(),
                        // chunk.axis_iter_mut(Axis(0))
                        (0..chunk.dim().0)
                            .flat_map(|x| std::iter::repeat(x).take(num_slices)),
                        roi_cycle,
                        lookup_cycle
                    )
                        .try_for_each(
                            |(&this_frame, this_chunk_idx, roi_plane, lookup_plane)|
                            -> Result<(), CorrosiffError> {
                            extract_intensity_mask_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut chunk.index_axis_mut(Axis(0), this_chunk_idx),
                                &roi_plane,
                                &lookup_plane,
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                        })?;
                },
                None => {
                    izip!(
                        local_frames.iter(),
                        // chunk.axis_iter_mut(Axis(0)),
                        (0..chunk.dim().0)
                            .flat_map(|x| std::iter::repeat(x).take(num_slices)),
                        roi_cycle,
                        lookup_cycle
                    )
                        .try_for_each(
                            |(&this_frame, this_chunk_idx, roi_plane, lookup_plane)|
                            -> Result<(), CorrosiffError> {
                            extract_intensity_mask(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                // &mut this_chunk,
                                &mut chunk.index_axis_mut(Axis(0), this_chunk_idx),
                                &roi_plane,
                                &lookup_plane,
                            )?; Ok(())
                        })?;
                }
            }
            Ok(())
        })?;

        Ok(array)
    }

    /// Sums the intensity of the frames requested within the
    /// region of interest (ROI) specified by the boolean array
    /// `roi`. The ROI should be the same shape as the frames' 
    /// `y` and `x` dimensions
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 2D boolean array with the same shape as the frames'
    /// `y` and `x` dimensions. The ROI is a mask which will be used
    /// to sum the intensity of the frames requested.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array1<u64>, CorrosiffError>` - An array of `u64` values
    /// corresponding to the sum of the intensity of the frames requested
    /// within the ROI specified.
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let roi = Array2::<bool>::from_elem((512, 512), true);
    /// // Set the ROI to false in the middle
    /// roi.slice(s![200..300, 200..300]).fill(false);
    /// 
    /// let sum = reader.sum_roi_flat(&roi, &[0, 1, 2], None);
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds, do not share the same shape, or the ROI does not share
    /// the same shape as the frames (the underlying `DimensionsError` is attached to this error)
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_volume` - for a 3D ROI mask
    /// - `sum_rois_flat` - for a set of 2D ROI masks
    /// - `sum_roi_volume` - for a set of 3D ROI masks
    pub fn sum_roi_flat(
        &self,
        roi : &ArrayView2<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<Array1<u64>, CorrosiffError> {
         // Check that the frames are in bounds
         _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != roi.dim() {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple(roi.dim()),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut array = Array1::<u64>::zeros(frames.len());

        let op = |frames : &[u64], chunk : &mut ArrayViewMut1<u64>, reader : &mut File| 
        -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    frames.iter().zip(chunk.iter_mut())
                        .try_for_each(
                            |(&this_frame, mut this_frame_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_intensity_mask_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_sum,
                                &roi.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                    })?;
                },
                None => {
                    frames.iter().zip(chunk.iter_mut())
                        .try_for_each(
                            |(&this_frame, mut this_frame_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_intensity_mask(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_sum,
                                &roi.view(),
                            )?; Ok(())
                        })?;
                },
            }
            Ok(())
        };
        
        parallelize_op!(
            array,
            2500,
            frames,
            self._filename,
            op
        );

        Ok(array)
    }

    /// Sums the intensity of the frames requested within the
    /// region of interest (ROI) specified by the boolean array
    /// `roi`. The ROI should have final two dimensions the same
    /// as the frames' `y` and `x` dimensions, and the first
    /// dimension will be looped over while cycling through frames
    /// (i.e. frame 1 will be masked by the first plane of the ROI,
    /// frame 2 by the second plane, etc).
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 3D boolean array with the first dimension equal to
    /// the number of planes in the ROI, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROI is
    /// a mask which will be used to sum the intensity of the frames requested.
    /// Each plane is cycled through in parallel to the frames.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array1<u64>, CorrosiffError>` - An array of `u64` values
    /// corresponding to the sum of the intensity of the frames requested
    /// within the ROIs specified. 
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let roi = Array3::<bool>::from_elem((3, 512, 512), true);
    /// // Set the ROI to false in the middle
    /// roi.slice(s![.., 200..300, 200..300]).fill(false);
    /// 
    /// let sum = reader.sum_roi_volume(&roi, &[0, 1, 2], None);
    /// 
    /// // The sum will be a 1D array of u64 values corresponding to the
    /// // sum of the intensity of the frames requested within the ROIs specified.
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds, do not share the same shape, or the ROI does not share
    /// the same shape as the frames (the underlying `DimensionsError` is attached to this error)
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_flat` - for a 2D ROI mask
    /// - `sum_rois_flat` - for a set of 2D ROI masks
    /// - `sum_rois_volume` - for a set of 3D ROI masks
    pub fn sum_roi_volume(
        &self,
        roi : &ArrayView3<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<Array1<u64>, CorrosiffError>{

        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;

        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (roi.dim().1, roi.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((roi.dim().1, roi.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut array = Array1::<u64>::zeros(frames.len());

        // SIGH my macro skills are not good enough for this job.
        let chunk_size = 2500;

        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = array.axis_chunks_iter_mut(Axis(0), chunk_size).collect();

        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, mut chunk)| -> Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(self._filename.clone()).unwrap();
            
            let roi_cycle = roi.axis_iter(Axis(0)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the roi_cycle
            let roi_cycle = roi_cycle.skip(start % roi.dim().0);

            match registration {
                Some(reg) => {
                    izip!(local_frames.iter(),chunk.iter_mut(), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_sum, roi_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_intensity_mask_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_sum,
                                &roi_plane,
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                    })?;
                },
                None => {
                    izip!(local_frames.iter(), chunk.iter_mut(), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_sum, roi_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_intensity_mask(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_sum,
                                &roi_plane,
                            )?; Ok(())
                        })?;
                },
            }
            Ok(())
            }
        )?;

        // parallelize_op![
        //     array,
        //     2500,
        //     frames,
        //     self._filename,
        //     op
        // ];
        
        Ok(array)

    }

    /// Sums a collection of masks over each frame requested
    /// and returns the sum of the intensity of the frames requested
    /// within each mask. Each ROI should have the same shape
    /// as the frames' `y` and `x` dimensions. Runs slightly slower
    /// than applying the mask to just one ROI, so the gains scale as ~n_masks.
    /// 
    /// ## Arguments
    /// 
    /// * `rois` - A 3D boolean array with the first dimension equal to
    /// the number of ROIs, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROIs are
    /// masks which will be used to sum the intensity of the frames requested.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array2<u64>, CorrosiffError>` - An array of `u64` values
    /// corresponding to the sum of the intensity of the frames requested
    /// within each ROI specified. Shape is `(frames.len(), rois.len())`
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let rois = Array3::<bool>::from_elem((3, 512, 512), true);
    /// // TODO FINISH
    /// ```
    /// ## See also
    /// 
    /// - `sum_roi_flat` - for a 2D ROI mask
    /// - `sum_rois_volume` - for a set of 3D ROI masks
    /// - `sum_roi_volume` - for a single 3D ROI mask
    pub fn sum_rois_flat(
        &self,
        rois : &ArrayView3<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<Array2<u64>, CorrosiffError> {
        
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (rois.dim().1, rois.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((rois.dim().1, rois.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, frames)?;

        let mut array = Array2::<u64>::zeros((frames.len(), rois.dim().0));

        let op = | frames : &[u64], chunk : &mut ArrayViewMut2<u64>, reader : &mut File |
        -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                    .try_for_each(
                        | (&this_frame, mut this_frame_sums) |
                        -> Result<(), CorrosiffError> {
                            Ok(
                                sum_intensity_masks_registered(
                                    reader,
                                    &self._ifds[this_frame as usize],
                                    &mut this_frame_sums,
                                    &rois.view(),
                                    *reg.get(&this_frame).unwrap(),
                                )?
                            )
                        }
                    )?;
                    Ok(())
                }
                None => {
                    frames.iter().zip(chunk.axis_iter_mut(Axis(0)))
                    .try_for_each(
                        | (&this_frame, mut this_frame_sums) |
                        -> Result<(), CorrosiffError> {
                            Ok(sum_intensity_masks(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_sums,
                                &rois.view(),
                            )?)
                        }
                    )?;
                    Ok(())
                }
            }
        };

        parallelize_op!(
            array, 
            2500, 
            frames, 
            self._filename,
            op
        );

        Ok(array)
    }

    /// Sums a collection of 3d masks over each frame requested
    /// and returns the sum of the intensity of the frames requested
    /// within each mask. Each ROI (the last 2 dimensions) should have the same shape
    /// as the frames' `y` and `x` dimensions. Iterates across the
    /// dimension 1 (i.e. the second dimension of the `rois` array)
    /// alongside the frames, so for each mask (masks are indexed along the slowest
    /// dimension) the 1st frame is applied to the the first plane of the mask,
    /// the 2nd frame to the second plane, and the nth frame to the 
    /// n % roi.dim().1 plane. Runs slightly slower than applying
    /// the mask to just one ROI, so the gains scale as ~n_masks.
    /// 
    /// ## Arguments
    /// 
    /// * `rois` - A 4D boolean array with the first dimension equal to
    /// the number of ROIs, the second dimension corresponding to each z
    /// plane for each mask, and the third and fourth dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROIs are
    /// masks which will be used to sum the intensity of the frames requested.
    /// Each plane is cycled through in parallel to the frames.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<Array2<u64>, CorrosiffError>` - An array of `u64` values
    /// corresponding to the sum of the intensity of the frames requested
    /// within each ROI specified. Shape is `(frames.len(), rois.len())`
    /// 
    /// ## Example
    /// ```rust,ignore
    /// 
    /// /* -- snip -- */
    /// // 6 masks, 3 planes, 512 y pixels, 256 x pixels
    /// let masks = Array4::<bool>::from_elem((6, 3, 512, 256), true);
    /// // Set the ROIs to random values
    /// use rand
    /// masks.mapv_inplace(|_| rand::random::<bool>());
    /// 
    /// let frames = vec![0,1,2,3,4,5,6];
    /// 
    /// let masks_summed_at_once = reader.sum_rois_volume(
    ///    &masks,
    ///    &frames,
    ///    None
    /// )
    /// 
    /// // TODO
    /// ```
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_flat` - for a 2D ROI mask
    /// - `sum_rois_flat` - for a set of 2D ROI masks
    /// - `sum_roi_volume` - for a single 3D ROI mask
    pub fn sum_rois_volume(
        &self,
        rois : &ArrayView4<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<Array2<u64>, CorrosiffError> {
        
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;

        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or_else(||DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (rois.dim().2, rois.dim().3) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((rois.dim().2, rois.dim().3)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut array = Array2::<u64>::zeros((frames.len(), rois.dim().0));

        // SIGH my macro skills are not good enough for this job.
        let chunk_size = 2500;

        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = array.axis_chunks_iter_mut(Axis(0), chunk_size).collect();

        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, mut chunk)| -> Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(self._filename.clone()).unwrap();
            
            let roi_cycle = rois.axis_iter(Axis(1)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the roi_cycle
            let roi_cycle = roi_cycle.skip(start % rois.dim().1);

            match registration {
                Some(reg) => {
                    izip!(local_frames.iter(),chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_sum, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_intensity_masks_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_sum,
                                &rois_plane,
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                    })?;
                },
                None => {
                    izip!(local_frames.iter(), chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_sum, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_intensity_masks(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_sum,
                                &rois_plane,
                            )?; Ok(())
                        })?;
                },
            }
            Ok(())
            }
        )?;

        // parallelize_op![
        //     array,
        //     2500,
        //     frames,
        //     self._filename,
        //     op
        // ];
        
        Ok(array)
    }

    /// Computes the empirical lifetime of the region masked
    /// by the ROI specified by the boolean array `roi`. The ROI
    /// should have the same shape as the frames' `y` and `x` dimensions.
    /// Returns the lifetime and intensity data in a tuple, since it's
    /// rare that you can use the lifetime without needing to refer to
    /// the intensity as well
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 2D boolean array with the same shape as the frames'
    /// `y` and `x` dimensions. The ROI is a mask determining which pixels
    /// of each frame should be used to compute the lifetime.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array1<f64>, Array1<u64>), CorrosiffError>` - A tuple
    /// containing the lifetime and intensity data for the frames requested
    /// (respectively), each of shape `(frames.len(),)`. 
    /// The lifetime is in units of arrival time bins, and
    /// to transform to picoseconds, multiply by the `picoseconds_per_bin`
    /// value in the metadata. The intensity data is photon counts in the
    /// ROI (as in `sum_roi_flat`).
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// // TODO
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds, do not share the same shape, or the ROI does not share
    /// the same shape as the frames (the underlying `DimensionsError` is attached to this error)
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_flim_volume` - for a 3D ROI mask
    /// - `sum_rois_flim_flat` - for a set of 2D ROI masks
    /// - `sum_rois_flim_volume` - for a set of 3D ROI masks
    pub fn sum_roi_flim_flat(
        &self,
        roi : &ArrayView2<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<(Array1<f64>, Array1<u64>), CorrosiffError> {
        
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != roi.dim() {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple(roi.dim()),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array1::<u64>::zeros(frames.len());
        let mut lifetime_array = Array1::<f64>::zeros(frames.len());

        let op = 
        |
            frames : &[u64],
            chunk_intensity : &mut ArrayViewMut1<u64>,
            chunk_lifetime : &mut ArrayViewMut1<f64>,
            reader : &mut File
        | -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    izip!(
                        frames,
                        chunk_lifetime.iter_mut(),
                        chunk_intensity.iter_mut(),
                    ).try_for_each(
                            |(&this_frame, mut lifetime_sum, mut intensity_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_mask_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut lifetime_sum,
                                &mut intensity_sum,
                                &roi.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?;
                            Ok(())
                    })?;
                },
                None => {
                    izip!(
                        frames,
                        chunk_lifetime.iter_mut(),
                        chunk_intensity.iter_mut(),
                    ).try_for_each(
                            |(&this_frame, mut lifetime_sum, mut intensity_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_mask(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut lifetime_sum,
                                &mut intensity_sum,
                                &roi.view(),
                            )?;
                            Ok(())
                        })?;
                },
            }
            Ok(())
        };

        parallelize_op!(
            (intensity_array, lifetime_array),
            2500,
            frames,
            self._filename,
            op
        );

        Ok((lifetime_array, intensity_array))
    }

    /// Computes the empirical lifetime of the region masked
    /// by the 3d ROI specified by the boolean array `roi`. The ROIs
    /// slowest axis is the slice axis, and the last two dimensions
    /// should have the same shape as the frames' `y` and `x` dimensions.
    /// The z planes will be iterated through alongside the frames, so
    /// that `roi.slice(s![0, .., ..])` will be applied to the first frame,
    /// `roi.slice(s![1, .., ..])` to the second frame, in general the
    /// `n % roi.dim().0` plane to the nth frame.
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 3D boolean array with the first dimension equal to
    /// the number of planes in the ROI, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROI is
    /// a mask which will be used to sum the intensity of the frames requested
    /// and compute the mean arrival time of the photons in the ROI.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array1<f64>, Array1<u64>), CorrosiffError>` - A tuple
    /// containing the lifetime and intensity data for the frames requested
    /// (respectively), each of shape `(frames.len(),)`. The lifetime
    /// is in units of arrival time bins, and to transform to picoseconds,
    /// multiply by the `picoseconds_per_bin` value in the metadata. The
    /// intensity data is photon counts in the ROI (as in `sum_roi_flat`).
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// // TODO
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds, do not share the same shape, or the ROI does not share
    /// the same shape as the frames (the underlying `DimensionsError` is attached to this error)
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_flim_flat` - for a 2D ROI mask
    /// - `sum_rois_flim_flat` - for a set of 2D ROI masks
    /// - `sum_rois_flim_volume` - for a set of 3D ROI masks
    pub fn sum_roi_flim_volume(
        &self,
        roi : &ArrayView3<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<(Array1<f64>, Array1<u64>), CorrosiffError> {
                
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (roi.dim().1, roi.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((roi.dim().1, roi.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array1::<u64>::zeros(frames.len());
        let mut lifetime_array = Array1::<f64>::zeros(frames.len());

        // Still not ready for macro magic! Hope to revisit this.
        let chunk_size = 2500;

        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let chunks_together = izip!(
            intensity_array.axis_chunks_iter_mut(Axis(0), chunk_size),
            lifetime_array.axis_chunks_iter_mut(Axis(0), chunk_size)
        ).collect::<Vec<_>>();

        chunks_together.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, (mut intensity_chunk, mut lifetime_chunk))|
            -> Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(self._filename.clone()).unwrap();
            
            let roi_cycle = roi.axis_iter(Axis(0)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the roi_cycle
            let roi_cycle = roi_cycle.skip(start % roi.dim().0);

            match registration {
                Some(reg) => {
                    izip!(local_frames.iter(),intensity_chunk.iter_mut(), lifetime_chunk.iter_mut(), roi_cycle)
                        .try_for_each(
                            |(&this_frame, this_intensity, this_lifetime, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_mask_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                this_lifetime,
                                this_intensity,
                                &rois_plane,
                                *reg.get(&this_frame).unwrap(),
                            )?; Ok(())
                    })?;
                },
                None => {
                    izip!(local_frames.iter(), intensity_chunk.iter_mut(), lifetime_chunk.iter_mut(), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_intensity, mut this_lifetime, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_mask(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_lifetime,
                                &mut this_intensity,
                                &rois_plane,
                            )?; Ok(())
                        })?;
                },
            }
            Ok(())
            }
        )?;

        Ok((lifetime_array, intensity_array))
    }

    /// Sums a collection of masks over each frame requested
    /// and returns the empirical lifetime and intensity of the frames requested
    /// within each mask. Each ROI should have the same shape
    /// as the frames' `y` and `x` dimensions. Runs slightly slower
    /// than applying the mask to just one ROI, so the gains scale as ~n_masks.
    /// 
    /// ## Arguments
    /// 
    /// * `rois` - A 3D boolean array with the first dimension equal to
    /// the number of ROIs, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROIs are
    /// masks which will be used to sum the intensity of the frames requested.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array2<f64>, Array2<u64>), CorrosiffError>` - A tuple
    /// containing the lifetime and intensity data for the frames requested
    /// within each ROI specified. The first element of the tuple is the
    /// lifetime data, and the second element is the intensity data. The
    /// shape of each array is `(frames.len(), rois.len())`
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// let rois = Array3::<bool>::from_elem((3, 512, 512), true);
    /// // TODO FINISH
    /// ```
    /// ## See also
    /// 
    /// - `sum_roi_flim_flat` - for a 2D ROI mask
    /// - `sum_rois_flim_volume` - for a set of 3D ROI masks
    /// - `sum_roi_flim_volume` - for a single 3D ROI mask
    pub fn sum_rois_flim_flat(
        &self,
        rois : &ArrayView3<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<(Array2<f64>, Array2<u64>), CorrosiffError> {
                        
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (rois.dim().1, rois.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((rois.dim().1, rois.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array2::<u64>::zeros((frames.len(), rois.dim().0));
        let mut lifetime_array = Array2::<f64>::zeros((frames.len(), rois.dim().0));

        let op = |
            frames : &[u64],
            chunk_lifetime : &mut ArrayViewMut2<f64>,
            chunk_intensity : &mut ArrayViewMut2<u64>,
            reader : &mut File
            | -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    izip!(
                        frames,
                        chunk_lifetime.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0)),
                    ).try_for_each(
                            |(&this_frame, mut this_frame_lifetimes, mut this_frame_intensities)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_masks_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_lifetimes,
                                &mut this_frame_intensities,
                                &rois,
                                *reg.get(&this_frame).unwrap(),
                            )?;
                            Ok(())
                    })?;
                    Ok(())
                },
                None => {
                    izip!(
                        frames,
                        chunk_lifetime.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0)),
                    ).try_for_each(
                            |(&this_frame, mut this_frame_lifetimes, mut this_frame_intensities)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_masks(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_lifetimes,
                                &mut this_frame_intensities,
                                &rois,
                            )?;
                            Ok(())
                        })?;
                    Ok(())
                }
            }
        };

        parallelize_op!(
            (lifetime_array, intensity_array),
            2500,
            frames,
            self._filename,
            op
        );

        Ok((lifetime_array, intensity_array))
    }

    /// Sums a collection of 3d masks over each frame requested
    /// and returns the empirical lifetime and total intensity of the frames requested
    /// within each mask. Each ROI (the last 2 dimensions) should have the same shape
    /// as the frames' `y` and `x` dimensions. Iterates across the
    /// dimension 1 (i.e. the second dimension of the `rois` array)
    /// alongside the frames, so for each mask (masks are indexed along the slowest
    /// dimension) the 1st frame is applied to the the first plane of the mask,
    /// the 2nd frame to the second plane, and the nth frame to the 
    /// n % roi.dim().1 plane. Runs slightly slower than applying
    /// the mask to just one ROI, so the gains scale as ~n_masks.
    /// 
    /// ## Arguments
    /// 
    /// * `rois` - A 4D boolean array with the first dimension equal to
    /// the number of ROIs, the second dimension corresponding to each z
    /// plane for each mask, and the third and fourth dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROIs are
    /// masks which will be used to sum the intensity of the frames requested.
    /// Each plane is cycled through in parallel to the frames.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array2<f64>, Array2<u64>), CorrosiffError>` - A tuple
    /// of arrays, the first of which contains the lifetime data for the frames
    /// (in units of arrival time bins) and the second of which contains the
    /// intensity data for the frames requested within each ROI specified. The
    /// shape of each array is `(frames.len(), rois.len())`. To convert the
    /// lifetime data to picoseconds, multiply by the `picoseconds_per_bin` value
    /// in the metadata.
    /// 
    /// ## Example
    /// ```rust,ignore
    /// 
    /// /* -- snip -- */
    /// // 6 masks, 3 planes, 512 y pixels, 256 x pixels
    /// let masks = Array4::<bool>::from_elem((6, 3, 512, 256), true);
    /// // Set the ROIs to random values
    /// use rand
    /// masks.mapv_inplace(|_| rand::random::<bool>());
    /// 
    /// let frames = vec![0,1,2,3,4,5,6];
    /// 
    /// let masks_summed_at_once = reader.sum_rois_volume(
    ///    &masks,
    ///    &frames,
    ///    None
    /// )
    /// 
    /// // TODO
    /// ```
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_flim_flat` - for a 2D ROI mask
    /// - `sum_rois_flim_flat` - for a set of 2D ROI masks
    /// - `sum_roi_flim_volume` - for a single 3D ROI mask
    /// - `sum_rois_volume` - the intensity-only version.
    pub fn sum_rois_flim_volume(
        &self,
        rois : &ArrayView4<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>
    ) -> Result<(Array2<f64>, Array2<u64>), CorrosiffError> {

        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (rois.dim().2, rois.dim().3) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((rois.dim().2, rois.dim().3)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array2::<u64>::zeros((frames.len(), rois.dim().0));
        let mut lifetime_array = Array2::<f64>::zeros((frames.len(), rois.dim().0));

        // More messy parallelization that _almost_ boilerplate but not quite
        let chunk_size = 2500;

        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = izip!(
            intensity_array.axis_chunks_iter_mut(Axis(0), chunk_size),
            lifetime_array.axis_chunks_iter_mut(Axis(0), chunk_size)
        ).collect();

        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, (mut intensity_chunk, mut lifetime_chunk))|
            -> Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(self._filename.clone()).unwrap();
            
            let roi_cycle = rois.axis_iter(Axis(1)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the roi_cycle
            let roi_cycle = roi_cycle.skip(start % rois.dim().1);

            match registration {
                Some(reg) => {
                    izip!(local_frames.iter(), intensity_chunk.axis_iter_mut(Axis(0)), lifetime_chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_intensities, mut this_frame_lifetimes, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_masks_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_lifetimes,
                                &mut this_frame_intensities,
                                &rois_plane,
                                *reg.get(&this_frame).unwrap(),
                            )?;
                            Ok(())
                    })?;
                },
                None => {
                    izip!(local_frames.iter(), intensity_chunk.axis_iter_mut(Axis(0)), lifetime_chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_intensities, mut this_frame_lifetimes, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_lifetime_intensity_masks(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_lifetimes,
                                &mut this_frame_intensities,
                                &rois_plane,
                            )?;
                            Ok(())
                        })?;
                },
            }
            Ok(())
            }
        )?;

        Ok((lifetime_array, intensity_array))
    }

    /// Computes the phasor representation of the region masked
    /// by the ROI specified by the boolean array `roi`. The ROI
    /// should have the same shape as the frames' `y` and `x` dimensions.
    /// Returns the phasor and intensity data in a tuple, since it's
    /// rare that you can use the phasor without needing to refer to
    /// the intensity as well
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 2D boolean array with the same shape as the frames'
    /// `y` and `x` dimensions. The ROI is a mask determining which pixels
    /// of each frame should be used to compute the phasor
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array1<Complex<f64>>, Array1<u64>), CorrosiffError>` - A tuple
    /// containing the phasor data and intensity data for the frames requested
    /// (respectively), each of shape `(frames.len(),)`. 
    /// The lifetime is in units of the histogram width, and
    /// to transform to picoseconds, you need to know the pulse
    /// repetition rate. The intensity data is photon counts in the
    /// ROI (as in `sum_roi_flat`).
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// // TODO
    /// ```
    /// 
    /// ## Errors
    /// 
    /// * `CorrosiffError::DimensionsError(DimensionsError)` - If the frames requested
    /// are out of bounds, do not share the same shape, or the ROI does not share
    /// the same shape as the frames (the underlying `DimensionsError` is attached to this error)
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_phasor_volume` - for a 3D ROI mask
    /// - `sum_rois_phasor_flat` - for a set of 2D ROI masks
    /// - `sum_rois_phasor_volume` - for a set of 3D ROI masks
    pub fn sum_roi_phasor_flat(
        &self,
        roi : &ArrayView2<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<(Array1<Complex<f64>>, Array1<u64>), CorrosiffError> {

        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != roi.dim() {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple(roi.dim()),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array1::<u64>::zeros(frames.len());
        let mut phasor_array = Array1::<Complex<f64>>::zeros(frames.len());

        let num_tau = self.file_format.num_flim_tau_bins().unwrap();
        let cos_lookup = Array1::from_iter(
            (0..num_tau)
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).cos())
        );

        let sin_lookup = Array1::from_iter(
            (0..self.file_format.num_flim_tau_bins().unwrap())
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).sin())
        );

        let op = 
        |
            frames : &[u64],
            chunk_intensity : &mut ArrayViewMut1<u64>,
            chunk_phasor : &mut ArrayViewMut1<Complex<f64>>,
            reader : &mut File
        | -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    izip!(
                        frames,
                        chunk_phasor.iter_mut(),
                        chunk_intensity.iter_mut(),
                    ).try_for_each(
                            |(&this_frame, mut phasor_sum, mut intensity_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_mask_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut phasor_sum,
                                &mut intensity_sum,
                                &roi.view(),
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?;
                            Ok(())
                    })?;
                },
                None => {
                    izip!(
                        frames,
                        chunk_phasor.iter_mut(),
                        chunk_intensity.iter_mut(),
                    ).try_for_each(
                            |(&this_frame, mut phasor_sum, mut intensity_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_mask(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut phasor_sum,
                                &mut intensity_sum,
                                &roi.view(),
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                            )?;
                            Ok(())
                        })?;
                },
            }
            Ok(())
        };

        parallelize_op!(
            (intensity_array, phasor_array),
            2500,
            frames,
            self._filename,
            op
        );

        Ok((phasor_array, intensity_array))
    }


    /// Computes the phasor representation of the region masked
    /// by the 3d ROI specified by the boolean array `roi`. The ROI's
    /// slowest axis is the slice axis, and the last two dimensions
    /// should have the same shape as the frames' `y` and `x` dimensions.
    /// The z planes will be iterated through alongside the frames, so
    /// that `roi.slice(s![0, .., ..])` will be applied to the first frame,
    /// `roi.slice(s![1, .., ..])` to the second frame, in general the
    ///  `n % roi.dim().0` plane to the nth frame.
    /// 
    /// ## Arguments
    /// 
    /// * `roi` - A 3D boolean array with the first dimension equal to
    /// the number of planes in the ROI, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROI is
    /// a mask which will be used to sum the intensity of the frames requested
    /// and compute the phasor of the photons in the ROI.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array1<Complex<f64>>, Array1<u64>), CorrosiffError>` - A tuple
    /// containing the phasor and intensity data for the frames requested
    /// (respectively), each of shape `(frames.len(),)`. The phasor
    /// is a complex number
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// // TODO
    /// 
    /// ```
    /// 
    /// ## See also
    /// 
    /// * `sum_roi_phasor_flat` - for a 2D ROI mask
    /// 
    /// * `sum_rois_phasor_flat` - for a set of 2D ROI masks
    /// 
    /// * `sum_rois_phasor_volume` - for a set of 3D ROI masks
    pub fn sum_roi_phasor_volume(
        &self,
        roi : &ArrayView3<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<(Array1<Complex<f64>>, Array1<u64>), CorrosiffError> {
        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;
        
        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (roi.dim().1, roi.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((roi.dim().1, roi.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array1::<u64>::zeros(frames.len());
        let mut phasor_array = Array1::<Complex<f64>>::zeros(frames.len());

        let num_tau = self.file_format.num_flim_tau_bins().unwrap();
        let cos_lookup = Array1::from_iter(
            (0..num_tau)
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).cos())
        );

        let sin_lookup = Array1::from_iter(
            (0..self.file_format.num_flim_tau_bins().unwrap())
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).sin())
        );

        // Still not ready for macro magic! Hope to revisit this
        let chunk_size = 2500;

        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = izip!(
            intensity_array.axis_chunks_iter_mut(Axis(0), chunk_size),
            phasor_array.axis_chunks_iter_mut(Axis(0), chunk_size)
        ).collect();
        
        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, (mut intensity_chunk, mut phasor_chunk))| ->
            Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(self._filename.clone()).unwrap();

            let roi_cycle = roi.axis_iter(Axis(0)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the roi_cycle
            let roi_cycle = roi_cycle.skip(start % roi.dim().0);

            match registration {
                Some(reg) => {
                    izip!(local_frames.iter(), intensity_chunk.iter_mut(), phasor_chunk.iter_mut(), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_intensities, mut this_frame_phasors, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_mask_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_phasors,
                                &mut this_frame_intensities,
                                &rois_plane,
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?;
                            Ok(())
                    })?;
                },
                None => {
                    izip!(local_frames.iter(), intensity_chunk.iter_mut(), phasor_chunk.iter_mut(), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_intensities, mut this_frame_phasors, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_mask(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_phasors,
                                &mut this_frame_intensities,
                                &rois_plane,
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                            )?;
                            Ok(())
                        })?;
                },
            }
            Ok(())
            }
        )?;

        Ok((phasor_array, intensity_array))
    }

    /// Sums a collection of masks over each frame requested
    /// and returns the phasor representation of photon arrival times
    /// and intensity of the frames requested. Each ROI should have the same shape
    /// as the frames' `y` and `x` dimensions. Runs slightly slower
    /// than applying the mask to just one ROI, so the gains scale as ~n_masks.
    /// 
    /// ## Arguments
    /// 
    /// * `rois` - A 3D boolean array with the first dimension equal to
    /// the number of ROIs, and the second and third dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROIs are
    /// masks which will be used to sum the intensity of the frames requested.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array2<Complex<f64>>, Array2<u64>), CorrosiffError>` - A tuple
    /// containing the phasor and intensity data for the frames requested
    /// within each ROI specified. The first element of the tuple is the
    /// phasor data, and the second element is the intensity data.
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// // TODO
    /// ```
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_phasor_flat` - for a 2D ROI mask
    /// - `sum_rois_phasor_volume` - for a set of 3D ROI masks
    /// - `sum_roi_phasor_volume` - for a single 3D ROI mask
    pub fn sum_rois_phasor_flat(
        &self,
        rois : &ArrayView3<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<(Array2<Complex<f64>>, Array2<u64>), CorrosiffError> {

        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;

        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (rois.dim().1, rois.dim().2) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((rois.dim().1, rois.dim().2)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array2::<u64>::zeros((frames.len(), rois.dim().0));
        let mut phasor_array = Array2::<Complex<f64>>::zeros((frames.len(), rois.dim().0));

        let num_tau = self.file_format.num_flim_tau_bins().unwrap();
        let cos_lookup = Array1::from_iter(
            (0..num_tau)
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).cos())
        );

        let sin_lookup = Array1::from_iter(
            (0..self.file_format.num_flim_tau_bins().unwrap())
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).sin())
        );

        let op =
        |
            frames : &[u64],
            chunk_intensity : &mut ArrayViewMut2<u64>,
            chunk_phasor : &mut ArrayViewMut2<Complex<f64>>,
            reader : &mut File
        | -> Result<(), CorrosiffError> {
            match registration {
                Some(reg) => {
                    izip!(
                        frames,
                        chunk_phasor.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0)),
                    ).try_for_each(
                            |(&this_frame, mut phasor_sum, mut intensity_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_masks_registered(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut phasor_sum,
                                &mut intensity_sum,
                                &rois.view(),
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?;
                            Ok(())
                    })?;
                },
                None => {
                    izip!(
                        frames,
                        chunk_phasor.axis_iter_mut(Axis(0)),
                        chunk_intensity.axis_iter_mut(Axis(0)),
                    ).try_for_each(
                            |(&this_frame, mut phasor_sum, mut intensity_sum)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_masks(
                                reader,
                                &self._ifds[this_frame as usize],
                                &mut phasor_sum,
                                &mut intensity_sum,
                                &rois.view(),
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                            )?;
                            Ok(())
                        })?;
                },
            }
            Ok(())
        };

        parallelize_op!(
            (intensity_array, phasor_array),
            2500,
            frames,
            self._filename,
            op
        );

        Ok((phasor_array, intensity_array))
    }


    /// Sums a collection of 3d masks over each frame requested
    /// and returns the phasor representation of photon arrival times
    /// and intensity of the frames requested. Each ROI should have the same shape
    /// as the frames' `y` and `x` dimensions. Iterates across the
    /// dimension 1 (i.e. the second dimension of the `rois` array)
    /// alongside the frames, so for each mask (masks are indexed along the slowest
    /// dimension) the 1st frame is applied to the the first plane of the mask,
    /// the 2nd frame to the second plane, and the nth frame to the
    /// n % roi.dim().1 plane. Runs slightly slower than applying
    /// the mask to just one ROI, so the gains scale as ~n_masks.
    /// 
    /// ## Arguments
    /// 
    /// * `rois` - A 4D boolean array with the first dimension equal to
    /// the number of ROIs, the second dimension corresponding to each z
    /// plane for each mask, and the third and fourth dimensions
    /// corresponding to the `y` and `x` dimensions of the frames. The ROIs are
    /// masks which will be used to sum the intensity of the frames requested.
    /// 
    /// * `frames` - A slice of `u64` values corresponding to the
    /// frame numbers to retrieve
    /// 
    /// * `registration` - An optional `HashMap<u64, (i32, i32)>` which
    /// contains the pixel shifts for each frame. If this is `None`,
    /// the frames are read unregistered (runs faster).
    /// 
    /// ## Returns
    /// 
    /// * `Result<(Array2<Complex<f64>>, Array2<u64>), CorrosiffError>` - A tuple
    /// containing the phasor and intensity data for the frames requested
    /// within each ROI specified. The first element of the tuple is the
    /// phasor data, and the second element is the intensity data.
    /// 
    /// ## Example
    /// 
    /// ```rust, ignore
    /// let reader = SiffReader::open("file.siff");
    /// // TODO
    /// ```
    /// 
    /// ## See also
    /// 
    /// - `sum_roi_phasor_flat` - for a 2D ROI mask
    /// - `sum_rois_phasor_flat` - for a set of 2D ROI masks
    /// - `sum_roi_phasor_volume` - for a single 3D ROI mask
    pub fn sum_rois_phasor_volume(
        &self,
        rois : &ArrayView4<bool>,
        frames : &[u64],
        registration : Option<&RegistrationDict>,
    ) -> Result<(Array2<Complex<f64>>, Array2<u64>), CorrosiffError> {

        // Check that the frames are in bounds
        _check_frames_in_bounds(&frames, &self._ifds)?;

        // Check that the frames share a shape with the mask
        let array_dims = self._image_dims.clone().or_else(
            || _check_shared_shape(frames, &self._ifds)
        ).ok_or(DimensionsError::NoConsistentDimensions)?;

        if array_dims.to_tuple() != (rois.dim().2, rois.dim().3) {
            return Err(FramesError::DimensionsError(
                DimensionsError::MismatchedDimensions{
                    required : array_dims,
                    requested : Dimensions::from_tuple((rois.dim().2, rois.dim().3)),
                }
            ).into());
        }

        let mut registration = registration;

        // Check that every frame requested has a registration value,
        // if registration is used. Otherwise just ignore.
        _check_registration(&mut registration, &frames)?;

        let mut intensity_array = Array2::<u64>::zeros((frames.len(), rois.dim().0));
        let mut phasor_array = Array2::<Complex<f64>>::zeros((frames.len(), rois.dim().0));

        let num_tau = self.file_format.num_flim_tau_bins().unwrap();
        let cos_lookup = Array1::from_iter(
            (0..num_tau)
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).cos())
        );

        let sin_lookup = Array1::from_iter(
            (0..self.file_format.num_flim_tau_bins().unwrap())
            .map(|x| (2.0_f64*std::f64::consts::PI*x as f64/num_tau as f64).sin())
        );

        // Still not ready for macro magic! Hope to revisit this

        let chunk_size = 2500;

        let n_threads = frames.len()/chunk_size + 1;
        let remainder = frames.len() % n_threads;

        // Compute the bounds for each threads operation
        let mut offsets = vec![];
        let mut start = 0;
        for i in 0..n_threads {
            let end = start + chunk_size + if i < remainder { 1 } else { 0 };
            offsets.push((start, end));
            start = end;
        }

        // Create an array of chunks to parallelize
        let array_chunks : Vec<_> = izip!(
            intensity_array.axis_chunks_iter_mut(Axis(0), chunk_size),
            phasor_array.axis_chunks_iter_mut(Axis(0), chunk_size)
        ).collect();

        array_chunks.into_par_iter().enumerate().try_for_each(
            |(chunk_idx, (mut intensity_chunk, mut phasor_chunk))| ->
            Result<(), CorrosiffError> {
            // Get the frame numbers and ifds for the frames in the chunk
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(frames.len());

            let local_frames = &frames[start..end];
            let mut local_f = File::open(self._filename.clone()).unwrap();

            let roi_cycle = rois.axis_iter(Axis(1)).cycle();
            // roi_cycle needs to be incremented by the start value
            // modulo the length of the roi_cycle
            let roi_cycle = roi_cycle.skip(start % rois.dim().1);

            match registration {
                Some(reg) => {
                    izip!(local_frames.iter(), intensity_chunk.axis_iter_mut(Axis(0)), phasor_chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_intensities, mut this_frame_phasors, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_masks_registered(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_phasors,
                                &mut this_frame_intensities,
                                &rois_plane,
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                                *reg.get(&this_frame).unwrap(),
                            )?;
                            Ok(())
                    })?;
                },
                None => {
                    izip!(local_frames.iter(), intensity_chunk.axis_iter_mut(Axis(0)), phasor_chunk.axis_iter_mut(Axis(0)), roi_cycle)
                        .try_for_each(
                            |(&this_frame, mut this_frame_intensities, mut this_frame_phasors, rois_plane)|
                            -> Result<(), CorrosiffError> {
                            sum_phasor_intensity_masks(
                                &mut local_f,
                                &self._ifds[this_frame as usize],
                                &mut this_frame_phasors,
                                &mut this_frame_intensities,
                                &rois_plane,
                                &cos_lookup.view(),
                                &sin_lookup.view(),
                            )?;
                            Ok(())
                        })?;
                },
            }
            Ok(())
            }
        )?;

        Ok((phasor_array, intensity_array))
    }


    /// Copies the tiff/siff header from the currently
    /// opened file to the position of the writer.
    /// 
    /// When finished, this can write either OME-TIFF
    /// compliant files or the default ScanImage format.
    /// 
    /// For now it only does the ScanImage format.
    /// 
    /// ## Arguments
    /// 
    /// * `file` - A mutable reference to a `Write` and `Seek` object
    /// which will be used to write the header to the file.
    /// Goes directly to the beginning of the file to write.
    /// 
    /// * `mode` - A `TiffMode` enum which specifies the mode
    /// to write the file in. For now, only `TiffMode::ScanImage`
    /// is supported.
    /// 
    pub fn write_header_to_file<WriterT: Write + Seek>(
        &self,
        file : &mut WriterT,
        mode : &TiffMode
    ) -> Result<(), CorrosiffError> {
        file.seek(std::io::SeekFrom::Start(0))?;
        match mode {
            TiffMode::ScanImage => {
                self.file_format.write(file)?;
                Ok(())
            },
            TiffMode::OME => {
                Err(
                    CorrosiffError::NotImplementedError
                )
            },
        }
    }

    /// Writes the frames requested into the file specified
    /// by the writer. The saved data is **intensity only**
    /// and is unregistered.
    /// 
    /// ## Arguments
    /// 
    /// * `file` - A mutable reference to a `Write` object
    /// pointing to the location to write the frames to.
    /// 
    /// * `frames` - An optional slice of `u64` values corresponding
    /// to the frame numbers to write to the file. If this is `None`,
    /// all frames are written to the file.
    pub fn write_tiff_frames_to_file<WriterT : Write + Seek>(
        &self,
        file : &mut WriterT,
        frames : Option<&[u64]>,
    ) -> Result<(), CorrosiffError> {
        let mut file_reader = File::open(self._filename.clone())?;

        if frames.is_none() {
            let mut reader_for_ifd = File::open(self._filename.clone())?;
            self.file_format.get_ifd_iter(&mut reader_for_ifd)
            .try_for_each(|ifd| {
                SiffFrame::write_frame_as_tiff(
                    &mut file_reader, 
                    file, 
                    &ifd
                )
            })?;
            return Ok(());
        }

        frames.unwrap().iter().map(|&frame| &self._ifds[frame as usize])
        .try_for_each(|ifd| {
            SiffFrame::write_frame_as_tiff(&mut file_reader,file, ifd)
        })?;

        Ok(())
    }

}

impl Display for SiffReader {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "SiffReader: {}\n{} frames",
            self._filename.to_str().unwrap(),
            self._ifds.len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{get_test_paths, UNCOMPRESSED_FRAME_NUM, COMPRESSED_FRAME_NUM};

    #[test]
    fn test_open_siff() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path);
        assert!(reader.is_ok());
    }

    #[test]
    fn read_frame() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        // Compressed frame
        let frame = reader.get_frames_intensity(&[35], None);
        assert!(frame.is_ok(), "Error: {:?}", frame);
        assert_eq!(frame.unwrap().sum(), 63333);

        let mut reg: RegistrationDict = HashMap::new();
        reg.insert(32, (0, 0));
        // Compressed frame with registration
        let frame = reader.get_frames_intensity(&[35], Some(&reg));
        assert!(frame.is_err());
        reg.insert(35, (0, 0));
        let frame = reader.get_frames_intensity(&[35], Some(&reg));
        if frame.is_err() {
            println!("{:?}", frame);
        }
        assert!(frame.is_ok());
        assert_eq!(frame.unwrap().sum(), 63333);

        // Uncompressed frame
        let frame = reader.get_frames_intensity(&[15], None);
        assert!(frame.is_ok());
        assert_eq!(frame.unwrap().sum(), 794
    );
    }

    use rand::Rng;
    /// Read several frames and test.
    #[test]
    fn read_frames(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        let frames = reader.get_frames_intensity(&[15, 35, 35], None);
        assert!(frames.is_ok());
        let frames = frames.unwrap();
        assert_eq!(frames.index_axis(Axis(0),0).sum(), 794);
        assert_eq!(frames.index_axis(Axis(0),1).sum(), 63333);
        assert_eq!(frames.index_axis(Axis(0),2).sum(), 63333);

        // First 100 frames
        let frames = reader.get_frames_intensity(
            &(0u64..100u64).collect::<Vec<u64>>(),
            None
        );
        println!("{:?}", frames);
        assert!(frames.is_ok());

        let mut frame_vec = vec![35; 40000];
        frame_vec[22] = 15;

        let mut reg = HashMap::<u64, (i32, i32)>::new();
        reg.insert(15, (-5, 10));
        reg.insert(35, (0, 1));

        let frames = reader.get_frames_intensity(&frame_vec, None);

        assert!(frames.is_ok());
        let frames = frames.unwrap();

        assert_eq!(frames.index_axis(Axis(0),22).sum(), 794);

        let mut rng = rand::thread_rng();
        for _ in 0..400 {
            // spot check -- they should all be the same but this makes sure no random
            // elements are wrong.

            assert_eq!(frames.index_axis(Axis(0), rng.gen_range(0..40000)).sum(), 63333);
        }
    }

    #[test]
    fn test_get_frames_flim() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        let frame_nums = [15u64, 35u64];
        let frames = reader.get_frames_flim(&frame_nums, None);
        assert!(frames.is_ok());

        let frame_nums = [12u64, 37u64];
        let frames = reader.get_frames_flim(&frame_nums, None);
        assert!(frames.is_ok());
        let (_lifetime, intensity) = frames.unwrap();
        let just_intensity = reader.get_frames_intensity(&frame_nums, None).unwrap();

        assert_eq!(just_intensity, intensity);

        let frame_nums = (0..300).map(|x| x as u64).collect::<Vec<_>>();
        let frames = reader.get_frames_flim(&frame_nums, None);
        assert!(frames.is_ok());
        
        let (_lifetime, intensity) = frames.unwrap();
        let just_intensity = reader.get_frames_intensity(&frame_nums, None).unwrap();
        
        assert_eq!(just_intensity, intensity);

        //println!("Lifetime : {:?}", lifetime);
    }

    #[test]
    fn test_get_frames_tau_d(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        let frame_nums = [15u64, 35u64];
        let frames = reader.get_frames_tau_d(&frame_nums, None).unwrap();

        let frames_intensity = reader.get_frames_intensity(&frame_nums, None).unwrap();

        assert_eq!(
            frames_intensity,
            frames.sum_axis(Axis(3))
        );

        let reg = RegistrationDict::new();
        let frames = reader.get_frames_tau_d(&frame_nums, Some(&reg)).unwrap();

        let frames_intensity = reader.get_frames_intensity(&frame_nums, Some(&reg)).unwrap();

        assert_eq!(
            frames_intensity,
            frames.sum_axis(Axis(3))
        );
    }

    #[test]
    fn read_histogram() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        
        let framelist = vec![15];
        let hist = reader.get_histogram(&framelist);
        assert!(hist.is_ok());
        let hist = hist.unwrap();
        let frames = reader.get_frames_intensity(&framelist, None);
        assert_eq!(hist.sum(), frames.unwrap().fold(0 as u64, |sum, &x| sum + (x as u64)));
    }

    #[test]
    fn test_masked_histograms() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();

        let mut three_d_roi = Array3::<bool>::from_elem(
            (6, reader.image_dims().unwrap().ydim as usize, reader.image_dims().unwrap().xdim as usize),
            true
        );

        three_d_roi.mapv_inplace(|_| rand::random::<bool>());

        let frames = (0..300).map(|x| x as u64).collect::<Vec<_>>();

        // Test whether the masked and masked_volume methods agree
        // TODO implement...
        
    }

    #[test]
    fn get_frame_metadata(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        let metadata = reader.get_frame_metadata(&[15, 35]);
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();
        assert_eq!(metadata.len(), 2);
        assert_eq!(
            FrameMetadata::frame_number_from_metadata_str(
                &metadata[0].metadata_string
            ),
            15
        );

        assert!(
            FrameMetadata::frame_time_epoch_from_metadata_str(
                &metadata[1].metadata_string
            ) > 1e16 as u64
        );
        assert_eq!(
            FrameMetadata::frame_number_from_metadata_str(
                &metadata[1].metadata_string
            ),
            35
        );
    }

    #[test]
    fn test_get_roi() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        
        let frames = [UNCOMPRESSED_FRAME_NUM as u64, COMPRESSED_FRAME_NUM as u64];

        let mut roi = Array2::<bool>::from_elem(
            reader.image_dims().unwrap().to_tuple(), true
        );

        let all_frame = reader.get_roi_flat(&roi.view(), &frames, None)
            .unwrap();
        assert_eq!(
            all_frame.shape(),
            &[frames.len(), roi.shape()[0]*roi.shape()[1]]
        );

        assert_eq!(
            all_frame.sum_axis(Axis(1)),
            reader.sum_roi_flat(&roi.view(), &frames, None).unwrap()
        );

        // Make an ROI that has some false values
        roi.iter_mut().for_each(|x| *x = rand::random::<bool>());
        let lesser_frame = reader.get_roi_flat(&roi.view(), &frames, None)
            .unwrap();

        assert_eq!(
            lesser_frame.shape(),
            &[frames.len(), roi.iter().filter(|&x| *x).count()]
        );

        assert!(lesser_frame.sum_axis(Axis(1)).iter().all(
            |&x| x < all_frame.sum_axis(Axis(1)).iter().sum::<u64>()
        ));

        assert_eq!(
            lesser_frame.sum_axis(Axis(1)),
            reader.sum_roi_flat(&roi.view(), &frames, None).unwrap()
        );

        // Make a 3D ROI with random true and false values
        let n_planes = 5;
        let mut roi_3d = Array3::<bool>::from_elem(
            (n_planes, reader.image_dims().unwrap().ydim as usize, reader.image_dims().unwrap().xdim as usize),
            true
        );

        let all_frame_3d = reader.get_roi_volume(&roi_3d.view(), &frames, None)
            .unwrap();
        assert_eq!(
            all_frame_3d.shape(),
            &[frames.len(), n_planes*roi_3d.shape()[1]*roi_3d.shape()[2]]
        );

        roi_3d.iter_mut().for_each(|x| *x = rand::random::<bool>());
        let lesser_frame_3d = reader.get_roi_volume(&roi_3d.view(), &frames, None)
            .unwrap();

        assert_eq!(
            lesser_frame_3d.sum_axis(Axis(1)),
            reader.sum_roi_volume(&roi_3d.view(), &frames, None).unwrap()
        );

        // Try it with registration
        let mut reg = HashMap::<u64, (i32, i32)>::new();
        reg.insert(frames[0] as u64, (-15, 12));
        reg.insert(frames[1] as u64, (6,9));
        let lesser_frame_reg = reader.get_roi_flat(&roi.view(), &frames, Some(&reg))
            .unwrap();

        assert_eq!(
            lesser_frame_reg.sum_axis(Axis(1)),
            reader.sum_roi_flat(&roi.view(), &frames, Some(&reg)).unwrap()
        );
    }

    #[test]
    fn test_sum_roi_methods() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();

        let frames = [UNCOMPRESSED_FRAME_NUM as u64, COMPRESSED_FRAME_NUM as u64];
        //let frames = [15, 35];

        // Test the wrong size ROI
        let wrong_roi = Array2::<bool>::from_elem((212, 329), true);

        let sum = reader.sum_roi_flat(&wrong_roi.view(), &frames, None);
        assert!(sum.is_err());

        // Test the correct size ROI        
        let mut roi = Array2::<bool>::from_elem(reader.image_dims().unwrap().to_tuple(), true);
        
        let whole_sum = reader.sum_roi_flat(&roi.view(), &frames, None).unwrap();

        let image_itself = reader.get_frames_intensity(&frames, None).unwrap();

        let image_itself_as_u64 = image_itself.mapv(|x| x as u64);
        assert_eq!(whole_sum, image_itself_as_u64.sum_axis(Axis(1)).sum_axis(Axis(1)));
        
        // Set the ROI to false in the middle
        roi.slice_mut(s![roi.shape()[0]/4..3*roi.shape()[0]/4, ..]).fill(false);
        let lesser_sum = reader.sum_roi_flat(&roi.view(), &frames, None).unwrap();
        assert!(lesser_sum.iter().all(|&x| x < whole_sum.iter().sum::<u64>()));

        // And assert it's what you actually get from multiplying it in!
        let roi_sum = image_itself_as_u64.axis_iter(Axis(0)).map(
            |frame| frame.indexed_iter().filter(
                |(idx, _)| roi[*idx]
            ).map(|(_, &x)| x).sum::<u64>()
        ).collect::<Vec<_>>();

        assert_eq!(roi_sum, lesser_sum.to_vec());

        // Now let's test whether registration gives consistent answers
        let mut reg = HashMap::<u64, (i32, i32)>::new();
        reg.insert(frames[0] as u64, (-15, 12));
        reg.insert(frames[1] as u64, (6,9));

        let whole_sum_reg = reader.sum_roi_flat(&roi.view(), &frames, Some(&reg)).unwrap();

        let shifted_image = reader.get_frames_intensity(&frames, Some(&reg)).unwrap();

        let shifted_roi_sum = shifted_image.mapv(|x| x as u64).axis_iter(Axis(0)).map(
            |frame| frame.iter().zip(roi.iter()).filter(
                |(_, &roi)| roi
            ).map(|(x, _)| *x).sum::<u64>()).collect::<Vec<_>>();

        assert_eq!(shifted_roi_sum, whole_sum_reg.to_vec());

        // Test multiple masks in seperate calls vs. one call to the `masks` method
        let mut rois = Array3::<bool>::from_elem(
        (
            10 as usize,
            reader.image_dims().unwrap().ydim as usize,
            reader.image_dims().unwrap().xdim as usize
            ), false
        );

        use rand;
        for element in rois.iter_mut() {*element = rand::random::<bool>();}

        let frames = (0..300).map(|x| x as u64).collect::<Vec<_>>();

        // First compute the framewise sums for each ROI and convert it to a 2d array
        let roi_sums = rois.axis_iter(Axis(0)).map(
            |roi| reader.sum_roi_flat(&roi, &frames, None).unwrap()
        ).collect::<Vec<_>>();

        let from_indiv = Array2::<u64>::from_shape_fn(
            (frames.len(), rois.dim().0),
            |(frame_idx, roi_idx)| roi_sums[roi_idx][frame_idx]
        );

        let one_pass = reader.sum_rois_flat(&rois.view(), &frames, None).unwrap();

        assert_eq!(from_indiv, one_pass);

        let mut reg_map = RegistrationDict::new();
        frames.iter().for_each(|&x| {
            reg_map.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let roi_sums = rois.axis_iter(Axis(0)).map(
            |roi| reader.sum_roi_flat(&roi, &frames, Some(&reg_map)).unwrap()
        ).collect::<Vec<_>>();

        let from_indiv = Array2::<u64>::from_shape_fn(
            (frames.len(), rois.dim().0),
            |(frame_idx, roi_idx)| roi_sums[roi_idx][frame_idx]
        );

        let one_pass = reader.sum_rois_flat(&rois.view(), &frames, Some(&reg_map)).unwrap();

        assert_eq!(from_indiv, one_pass);
    }

    #[test]
    fn test_3d_roi_mask(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let BIG_FILE_PATH = test_paths.get("BIG_FILE_PATH").expect("BIG_FILE_PATH not found");
        /************
         * Now for 3d!
         */
        let reader = SiffReader::open(BIG_FILE_PATH).unwrap();
        // First 10000 frames
        //let frames = [UNCOMPRESSED_FRAME_NUM as u64, COMPRESSED_FRAME_NUM as u64];
        //let frames = [14 as u64, 40 as u64];
        let frames = (0..500).map(|x| x as u64).collect::<Vec<_>>();
        let frame_dims = reader.image_dims().unwrap().to_tuple();
        let n_planes = 4;

        let mut three_d_roi = Array3::<bool>::from_elem(
            (n_planes, frame_dims.0, frame_dims.1),
            true
        );

        three_d_roi.mapv_inplace(|_| rand::random::<bool>());

        let three_d_sum = reader.sum_roi_volume(&three_d_roi.view(), &frames, None).unwrap();

        let image_itself = reader.get_frames_intensity(&frames, None).unwrap();

        let image_itself_as_u64 = image_itself.mapv(|x| x as u64);

        let piecewise = image_itself_as_u64.axis_iter(Axis(0)).zip(three_d_roi.axis_iter(Axis(0)).cycle()).map(
            |(frame, roi_plane)| frame.indexed_iter().filter(
                |(idx, _)| roi_plane[*idx]
            ).map(|(_, &x)| x).sum::<u64>()
        ).collect::<Vec<_>>();

        assert_eq!(three_d_sum.to_vec(), piecewise);

        let by_plane = three_d_roi.axis_iter(Axis(0)).enumerate().map(
            |(idx,roi)| {
                reader.sum_roi_flat(
                    &roi, 
                    frames.iter().skip(idx).step_by(three_d_roi.dim().0)
                    .map(|x|*x).collect::<Vec<_>>().as_slice(), 
                    None
                ).unwrap()
            }
        ).collect::<Vec<_>>();

        let from_indiv = Array1::<u64>::from_shape_fn(
            frames.len(),
            |frame_idx| 
            by_plane[frame_idx % three_d_roi.dim().0 as usize][frame_idx / three_d_roi.dim().0 as usize]
        );
        assert_eq!(from_indiv, three_d_sum);


        // Test multiple masks in seperate calls vs. one call to the `masks` method
        let mut rois = Array4::<bool>::from_elem(
            (
                7 as usize,
                10 as usize,
                reader.image_dims().unwrap().ydim as usize,
                reader.image_dims().unwrap().xdim as usize
                ), false
            );
    
        use rand;
        for element in rois.iter_mut() {*element = rand::random::<bool>();}

        //let frames = [15, 40];
        let frames = (0..500).map(|x| x as u64).collect::<Vec<_>>();

        // First compute each plane separately
        let roi_sums = rois.axis_iter(Axis(1)).enumerate().map(
            |(idx, roi)| {
                // Takes the idx-th plane, skip rois.dim().1 frames each time
                reader.sum_rois_flat(
                    &roi, 
                    frames.iter().skip(idx).step_by(rois.dim().1)
                    .map(|x| *x).collect::<Vec<_>>().as_slice(),
                    None
                ).unwrap()
            }
        ).collect::<Vec<_>>();

        // Iterate through roi_sums and zip them together
        let from_indiv = Array2::<u64>::from_shape_fn(
            (frames.len(), rois.dim().0),
            |(frame_idx, roi_idx)|
                roi_sums[frame_idx % rois.dim().1 as usize]
                [[(frame_idx/rois.dim().1) as usize, roi_idx]]
        );

        let one_pass = reader.sum_rois_volume(&rois.view(), &frames, None).unwrap();

        assert_eq!(from_indiv, one_pass);
    
        let mut reg_map = RegistrationDict::new();
        frames.iter().for_each(|&x| {
            reg_map.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });
        
        let roi_sums = rois.axis_iter(Axis(1)).enumerate().map(
            |(idx, roi)| {
                // Takes the idx-th plane, skip rois.dim().1 frames each time
                reader.sum_rois_flat(
                    &roi, 
                    frames.iter().skip(idx).step_by(rois.dim().1)
                    .map(|x| *x).collect::<Vec<_>>().as_slice(),
                    Some(&reg_map)
                ).unwrap()
            }
        ).collect::<Vec<_>>();

        // Iterate through roi_sums and zip them together
        let from_indiv = Array2::<u64>::from_shape_fn(
            (frames.len(), rois.dim().0),
            |(frame_idx, roi_idx)|
                roi_sums[frame_idx % rois.dim().1 as usize]
                [[(frame_idx/rois.dim().1) as usize, roi_idx]]
        );

        let one_pass = reader.sum_rois_volume(&rois.view(), &frames, Some(&reg_map)).unwrap();

        assert_eq!(from_indiv, one_pass);
    }

    #[test]
    fn test_2d_roi_flim(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        let frames = (0..300).map(|x| x as u64).collect::<Vec<_>>();

        let mut roi = Array2::<bool>::from_elem(
            (reader.image_dims().unwrap().ydim as usize, reader.image_dims().unwrap().xdim as usize),
            true
        );

        roi.mapv_inplace(|_| rand::random::<bool>());

        let (flim, intensity) = reader.sum_roi_flim_flat(&roi.view(), &frames, None).unwrap();

        let intensity_from_mask = reader.sum_roi_flat(&roi.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let (flim_array, intensity_array) = reader.get_frames_flim(&frames, None).unwrap();

        let flim_sum_from_mask = Array1::<f64>::from_iter(
            izip!(
            flim_array.axis_iter(Axis(0)),
            intensity_array.axis_iter(Axis(0)),
        ).map(
            |(flim_frame, intensity_frame)| {
                izip!(
                    flim_frame.iter(),
                    intensity_frame.iter(),
                    roi.iter()
                ).fold(0.0, |sum, (&flim,   intensity, &mask)| {
                    if mask && flim.is_finite() { sum + flim*(*intensity as f64) } else { sum }
                })
            }
        ));

        let as_float = intensity.mapv(|x| x as f64);
        let empirical = flim_sum_from_mask/as_float;

        // Assert eq without nans
        assert_eq!(
            flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
            empirical.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
        );

        // Now do the same with registration

        let mut reg = RegistrationDict::new();
        frames.iter().for_each(|&x| {
            reg.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let (flim, intensity) = reader.sum_roi_flim_flat(&roi.view(), &frames, Some(&reg)).unwrap();

        let intensity_from_mask = reader.sum_roi_flat(&roi.view(), &frames, Some(&reg)).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let (flim_array, intensity_array) = reader.get_frames_flim(&frames, Some(&reg)).unwrap();

        let flim_sum_from_mask = Array1::<f64>::from_iter(
            izip!(
            flim_array.axis_iter(Axis(0)),
            intensity_array.axis_iter(Axis(0)),
        ).map(
            |(flim_frame, intensity_frame)| {
                izip!(
                    flim_frame.iter(),
                    intensity_frame.iter(),
                    roi.iter()
                ).fold(0.0, |sum, (&flim,   intensity, &mask)| {
                    if mask && flim.is_finite() { sum + flim*(*intensity as f64) } else { sum }
                })
            }
        ));

        let as_float = intensity.mapv(|x| x as f64);
        let empirical = flim_sum_from_mask/as_float;
        
        // Assert eq without nans
        assert_eq!(
            flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
            empirical.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
        );

        //////// MULTIPLE 2D ROIS NOW //////////
        
        let mut rois = Array3::<bool>::from_elem(
            (
                10 as usize,
                reader.image_dims().unwrap().ydim as usize,
                reader.image_dims().unwrap().xdim as usize
            ),
            true
        );

        rois.mapv_inplace(|_| rand::random::<bool>());

        let (flim, intensity) = reader.sum_rois_flim_flat(&rois.view(), &frames, None).unwrap();

        // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_flat(&rois.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_flim_flat(&roi, &frames, None).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // Compare against the single roi mask method
        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        );

        // Now test with registration

        let mut reg = RegistrationDict::new();

        frames.iter().for_each(|&x| {
            reg.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let (flim, intensity) = reader.sum_rois_flim_flat(&rois.view(), &frames, Some(&reg)).unwrap();

        // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_flat(&rois.view(), &frames, Some(&reg)).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_flim_flat(&roi, &frames, Some(&reg)).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // Compare against the single roi mask method

        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        );
    }

    #[test]
    fn test_2d_roi_phasor() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        let frames = (0..300).map(|x| x as u64).collect::<Vec<_>>();

        let mut roi = Array2::<bool>::from_elem(
            (reader.image_dims().unwrap().ydim as usize, reader.image_dims().unwrap().xdim as usize),
            true
        );

        roi.mapv_inplace(|_| rand::random::<bool>());

        let (flim, intensity) = reader.sum_roi_phasor_flat(&roi.view(), &frames, None).unwrap();

        let intensity_from_mask = reader.sum_roi_flat(&roi.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        println!("{:?}", flim);


        // Now do the same with registration
        // Assert eq without nans
        // assert_eq!(
        //     flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
        //     empirical.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
        // );

        //////// MULTIPLE 2D ROIS NOW //////////
        
        let mut rois = Array3::<bool>::from_elem(
            (
                10 as usize,
                reader.image_dims().unwrap().ydim as usize,
                reader.image_dims().unwrap().xdim as usize
            ),
            true
        );

        rois.mapv_inplace(|_| rand::random::<bool>());

        let (flim, intensity) = reader.sum_rois_phasor_flat(&rois.view(), &frames, None).unwrap();

        // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_flat(&rois.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_phasor_flat(&roi, &frames, None).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // // Compare against the single roi mask method
        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        );

        // // Now test with registration

        let mut reg = RegistrationDict::new();

        frames.iter().for_each(|&x| {
            reg.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let (flim, intensity) = reader.sum_rois_phasor_flat(&rois.view(), &frames, Some(&reg)).unwrap();

        // // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_flat(&rois.view(), &frames, Some(&reg)).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_phasor_flat(&roi, &frames, Some(&reg)).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // Compare against the single roi mask method
        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        );
    }

    #[test]
    fn test_3d_roi_flim(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let BIG_FILE_PATH = test_paths.get("BIG_FILE_PATH").expect("BIG_FILE_PATH not found");
        let reader = SiffReader::open(BIG_FILE_PATH).unwrap();
        let frames = (0..300).map(|x| x as u64).collect::<Vec<_>>();

        // 10 slice roi
        let mut roi = Array3::<bool>::from_elem(
            (
                10 as usize,
                reader.image_dims().unwrap().ydim as usize,
                reader.image_dims().unwrap().xdim as usize
            ),
            true
        );

        roi.mapv_inplace(|_| rand::random::<bool>());

        let (flim, intensity) = reader.sum_roi_flim_volume(&roi.view(), &frames, None).unwrap();

        let intensity_from_mask = reader.sum_roi_volume(&roi.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let (flim_array, intensity_array) = reader.get_frames_flim(&frames, None).unwrap();

        let flim_sum_from_mask = Array1::<f64>::from_iter(
            izip!(
            flim_array.axis_iter(Axis(0)),
            intensity_array.axis_iter(Axis(0)),
            roi.axis_iter(Axis(0)).cycle()
        ).map(
            |(flim_frame, intensity_frame, roi_plane)| {
                izip!(
                    flim_frame.iter(),
                    intensity_frame.iter(),
                    roi_plane.iter()
                ).fold(0.0, |sum, (&flim,   intensity, &mask)| {
                    if mask && flim.is_finite() { sum + flim*(*intensity as f64) } else { sum }
                })
            }
        ));

        let as_float = intensity.mapv(|x| x as f64);
        let empirical = flim_sum_from_mask/as_float;

        // Assert eq without nans
        assert_eq!(
            flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
            empirical.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
        );

        // Now do the same with registration

        let mut reg = RegistrationDict::new();
        frames.iter().for_each(|&x| {
            reg.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let (flim, intensity) = reader.sum_roi_flim_volume(&roi.view(), &frames, Some(&reg)).unwrap();

        let intensity_from_mask = reader.sum_roi_volume(&roi.view(), &frames, Some(&reg)).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let (flim_array, intensity_array) = reader.get_frames_flim(&frames, Some(&reg)).unwrap();

        let flim_sum_from_mask = Array1::<f64>::from_iter(
            izip!(
            flim_array.axis_iter(Axis(0)),
            intensity_array.axis_iter(Axis(0)),
            roi.axis_iter(Axis(0)).cycle()
        ).map(
            |(flim_frame, intensity_frame, roi_plane)| {
                izip!(
                    flim_frame.iter(),
                    intensity_frame.iter(),
                    roi_plane.iter()
                ).fold(0.0, |sum, (&flim,   intensity, &mask)| {
                    if mask && flim.is_finite() { sum + flim*(*intensity as f64) } else { sum }
                })
            }
        ));

        let as_float = intensity.mapv(|x| x as f64);
        let empirical = flim_sum_from_mask/as_float;

        // Assert eq without nans
        assert_eq!(
            flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
            empirical.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
        );

        //////// MULTIPLE 3D ROIS NOW //////////

        let mut rois = Array4::<bool>::from_elem(
            (
                7 as usize,
                10 as usize,
                reader.image_dims().unwrap().ydim as usize,
                reader.image_dims().unwrap().xdim as usize
            ),
            true
        );

        rois.mapv_inplace(|_| rand::random::<bool>());

        let (flim, intensity) = reader.sum_rois_flim_volume(&rois.view(), &frames, None).unwrap();

        // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_volume(&rois.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_flim_volume(&roi, &frames, None).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // Compare against the single roi mask method
        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        );

        // Now test with registration

        let mut reg = RegistrationDict::new();

        frames.iter().for_each(|&x| {
            reg.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let (flim, intensity) = reader.sum_rois_flim_volume(&rois.view(), &frames, Some(&reg)).unwrap();

        // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_volume(&rois.view(), &frames, Some(&reg)).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_flim_volume(&roi, &frames, Some(&reg)).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // Compare against the single roi mask method

        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        );
    }

    #[test]
    fn test_3d_roi_phasor() {
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let BIG_FILE_PATH = test_paths.get("BIG_FILE_PATH").expect("BIG_FILE_PATH not found");
        let reader = SiffReader::open(BIG_FILE_PATH).unwrap();
        let frames = (0..300).map(|x| x as u64).collect::<Vec<_>>();

        // 10 slice roi
        let mut roi = Array3::<bool>::from_elem(
            (
                10 as usize,
                reader.image_dims().unwrap().ydim as usize,
                reader.image_dims().unwrap().xdim as usize
            ),
            true
        );

        roi.mapv_inplace(|_| rand::random::<bool>());

        let (_flim, intensity) = reader.sum_roi_phasor_volume(&roi.view(), &frames, None).unwrap();

        let intensity_from_mask = reader.sum_roi_volume(&roi.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        // Now do the same with registration

        let mut reg = RegistrationDict::new();
        frames.iter().for_each(|&x| {
            reg.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let (_flim, intensity) = reader.sum_roi_phasor_volume(&roi.view(), &frames, Some(&reg)).unwrap();

        let intensity_from_mask = reader.sum_roi_volume(&roi.view(), &frames, Some(&reg)).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        //////// MULTIPLE 3D ROIS NOW //////////

        let mut rois = Array4::<bool>::from_elem(
            (
                7 as usize,
                10 as usize,
                reader.image_dims().unwrap().ydim as usize,
                reader.image_dims().unwrap().xdim as usize
            ),
            true
        );

        rois.mapv_inplace(|_| rand::random::<bool>());

        let (flim, intensity) = reader.sum_rois_phasor_volume(&rois.view(), &frames, None).unwrap();

        // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_volume(&rois.view(), &frames, None).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_phasor_volume(&roi, &frames, None).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // Compare against the single roi mask method
        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        );

        // Now test with registration

        let mut reg = RegistrationDict::new();

        frames.iter().for_each(|&x| {
            reg.insert(x, (rand::random::<i32>() % reader.image_dims().unwrap().ydim as i32, rand::random::<i32>() % reader.image_dims().unwrap().xdim as i32));
        });

        let (flim, intensity) = reader.sum_rois_phasor_volume(&rois.view(), &frames, Some(&reg)).unwrap();

        // Compare against just manually masking the intensity array
        let intensity_from_mask = reader.sum_rois_volume(&rois.view(), &frames, Some(&reg)).unwrap();

        assert_eq!(intensity, intensity_from_mask);

        let roi_wise_sums = rois.axis_iter(Axis(0)).map(
            |roi| {
                let (flim, intensity) = reader.sum_roi_phasor_volume(&roi, &frames, Some(&reg)).unwrap();
                (flim, intensity)
            }
        ).collect::<Vec<_>>();

        // Compare against the single roi mask method

        izip!(
            flim.axis_iter(Axis(1)),
            intensity.axis_iter(Axis(1)),
            roi_wise_sums.iter()
        ).for_each(
            |(flim_roiwise, intensity_roiwise,(single_roi_flim, single_roi_intensity))|
            {
                assert_eq!(
                    flim_roiwise.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>(),
                    single_roi_flim.iter().filter_map(|&x| if x.is_finite() { Some(x) } else { None }).collect::<Vec<_>>()
                );
                assert_eq!(intensity_roiwise, single_roi_intensity);
            }
        ); 
    }

    #[test]
    fn time_methods(){
        let test_paths = get_test_paths().expect("Failed to read test paths");
        let test_file_path = test_paths.get("TEST_PATH").expect("TEST_PATH not found");
        let reader = SiffReader::open(test_file_path).unwrap();
        let times = reader.get_experiment_timestamps(&[15, 35]);
        assert!(times.is_ok());
        let times = times.unwrap();
        assert_eq!(times.len(), 2);
        assert_ne!(times[0], times[1]);
        println!("Experiment times : {:?}", times);


        let times = reader.get_epoch_timestamps_laser(&[15, 35]);
        assert!(times.is_ok());
        let times = times.unwrap();
        assert_eq!(times.len(), 2);
        assert_ne!(times[0], times[1]);
        println!("Epoch time (laser) : {:?}", times);

        let times = reader.get_epoch_timestamps_system(&[15, 16, 205]);
        assert!(times.is_ok());
        let times = times.unwrap();
        assert_eq!(times.len(), 3);
        // only updates once every few seconds.
        assert_eq!(times[0], times[1]);
        assert_ne!(times[0], times[2]);

        let both_times = reader.get_epoch_timestamps_both(&[15, 16, 205]);
        assert!(both_times.is_ok());
        let both_times = both_times.unwrap();
        assert_eq!(both_times.shape(), &[2, 3]);
        println!("Both times : {:?}", both_times);
        assert_ne!(both_times[(0, 0)], both_times[(0, 1)]);
        assert_ne!(both_times[(0, 0)], both_times[(0, 2)]);
        assert_eq!(both_times[(1, 0)], both_times[(1, 1)]);
        assert_ne!(both_times[(1, 0)], both_times[(1, 2)]);


    }
}