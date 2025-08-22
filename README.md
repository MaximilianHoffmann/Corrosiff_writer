# Corrosiff

A `Rust`-based reader for `.siff` file data. An excuse to learn
and use `Rust`, which I have to admit has been pretty enjoyable.

While most functionality is `.siff`-exclusive (because it's built
around arrival time data), it can read frames from `.tiff` files too.

TODOS:

- A real README.

- Testing has gotten reaaaaaal lax...

- More methods for estimating "lifetime" (e.g. RLD, MLE...)

- `mask` methods for `tau_d` or for `.tiff` files.

- Fast timestamp checking without reading as much metadata?
Make a `scan_timestamps` method.

- `get_epoch_timestamps_both` doesn't error if `system` timestamps
don't exist! It will just crash! Because I don't use the `?` correctly.

- `C`-compatible `FFI`.

- More sophisticated macros so that I don't have to manually go through
each file to make every combination of registered / unregistered /
mask / masks + flim. This seems like exactly the type of thing the
`Rust` macro system is perfect for (once I understand it better!). This is
obviously a very bad boilerplate-y v0.0.1 type implementation.

It also seems like this is a very natural place for using better
`struct`s and `trait`s. I think what I want to do is create a trait
for all the varied return types,
```rust
trait FileReadImage {
    pub fn read_raw(...) -> Result<(), IOError>;
    pub fn read_compressed(...)
    pub fn read_raw_registered(...)
    pub fn read_compressed_registered(...)
}

struct MaskArray{}

impl FileReadImage for MaskArray {
    ...
}

```
for things like `sum_mask`, `sum_masks`, etc.
The tricky part is that each one needs some subtly different
args...

- Consider making more useful and interesting `Rust` structs for
return values, rather than just passing back `ndarray` objects. I
could wrap these in, for example, the `SiffFrame` struct buried deep
in the `data` submodule and then would have access to some faster
operations implemented in `Rust` rather than relying on my `Python`
interface to do everything complicated (though access to `numpy`
really is nice...)

# Installation
---------------

This is not yet a publicly available crate, so for now you have to download
it and install it yourself!

First clone the repository:

```
git clone git@github.com:MaimonLab/Corrosiff.git
```

then navigate to the directory containing the repository. You can then
build it with
`cargo build --release`, providing libraries for use by other `Rust`
code (such as the `corrosiffpy` package).

Alternatively, you can simply specify this repository in your
`Cargo.toml`:

```
[dependencies]
corrosiff = { git = "https://github.com/MaimonLab/Corrosiff.git" }
```


# Sections
-----------

## File data

## Image Time

## Image

## Metadata

# Troubleshooting
------------------

Haven't had any problems yet, so I'm not sure what to put here!

# Testing
----------

This is not really resolved -- right now I run all the tests on
local files from my computer! I'm working on hosting remote files
to download and test automatically! I'm just still learning this stuff..

The main modules have several tests built in. From the main
directory `corrosiff`, run

```
cargo test
```

and it will run the test suite and print the results.

If any of the tests fail, let me (SCT) know and I'll do my best
to address the problems.

## Benchmarking

The `corrosiff` library was implemented to
make up for my poor `C/C++` skills. Some of the
`siffreadermodule` calls of the `Python` extension
module were slow -- much slower than I'd like. So
I decided I'd learn `Rust` and see how that changes.

This section documents the speed on the computers
I've tested so far.