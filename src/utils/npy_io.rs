//! Fast binary I/O using NumPy's .npy format
//!
//! This provides ~100x faster writes compared to CSV for large arrays.

use std::fs::File;
use std::io::{Write, BufWriter};

/// Write a 2D array of f64 to .npy format (row-major, C order)
/// Shape: (n_rows, n_cols)
pub fn write_npy_f64(path: &str, data: &[f64], n_rows: usize, n_cols: usize) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, file); // 1MB buffer

    // NPY format header
    let magic = b"\x93NUMPY";
    let version = [1u8, 0u8]; // version 1.0

    // Header dict - describes array
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}), }}",
        n_rows, n_cols
    );

    // Pad header to 64-byte alignment (including magic + version + header_len)
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let total_header_len = prefix_len + header.len() + 1; // +1 for newline
    let padding = (64 - (total_header_len % 64)) % 64;
    let padded_header_len = header.len() + padding + 1;

    writer.write_all(magic)?;
    writer.write_all(&version)?;
    writer.write_all(&(padded_header_len as u16).to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    for _ in 0..padding {
        writer.write_all(b" ")?;
    }
    writer.write_all(b"\n")?;

    // Write data as raw bytes (little-endian f64)
    for &val in data {
        writer.write_all(&val.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

/// Write a 2D array of u32 to .npy format (row-major, C order)
pub fn write_npy_u32(path: &str, data: &[u32], n_rows: usize, n_cols: usize) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, file);

    let magic = b"\x93NUMPY";
    let version = [1u8, 0u8];

    let header = format!(
        "{{'descr': '<u4', 'fortran_order': False, 'shape': ({}, {}), }}",
        n_rows, n_cols
    );

    let prefix_len = 10;
    let total_header_len = prefix_len + header.len() + 1;
    let padding = (64 - (total_header_len % 64)) % 64;
    let padded_header_len = header.len() + padding + 1;

    writer.write_all(magic)?;
    writer.write_all(&version)?;
    writer.write_all(&(padded_header_len as u16).to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    for _ in 0..padding {
        writer.write_all(b" ")?;
    }
    writer.write_all(b"\n")?;

    for &val in data {
        writer.write_all(&val.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

/// Write a 1D array of f64 to .npy format
pub fn write_npy_f64_1d(path: &str, data: &[f64]) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(1 << 20, file);

    let magic = b"\x93NUMPY";
    let version = [1u8, 0u8];

    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({},), }}",
        data.len()
    );

    let prefix_len = 10;
    let total_header_len = prefix_len + header.len() + 1;
    let padding = (64 - (total_header_len % 64)) % 64;
    let padded_header_len = header.len() + padding + 1;

    writer.write_all(magic)?;
    writer.write_all(&version)?;
    writer.write_all(&(padded_header_len as u16).to_le_bytes())?;
    writer.write_all(header.as_bytes())?;
    for _ in 0..padding {
        writer.write_all(b" ")?;
    }
    writer.write_all(b"\n")?;

    for &val in data {
        writer.write_all(&val.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_npy_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        write_npy_f64("/tmp/test_2d.npy", &data, 2, 3).unwrap();
    }

    #[test]
    fn test_write_npy_1d() {
        let data = vec![1.0, 2.0, 3.0];
        write_npy_f64_1d("/tmp/test_1d.npy", &data).unwrap();
    }
}
