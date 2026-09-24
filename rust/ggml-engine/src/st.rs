// SPDX-License-Identifier: Apache-2.0
//! Minimal memory-mapped safetensors reader (single file or sharded).

use anyhow::{anyhow, bail, Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    pub fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::BF16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    file: usize,
    start: usize,
    end: usize,
}

pub struct SafeTensors {
    maps: Vec<Mmap>,
    tensors: HashMap<String, TensorInfo>,
}

impl SafeTensors {
    /// Open every `*.safetensors` file in `dir`.
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading model dir {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|e| e == "safetensors").unwrap_or(false))
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("no .safetensors files in {}", dir.display());
        }
        let mut maps = Vec::new();
        let mut tensors = HashMap::new();
        for (fi, path) in files.iter().enumerate() {
            let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
            // SAFETY: the file is opened read-only and not modified while mapped.
            let map = unsafe { Mmap::map(&f)? };
            if map.len() < 8 {
                bail!("{} is too small to be safetensors", path.display());
            }
            let n = u64::from_le_bytes(map[..8].try_into().unwrap()) as usize;
            let header: serde_json::Value = serde_json::from_slice(&map[8..8 + n])
                .with_context(|| format!("parsing header of {}", path.display()))?;
            let base = 8 + n;
            for (name, v) in header.as_object().ok_or_else(|| anyhow!("bad header"))? {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = match v["dtype"].as_str().unwrap_or("") {
                    "F32" => DType::F32,
                    "F16" => DType::F16,
                    "BF16" => DType::BF16,
                    // Non-float tensors are not used by the supported models.
                    _ => continue,
                };
                let shape = v["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow!("bad shape for {name}"))?
                    .iter()
                    .map(|x| x.as_u64().unwrap() as usize)
                    .collect();
                let offs = v["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow!("bad offsets"))?;
                let start = base + offs[0].as_u64().unwrap() as usize;
                let end = base + offs[1].as_u64().unwrap() as usize;
                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        dtype,
                        shape,
                        file: fi,
                        start,
                        end,
                    },
                );
            }
            maps.push(map);
        }
        Ok(Self { maps, tensors })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing tensor '{name}'"))
    }

    pub fn bytes(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self.info(name)?;
        Ok((info, &self.maps[info.file][info.start..info.end]))
    }

    /// Read a tensor converted to f32.
    pub fn f32(&self, name: &str) -> Result<Vec<f32>> {
        let (info, raw) = self.bytes(name)?;
        Ok(to_f32(info.dtype, raw))
    }

    /// Raw bytes of row `row` of a 2-D tensor.
    pub fn row_bytes(&self, name: &str, row: usize) -> Result<(&TensorInfo, &[u8])> {
        let (info, raw) = self.bytes(name)?;
        if info.shape.len() != 2 {
            bail!("{name} is not 2-D");
        }
        let rb = info.shape[1] * info.dtype.size();
        if row >= info.shape[0] {
            bail!("row {row} out of range for {name}");
        }
        Ok((info, &raw[row * rb..(row + 1) * rb]))
    }
}

pub fn to_f32(dtype: DType, raw: &[u8]) -> Vec<f32> {
    match dtype {
        DType::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect(),
        DType::F16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f16::from_le_bytes(*c).to_f32())
            .collect(),
        DType::BF16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_le_bytes(*c).to_f32())
            .collect(),
    }
}
