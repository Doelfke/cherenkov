use anyhow::{Context, Result};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U32,
    F32,
    F16,
    BF16,
    I64,
}

impl Dtype {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "U32" => Dtype::U32,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I64" => Dtype::I64,
            other => anyhow::bail!("unsupported safetensors dtype {other:?}"),
        })
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::I64 => 8,
            Dtype::U32 | Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Index into `ModelWeights::shards`.
    pub shard: usize,
    /// Absolute byte offset of the tensor data within the shard file.
    pub offset: usize,
    pub nbytes: usize,
}

pub struct Shard {
    pub mmap: Mmap,
}

pub struct ModelWeights {
    pub shards: Vec<Shard>,
    pub tensors: HashMap<String, TensorInfo>,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

#[derive(Deserialize)]
struct IndexFile {
    weight_map: BTreeMap<String, String>,
}

impl ModelWeights {
    /// Map shards as they are on disk, without the realigned copies. Tensor
    /// bytes may then be unaligned for their dtype; callers must copy or
    /// read them bytewise (the qwen4-exp packer, which rewrites everything
    /// into aligned files, is the intended user).
    pub fn load_raw(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let shard_names: Vec<String> = if index_path.exists() {
            let bytes = std::fs::read(&index_path)
                .with_context(|| format!("reading {}", index_path.display()))?;
            let index: IndexFile =
                serde_json::from_slice(&bytes).context("parsing safetensors index")?;
            let mut names: Vec<String> = index.weight_map.into_values().collect();
            names.sort();
            names.dedup();
            names
        } else {
            vec!["model.safetensors".to_string()]
        };

        let mut shards = Vec::with_capacity(shard_names.len());
        let mut tensors = HashMap::new();
        for (shard_idx, name) in shard_names.iter().enumerate() {
            let path = model_dir.join(name);
            let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // Weight files must remain unchanged while mapped.
            let mmap =
                unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
            parse_header(&mmap, shard_idx, &mut tensors)
                .with_context(|| format!("parsing header of {}", path.display()))?;
            shards.push(Shard { mmap });
        }
        Ok(ModelWeights { shards, tensors })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .with_context(|| format!("tensor {name:?} not found"))
    }

    pub fn tensor_bytes(&self, info: &TensorInfo) -> &[u8] {
        &self.shards[info.shard].mmap[info.offset..info.offset + info.nbytes]
    }
}

fn parse_header(
    mmap: &Mmap,
    shard_idx: usize,
    tensors: &mut HashMap<String, TensorInfo>,
) -> Result<()> {
    anyhow::ensure!(mmap.len() >= 8, "file too small for safetensors header");
    let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    anyhow::ensure!(
        8 + header_len <= mmap.len(),
        "safetensors header length {header_len} exceeds file size"
    );
    let header: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&mmap[8..8 + header_len]).context("header JSON")?;
    let data_start = 8 + header_len;
    for (name, value) in header {
        if name == "__metadata__" {
            continue;
        }
        let raw: RawTensor =
            serde_json::from_value(value).with_context(|| format!("tensor entry {name:?}"))?;
        let dtype = Dtype::parse(&raw.dtype)?;
        let nbytes = raw.data_offsets[1] - raw.data_offsets[0];
        let expected: usize = raw.shape.iter().product::<usize>() * dtype.size();
        anyhow::ensure!(
            nbytes == expected,
            "tensor {name:?}: byte span {nbytes} != shape-implied {expected}"
        );
        let offset = data_start + raw.data_offsets[0];
        anyhow::ensure!(
            offset + nbytes <= mmap.len(),
            "tensor {name:?} out of bounds"
        );
        tensors.insert(
            name,
            TensorInfo {
                dtype,
                shape: raw.shape,
                shard: shard_idx,
                offset,
                nbytes,
            },
        );
    }
    Ok(())
}
