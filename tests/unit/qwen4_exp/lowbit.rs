use super::*;

fn source_fixture(dir: &Path) -> ExpertLayout {
    let (mut e, _) = layout(2);
    e.record_stride = PAGE as u64;
    e.record_bytes = 204;
    e.experts = 37; // Multiple worker batches and a partial last batch.
    let bytes: Vec<_> = (0..e.record_stride as usize * e.experts)
        .map(|i| (i.wrapping_mul(37) + i / PAGE) as u8)
        .collect();
    std::fs::write(dir.join("experts.bin"), bytes).unwrap();
    e
}

#[test]
fn combined_stores_match_separate_builds_and_preserve_source() {
    let together = tempfile::tempdir().unwrap();
    let separate = tempfile::tempdir().unwrap();
    let e = source_fixture(together.path());
    source_fixture(separate.path());
    let layouts = ensure_many(together.path(), &e, &[3, 2, 3]).unwrap();
    assert_eq!(layouts.iter().map(|l| l.bits).collect::<Vec<_>>(), [2, 3]);
    for bits in [2, 3] {
        ensure(separate.path(), &e, bits, false).unwrap();
        let (bin, man) = paths(together.path(), bits);
        assert_eq!(
            std::fs::read(&bin).unwrap(),
            std::fs::read(separate.path().join(bin.file_name().unwrap())).unwrap()
        );
        assert_eq!(
            std::fs::read(&man).unwrap(),
            std::fs::read(separate.path().join(man.file_name().unwrap())).unwrap()
        );
    }
    assert_eq!(
        std::fs::read(together.path().join("experts.bin")).unwrap(),
        std::fs::read(separate.path().join("experts.bin")).unwrap()
    );
}

#[test]
fn selected_stores_reuse_valid_files_and_build_only_missing_targets() {
    let dir = tempfile::tempdir().unwrap();
    let e = source_fixture(dir.path());
    ensure_many(dir.path(), &e, &[3]).unwrap();
    assert!(!paths(dir.path(), 2).0.exists());
    assert!(!paths(dir.path(), 2).1.exists());
    let (bin, man) = paths(dir.path(), 3);
    let old_time = std::time::UNIX_EPOCH + std::time::Duration::from_secs(1000);
    for path in [&bin, &man] {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(old_time)
            .unwrap();
    }
    ensure_many(dir.path(), &e, &[2, 3]).unwrap();
    ensure_many(dir.path(), &e, &[2, 3]).unwrap();
    assert_eq!(
        std::fs::metadata(bin).unwrap().modified().unwrap(),
        old_time
    );
    assert_eq!(
        std::fs::metadata(man).unwrap().modified().unwrap(),
        old_time
    );
    assert!(usable(
        dir.path(),
        &e,
        &Layout::new(&e, 2).unwrap(),
        e.experts
    ));
}

#[test]
fn failed_combined_build_never_publishes_ready_manifests() {
    let dir = tempfile::tempdir().unwrap();
    let e = source_fixture(dir.path());
    ensure_many(dir.path(), &e, &[2, 3]).unwrap();
    std::fs::File::options()
        .write(true)
        .open(dir.path().join("experts.bin"))
        .unwrap()
        .set_len(1)
        .unwrap();
    assert!(ensure_many(dir.path(), &e, &[2, 3]).is_err());
    assert!(!paths(dir.path(), 2).1.exists());
    assert!(!paths(dir.path(), 3).1.exists());
}

#[test]
fn invalid_target_fails_before_creating_any_store() {
    let dir = tempfile::tempdir().unwrap();
    let e = source_fixture(dir.path());
    assert!(ensure_many(dir.path(), &e, &[2, 4]).is_err());
    assert!(!paths(dir.path(), 2).0.exists());
}

#[test]
fn disabled_store_construction_fails_without_creating_files() {
    let dir = std::env::temp_dir().join(format!("cherenkov-no-build-{}", std::process::id()));
    assert!(!dir.exists());
    let (e, _) = layout(2);
    let error = ensure_with_policy(&dir, &e, 2, false, false).err().unwrap();
    assert!(error.to_string().contains("policy forbids building"));
    assert!(!dir.exists());
}

fn layout(bits: u32) -> (ExpertLayout, Layout) {
    // A miniature record with the real shapes' divisibility.
    let e = ExpertLayout {
        layers: 1,
        experts: 1,
        inter: 2,
        hidden: 64,
        group: 64,
        record_bytes: 0,
        record_stride: 0,
        layer_prefixes: vec![],
        gate_w: 0,
        up_w: 64,
        down_w: 128,
        gate_s: 192,
        gate_b: 194,
        up_s: 196,
        up_b: 198,
        down_s: 200,
        down_b: 202,
    };
    let l = Layout::new(&e, bits).unwrap();
    (e, l)
}

/// Every slot must be distinct and inside its word, or codes would
/// overwrite one another.
#[test]
fn slots_are_a_permutation() {
    let mut seen = std::collections::HashSet::new();
    for (t, off) in q2_slots() {
        assert!(off <= 30 && off % 2 == 0, "2-bit offset {off} out of range");
        assert!(seen.insert((t, off)), "2-bit slot ({t}, {off}) used twice");
    }
    let mut lo_seen = std::collections::HashSet::new();
    let mut hi_seen = std::collections::HashSet::new();
    for (t, lo, hi) in q3_slots() {
        assert!(lo <= 30 && hi <= 31);
        assert!(lo_seen.insert((t, lo)), "3-bit low slot used twice");
        assert!(hi_seen.insert(hi), "3-bit high slot {hi} used twice");
    }
}

/// Unpack the way the kernel does and check we get q >> 1 (or q >> 2).
#[test]
#[expect(
    clippy::excessive_nesting,
    reason = "Keep the independent kernel-unpacking oracle's word, lane and byte loops explicit"
)]
fn packing_round_trips() {
    for bits in [2u32, 3] {
        let (e, l) = layout(bits);
        let codes = e.inter * e.hidden;
        let mut src = vec![0u8; codes / 2];
        for (i, b) in src.iter_mut().enumerate() {
            *b = ((i * 37 + 11) % 256) as u8;
        }
        let mut dst = vec![0u8; l.mat];
        pack_matrix(&src, codes, bits, &mut dst);
        for chunk in 0..codes / 32 {
            // What the source says.
            let mut want = [0u8; 32];
            for w in 0..4 {
                let x = read_word(&src, chunk * 4 + w);
                for j in 0..8 {
                    want[w * 8 + j] =
                        (((x >> (4 * j)) & 0xF) >> (if bits == 2 { 2 } else { 1 })) as u8;
                }
            }
            // What the kernel would read.
            let got = if bits == 2 {
                unpack_q2_chunk(&dst, chunk)
            } else {
                unpack_q3_chunk(&dst, chunk)
            };
            assert_eq!(got, want, "{bits}-bit chunk {chunk} does not round-trip");
        }
    }
}

fn read_word(bytes: &[u8], index: usize) -> u32 {
    let offset = 4 * index;
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

// Independent transcription of the 2-bit kernel's lane unpacking.
fn unpack_q2_chunk(dst: &[u8], chunk: usize) -> [u8; 32] {
    let mut got = [0u8; 32];
    for t in 0..2 {
        let a = read_word(dst, chunk * 2 + t);
        for p in 0..4u32 {
            let cast = (a >> (2 * p)) & 0x03030303;
            for b in 0..4u32 {
                let q = ((cast >> (8 * b)) & 3) as u8;
                let r = match p {
                    0 => 2 * b,
                    1 => 2 * b + 8,
                    2 => 2 * b + 1,
                    _ => 2 * b + 9,
                };
                got[16 * t + r as usize] = q;
            }
        }
    }
    got
}

// Independent transcription of the 3-bit kernel's lane unpacking.
fn unpack_q3_chunk(dst: &[u8], chunk: usize) -> [u8; 32] {
    let mut got = [0u8; 32];
    let bp = read_word(dst, chunk * 3 + 2);
    for t in 0..2usize {
        let a = read_word(dst, chunk * 3 + t);
        for j in 0..4u32 {
            let lo = (a >> (2 * j)) & 0x03030303;
            let hi = (bp >> (4 * t as u32 + j)) & 0x01010101;
            for b in 0..4u32 {
                let q = 2 * ((lo >> (8 * b)) & 3) + ((hi >> (8 * b)) & 1);
                let r = match j {
                    0 => 2 * b,
                    1 => 2 * b + 8,
                    2 => 2 * b + 1,
                    _ => 2 * b + 9,
                };
                got[16 * t + r as usize] = q as u8;
            }
        }
    }
    got
}
