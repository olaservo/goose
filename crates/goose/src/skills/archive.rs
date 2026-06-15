//! Unpacking of archive-distributed skills, per SEP `skills[].archives[]`.
//!
//! A server may distribute a skill as a single `.tar.gz` or `.zip` resource
//! instead of (or alongside) individually-addressable files. The archive
//! unpacks into the skill's virtual file tree with `SKILL.md` at the root.
//!
//! Security (SEP MUST): the host MUST reject path-traversal sequences and
//! absolute paths, reject symlinks/hardlinks resolving outside the skill
//! directory, and enforce a total-unpacked-size limit to prevent
//! decompression bombs. Goose is auto-inject only — unpacked content is
//! held in memory and presented to the model as untrusted text; it is never
//! written to disk or made executable.

use std::collections::HashMap;
use std::io::Read;

/// Cap on total unpacked bytes across all entries — decompression-bomb guard.
const MAX_UNPACKED_BYTES: u64 = 32 * 1024 * 1024;
/// Cap on the number of entries in a single archive.
const MAX_ENTRIES: usize = 4096;

/// The virtual file tree of an unpacked skill: relative path → file bytes.
/// Paths use `/` separators and are relative to the skill root.
pub type SkillTree = HashMap<String, Vec<u8>>;

/// Returns true if this host can unpack the given archive `mediaType`.
pub fn supports_media_type(media_type: &str) -> bool {
    matches!(
        media_type,
        "application/gzip" | "application/x-gzip" | "application/x-tar+gzip" | "application/zip"
    )
}

/// Unpack a skill archive into its virtual file tree, enforcing the SEP's
/// archive-safety requirements. Returns an error (rather than a partial
/// tree) if any entry is unsafe or the size cap is exceeded.
pub fn unpack_skill_archive(bytes: &[u8], media_type: &str) -> Result<SkillTree, String> {
    match media_type {
        "application/gzip" | "application/x-gzip" | "application/x-tar+gzip" => {
            unpack_tar_gz(bytes)
        }
        "application/zip" => unpack_zip(bytes),
        other => Err(format!("unsupported archive mediaType '{}'", other)),
    }
}

/// Normalize and validate an archive entry path. Rejects absolute paths,
/// `..` traversal, drive-letter / UNC prefixes, and empty names. Returns the
/// `/`-separated relative path on success.
fn safe_relative_path(raw: &str) -> Result<String, String> {
    let normalized = raw.replace('\\', "/");
    if normalized.is_empty() {
        return Err("archive entry has an empty path".to_string());
    }
    if normalized.starts_with('/') {
        return Err(format!("archive entry '{}' is an absolute path", raw));
    }
    // A Windows drive prefix (`C:`) survives the `/` fold; reject it.
    if normalized
        .split('/')
        .next()
        .is_some_and(|seg| seg.contains(':'))
    {
        return Err(format!("archive entry '{}' has a drive/scheme prefix", raw));
    }
    if normalized.split('/').any(|seg| seg == "..") {
        return Err(format!("archive entry '{}' contains '..'", raw));
    }
    Ok(normalized)
}

/// Read an archive entry fully into memory while enforcing the cumulative
/// unpack budget. The read is bounded to one byte past the remaining budget,
/// so a single oversized entry is rejected *before* it is wholly buffered —
/// the cumulative-only check would otherwise OOM on a one-entry bomb.
fn read_within_budget(reader: impl Read, total: &mut u64, what: &str) -> Result<Vec<u8>, String> {
    let remaining = MAX_UNPACKED_BYTES.saturating_sub(*total);
    let mut buf = Vec::new();
    let read = reader
        .take(remaining + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("failed to read {}: {}", what, e))? as u64;
    if read > remaining {
        return Err(format!(
            "archive unpacks to more than the {}MiB limit",
            MAX_UNPACKED_BYTES / (1024 * 1024)
        ));
    }
    *total += read;
    Ok(buf)
}

fn unpack_tar_gz(bytes: &[u8]) -> Result<SkillTree, String> {
    use flate2::read::GzDecoder;
    use tar::{Archive, EntryType};

    let mut archive = Archive::new(GzDecoder::new(bytes));
    let entries = archive
        .entries()
        .map_err(|e| format!("failed to read tar archive: {}", e))?;

    let mut tree = SkillTree::new();
    let mut total: u64 = 0;
    let mut count: usize = 0;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("corrupt tar entry: {}", e))?;
        let entry_type = entry.header().entry_type();
        // Reject links outright — they can resolve outside the skill dir, and
        // goose has no on-disk tree for them to point into anyway.
        if matches!(entry_type, EntryType::Symlink | EntryType::Link) {
            return Err("archive contains a symlink or hard link".to_string());
        }
        if !entry_type.is_file() {
            continue; // directories and other metadata entries carry no content
        }
        let path = entry
            .path()
            .map_err(|e| format!("invalid tar entry path: {}", e))?
            .to_string_lossy()
            .into_owned();
        let rel = safe_relative_path(&path)?;

        // Count processed file entries, not surviving tree keys: duplicate
        // paths overwrite in the tree, so `tree.len()` would undercount.
        count += 1;
        if count > MAX_ENTRIES {
            return Err(format!("archive has more than {} entries", MAX_ENTRIES));
        }
        let buf = read_within_budget(&mut entry, &mut total, &format!("tar entry '{}'", rel))?;
        tree.insert(rel, buf);
    }
    Ok(tree)
}

fn unpack_zip(bytes: &[u8]) -> Result<SkillTree, String> {
    use std::io::Cursor;
    use zip::ZipArchive;

    let mut archive =
        ZipArchive::new(Cursor::new(bytes)).map_err(|e| format!("failed to read zip: {}", e))?;

    if archive.len() > MAX_ENTRIES {
        return Err(format!("archive has more than {} entries", MAX_ENTRIES));
    }

    let mut tree = SkillTree::new();
    let mut total: u64 = 0;
    for i in 0..archive.len() {
        let mut file = archive
            .by_index(i)
            .map_err(|e| format!("corrupt zip entry: {}", e))?;
        // `enclosed_name` returns None for traversal/absolute paths.
        let Some(name) = file.enclosed_name() else {
            return Err(format!(
                "zip entry '{}' is unsafe (traversal or absolute path)",
                file.name()
            ));
        };
        // Reject symlinks, mirroring the tar path: a symlink entry's bytes
        // are a link target that could resolve outside the skill dir if the
        // tree is ever materialized to disk.
        if file
            .unix_mode()
            .is_some_and(|mode| mode & 0o170000 == 0o120000)
        {
            return Err(format!(
                "zip entry '{}' is a symlink",
                name.to_string_lossy()
            ));
        }
        if file.is_dir() {
            continue;
        }
        let rel = safe_relative_path(&name.to_string_lossy())?;
        let buf = read_within_budget(&mut file, &mut total, &format!("zip entry '{}'", rel))?;
        tree.insert(rel, buf);
    }
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tar_gz(files: &[(&str, &[u8])]) -> Vec<u8> {
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let mut builder = tar::Builder::new(GzEncoder::new(Vec::new(), Compression::default()));
        for (name, content) in files {
            let mut header = tar::Header::new_gnu();
            header.set_size(content.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, name, *content).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap()
    }

    fn make_zip(files: &[(&str, &[u8])]) -> Vec<u8> {
        use std::io::Write;
        use zip::write::SimpleFileOptions;
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            for (name, content) in files {
                zw.start_file(*name, SimpleFileOptions::default()).unwrap();
                zw.write_all(content).unwrap();
            }
            zw.finish().unwrap();
        }
        buf
    }

    #[test]
    fn test_unpack_tar_gz_happy_path() {
        let bytes = make_tar_gz(&[
            ("SKILL.md", b"skill body"),
            ("references/GUIDE.md", b"guide"),
        ]);
        let tree = unpack_skill_archive(&bytes, "application/gzip").unwrap();
        assert_eq!(tree.get("SKILL.md").unwrap(), b"skill body");
        assert_eq!(tree.get("references/GUIDE.md").unwrap(), b"guide");
    }

    #[test]
    fn test_unpack_zip_happy_path() {
        let bytes = make_zip(&[("SKILL.md", b"zip body"), ("scripts/run.py", b"print()")]);
        let tree = unpack_skill_archive(&bytes, "application/zip").unwrap();
        assert_eq!(tree.get("SKILL.md").unwrap(), b"zip body");
        assert_eq!(tree.get("scripts/run.py").unwrap(), b"print()");
    }

    // Note: the `tar` writer itself refuses to emit a `..` entry, so a
    // traversal tarball can't be built via `make_tar_gz`. The unpack-side
    // guard is exercised directly by `test_safe_relative_path_rejects_*`
    // (shared by both the tar and zip paths) and by `test_zip_rejects_traversal`.

    #[test]
    fn test_zip_rejects_traversal() {
        let bytes = make_zip(&[("../escape.md", b"nope")]);
        assert!(unpack_skill_archive(&bytes, "application/zip").is_err());
    }

    #[test]
    fn test_zip_rejects_symlink() {
        use zip::write::SimpleFileOptions;
        let mut buf = Vec::new();
        {
            let mut zw = zip::ZipWriter::new(std::io::Cursor::new(&mut buf));
            zw.add_symlink("link", "/etc/passwd", SimpleFileOptions::default())
                .unwrap();
            zw.finish().unwrap();
        }
        let err = unpack_skill_archive(&buf, "application/zip").unwrap_err();
        assert!(err.contains("symlink"), "unexpected error: {}", err);
    }

    #[test]
    fn test_unsupported_media_type() {
        assert!(unpack_skill_archive(b"x", "application/x-7z-compressed").is_err());
        assert!(!supports_media_type("application/x-7z-compressed"));
        assert!(supports_media_type("application/gzip"));
        assert!(supports_media_type("application/zip"));
    }

    #[test]
    fn test_rejects_single_oversized_entry() {
        // One entry larger than the cap must be rejected without first
        // buffering the whole thing — the decompression-bomb guard.
        let big = vec![0u8; (MAX_UNPACKED_BYTES + 1) as usize];
        let bytes = make_tar_gz(&[("SKILL.md", &big)]);
        let err = unpack_skill_archive(&bytes, "application/gzip").unwrap_err();
        assert!(err.contains("limit"), "unexpected error: {}", err);

        let bytes = make_zip(&[("SKILL.md", &big)]);
        let err = unpack_skill_archive(&bytes, "application/zip").unwrap_err();
        assert!(err.contains("limit"), "unexpected error: {}", err);
    }

    #[test]
    fn test_tar_entry_count_counts_duplicates() {
        // Duplicate paths must count against MAX_ENTRIES even though they
        // collapse to one tree key.
        let dupes: Vec<(&str, &[u8])> = (0..=MAX_ENTRIES)
            .map(|_| ("SKILL.md", b"x" as &[u8]))
            .collect();
        let bytes = make_tar_gz(&dupes);
        let err = unpack_skill_archive(&bytes, "application/gzip").unwrap_err();
        assert!(err.contains("more than"), "unexpected error: {}", err);
    }

    #[test]
    fn test_safe_relative_path_rejects_absolute_and_drive() {
        assert!(safe_relative_path("/etc/passwd").is_err());
        assert!(safe_relative_path("C:/Windows/x").is_err());
        assert!(safe_relative_path("a/../b").is_err());
        assert_eq!(safe_relative_path("a\\b").unwrap(), "a/b");
    }
}
