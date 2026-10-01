// SPDX-License-Identifier: Apache-2.0
//! Resolution of the upstream credential.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

/// The upstream auth header line, resolved at each use.
///
/// A fixed value (flag or env) is read once at startup, so rotating it means
/// restarting. That is fine for a long-lived PAT and wrong for a credential that
/// expires while the proxy runs - a GitHub App installation token lasts an hour.
/// Restarting to rotate is not an option here: a new process starts on a cold
/// mirror and re-clones every repo, which is the cost the proxy exists to avoid.
///
/// `File` closes that gap. The path is re-read on every upstream operation, so an
/// external refresher that rewrites the file is picked up by the next fetch with
/// no restart and no cache loss. Reads are a few microseconds against operations
/// that take a network round trip, so there is nothing to memoize.
#[derive(Clone)]
pub struct AuthHeader(Source);

#[derive(Clone)]
enum Source {
    None,
    Fixed(String),
    File {
        path: PathBuf,
        /// Last value successfully read. A refresher that rewrites the file in
        /// place (rather than renaming over it) leaves a window where the file is
        /// empty or half-written; serving the previous token through that window
        /// beats failing a fetch that would have succeeded. The token outlives the
        /// window by design - it is refreshed well before it expires.
        last_good: Arc<RwLock<Option<String>>>,
    },
}

impl AuthHeader {
    /// No upstream credential: contact upstream anonymously.
    pub fn none() -> Self {
        Self(Source::None)
    }

    /// A header line supplied up front. Blank collapses to [`AuthHeader::none`].
    pub fn fixed(line: impl AsRef<str>) -> Self {
        match sanitize(line.as_ref()) {
            Some(line) => Self(Source::Fixed(line)),
            None => Self::none(),
        }
    }

    /// A header line read from `path` on every use.
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self(Source::File {
            path: path.into(),
            last_good: Arc::new(RwLock::new(None)),
        })
    }

    /// The header line to send upstream now, or `None` to go anonymous.
    pub fn get(&self) -> Option<String> {
        match &self.0 {
            Source::None => None,
            Source::Fixed(line) => Some(line.clone()),
            Source::File { path, last_good } => Self::read_file(path, last_good),
        }
    }

    /// True when this resolves to no credential right now.
    pub fn is_empty(&self) -> bool {
        self.get().is_none()
    }

    fn read_file(path: &Path, last_good: &RwLock<Option<String>>) -> Option<String> {
        let fresh = match std::fs::read_to_string(path) {
            Ok(contents) => sanitize(&contents),
            Err(error) => {
                tracing::warn!(
                    path = %path.display(),
                    %error,
                    "cannot read upstream auth header file",
                );
                None
            }
        };
        match fresh {
            Some(line) => {
                // Only touch the write lock when the value actually moved; the
                // steady state is the same token read over and over.
                if last_good.read().expect("auth header lock").as_deref() != Some(line.as_str()) {
                    *last_good.write().expect("auth header lock") = Some(line.clone());
                }
                Some(line)
            }
            None => last_good.read().expect("auth header lock").clone(),
        }
    }
}

impl fmt::Debug for AuthHeader {
    /// Redacted: the value is a credential and this type ends up inside structs
    /// that are otherwise safe to print.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Source::None => f.write_str("AuthHeader::None"),
            Source::Fixed(_) => f.write_str("AuthHeader::Fixed(<redacted>)"),
            Source::File { path, .. } => write!(f, "AuthHeader::File({})", path.display()),
        }
    }
}

/// Trim surrounding whitespace (a file written by `echo` or mounted from a Secret
/// usually ends in a newline) and reject anything that cannot be one header line.
/// An embedded newline would be passed to git as a `http.extraHeader` value and
/// produce a malformed request rather than an honest failure.
fn sanitize(raw: &str) -> Option<String> {
    let line = raw.trim();
    if line.is_empty() {
        return None;
    }
    if line.contains(['\n', '\r']) {
        tracing::warn!("ignoring upstream auth header: value spans multiple lines");
        return None;
    }
    Some(line.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_returns_the_line_and_blank_collapses_to_none() {
        assert_eq!(
            AuthHeader::fixed("Authorization: Basic xyz")
                .get()
                .as_deref(),
            Some("Authorization: Basic xyz")
        );
        assert_eq!(
            AuthHeader::fixed("  Authorization: Basic xyz\n")
                .get()
                .as_deref(),
            Some("Authorization: Basic xyz")
        );
        assert!(AuthHeader::fixed("   ").is_empty());
        assert!(AuthHeader::none().is_empty());
    }

    #[test]
    fn file_is_reread_on_every_get() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("header");
        std::fs::write(&path, "Authorization: Basic first\n").expect("write");

        let auth = AuthHeader::file(&path);
        assert_eq!(auth.get().as_deref(), Some("Authorization: Basic first"));

        std::fs::write(&path, "Authorization: Basic second\n").expect("write");
        assert_eq!(auth.get().as_deref(), Some("Authorization: Basic second"));
    }

    #[test]
    fn file_falls_back_to_the_last_good_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("header");
        std::fs::write(&path, "Authorization: Basic token\n").expect("write");

        let auth = AuthHeader::file(&path);
        assert_eq!(auth.get().as_deref(), Some("Authorization: Basic token"));

        // Mid-rewrite (truncated) and gone entirely both keep serving the token
        // that was last read, rather than silently dropping to anonymous.
        std::fs::write(&path, "").expect("truncate");
        assert_eq!(auth.get().as_deref(), Some("Authorization: Basic token"));

        std::fs::remove_file(&path).expect("remove");
        assert_eq!(auth.get().as_deref(), Some("Authorization: Basic token"));
    }

    #[test]
    fn missing_file_with_no_history_is_anonymous() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert!(AuthHeader::file(dir.path().join("absent")).is_empty());
    }

    #[test]
    fn multi_line_value_is_rejected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("header");
        std::fs::write(&path, "Authorization: Basic xyz\nX-Smuggled: 1\n").expect("write");
        assert!(AuthHeader::file(&path).is_empty());
    }

    #[test]
    fn debug_never_prints_the_credential() {
        let rendered = format!("{:?}", AuthHeader::fixed("Authorization: Basic secret"));
        assert!(!rendered.contains("secret"), "{rendered}");
    }
}
