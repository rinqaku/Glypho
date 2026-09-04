use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{Error, Result};

const MAX_STDOUT_BYTES: usize = 64 * 1024 * 1024;
const MAX_STDERR_BYTES: usize = 1024 * 1024;
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct Output {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

pub(crate) fn run(program: &Path, args: &[OsString], timeout: Duration) -> Result<Output> {
    let display_name = program.display().to_string();
    let mut stdout = TemporaryFile::new("stdout")?;
    let mut stderr = TemporaryFile::new("stderr")?;
    let mut command = Command::new(program);
    command
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout.clone_file()?))
        .stderr(Stdio::from(stderr.clone_file()?));
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    let mut child = command.spawn().map_err(|error| Error::io(program, error))?;
    let started = Instant::now();

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if stdout.len()? > MAX_STDOUT_BYTES as u64 => {
                terminate(&mut child);
                return Err(Error::ProcessOutputTooLarge {
                    program: display_name,
                    stream: "stdout".to_owned(),
                    limit_bytes: MAX_STDOUT_BYTES,
                });
            }
            Ok(None) if stderr.len()? > MAX_STDERR_BYTES as u64 => {
                terminate(&mut child);
                return Err(Error::ProcessOutputTooLarge {
                    program: display_name,
                    stream: "stderr".to_owned(),
                    limit_bytes: MAX_STDERR_BYTES,
                });
            }
            Ok(None) if started.elapsed() < timeout => thread::sleep(Duration::from_millis(10)),
            Ok(None) => {
                terminate(&mut child);
                return Err(Error::ProcessTimedOut {
                    program: display_name,
                    seconds: timeout.as_secs(),
                });
            }
            Err(error) => {
                terminate(&mut child);
                return Err(Error::io(program, error));
            }
        }
    };

    if stdout.len()? > MAX_STDOUT_BYTES as u64 {
        return Err(Error::ProcessOutputTooLarge {
            program: display_name,
            stream: "stdout".to_owned(),
            limit_bytes: MAX_STDOUT_BYTES,
        });
    }
    if stderr.len()? > MAX_STDERR_BYTES as u64 {
        return Err(Error::ProcessOutputTooLarge {
            program: display_name,
            stream: "stderr".to_owned(),
            limit_bytes: MAX_STDERR_BYTES,
        });
    }
    let stdout = stdout.read(MAX_STDOUT_BYTES)?;
    let stderr = stderr.read(MAX_STDERR_BYTES)?;
    if !status.success() {
        return Err(Error::ProcessFailed {
            program: display_name,
            status: status.code(),
            stderr: String::from_utf8_lossy(&stderr).into_owned(),
        });
    }

    Ok(Output { stdout, stderr })
}

fn terminate(child: &mut std::process::Child) {
    // SAFETY: `kill` only receives the negated PID of the process group created above.
    #[cfg(unix)]
    unsafe {
        // The child starts in its own process group, so this also stops descendants.
        let _ = libc::kill(-(child.id() as i32), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = child.kill();
    let _ = child.wait();
}

struct TemporaryFile {
    path: PathBuf,
    file: File,
}

impl TemporaryFile {
    fn new(label: &str) -> Result<Self> {
        let directory = std::env::temp_dir();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();

        for _ in 0..100 {
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                "glypho-{}-{timestamp}-{counter}-{label}.tmp",
                std::process::id()
            ));
            let mut options = OpenOptions::new();
            options.read(true).write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&path) {
                Ok(file) => return Ok(Self { path, file }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(Error::io(&path, error)),
            }
        }

        Err(Error::Io {
            path: Some(directory),
            source: std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "could not create a unique process output file",
            ),
        })
    }

    fn clone_file(&self) -> Result<File> {
        self.file
            .try_clone()
            .map_err(|error| Error::io(&self.path, error))
    }

    fn len(&self) -> Result<u64> {
        self.file
            .metadata()
            .map(|metadata| metadata.len())
            .map_err(|error| Error::io(&self.path, error))
    }

    fn read(&mut self, limit: usize) -> Result<Vec<u8>> {
        self.file
            .seek(SeekFrom::Start(0))
            .map_err(|error| Error::io(&self.path, error))?;
        let mut output = Vec::with_capacity(self.len()?.min(limit as u64) as usize);
        self.file
            .by_ref()
            .take(limit as u64)
            .read_to_end(&mut output)
            .map_err(|error| Error::io(&self.path, error))?;
        Ok(output)
    }
}

impl Drop for TemporaryFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn terminates_a_process_at_the_deadline() {
        let started = Instant::now();
        let error = run(
            Path::new("/bin/sh"),
            &[OsString::from("-c"), OsString::from("sleep 2")],
            Duration::from_millis(20),
        )
        .err()
        .expect("the process should time out");

        assert!(matches!(error, Error::ProcessTimedOut { .. }));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn terminates_descendants_at_the_deadline() {
        let marker = std::env::temp_dir().join(format!(
            "glypho-process-group-test-{}-{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let error = run(
            Path::new("/bin/sh"),
            &[
                OsString::from("-c"),
                OsString::from("(sleep 0.2; : > \"$1\") & wait"),
                OsString::from("glypho-process-test"),
                marker.as_os_str().to_owned(),
            ],
            Duration::from_millis(20),
        )
        .err()
        .expect("the process group should time out");

        assert!(matches!(error, Error::ProcessTimedOut { .. }));
        thread::sleep(Duration::from_millis(300));
        assert!(!marker.exists(), "a descendant survived the timeout");
    }
}
