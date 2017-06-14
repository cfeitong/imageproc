// build.rs

use std::process::Command;
use std::env;
use std::path::Path;

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(t) => t,
        Err(e) => panic!("{} return the error {}", stringify!($e), e),
    })
}

fn run(cmd: &mut Command) {
    println!("running: {:?}", cmd);
    assert!(t!(cmd.status()).success());
}

fn make() -> &'static str {
    if cfg!(target_os = "freebsd") {"gmake"} else {"make"}
}

fn main() {
    let freeimage_dir = env::current_dir().unwrap()
                        .join(Path::new("3rdparty/FreeImage/Dist"))
                        .into_os_string()
                        .into_string()
                        .unwrap();

    run(Command::new(make())
        .arg("-j4")
        .current_dir("3rdparty/FreeImage"));

    println!("cargo:rustc-link-search=native={}", freeimage_dir);
    println!("cargo:rustc-flags=-l dylib=c++");
}
