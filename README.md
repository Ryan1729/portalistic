# Portalistic

Currently in prototype/experimentation stage. Using [live-code-sdl2-opengl-2_1-template](https://github.com/Ryan1729/live-code-sdl2-opengl-2_1-template).

## Compiling release mode

Comment out the line containing `crate-type = ["dylib"]` in the `Cargo.toml` in the `state_manipulation` folder. It would be nice if this step was eliminated, but AFAIK the only way to do that would be to build both the `dylib` and the `rlib` every time. Given how relatively rare release builds are, that seems like a waste of compile time.

Then run `cargo build --release`
