# Rust scripting languages benchmark

The project goal is to benchmark most popular embedded scripting languages for Rust.

-   [boa](https://boajs.dev)
-   [mlua](https://crates.io/crates/mlua) (Lua 5.4 and Luau)
-   [rhai](https://crates.io/crates/rhai)
-   [rquickjs](https://crates.io/crates/rquickjs)
-   [rune](https://crates.io/crates/rune)
-   [wasmi](https://crates.io/crates/wasmi)
-   [wasmtime](https://crates.io/crates/wasmtime)

The benchmark is designed to cover not only the performance of code evaluation but interoperability with Rust too.

## Getting your own results

Simply run the `bench.py` script to generate images. It requires `cargo criterion` and `python3-matplotlib` package installed.

You also must have `wasm32-unknown-unknown` target installed for webassembly benchmarks.

## Environment

|          |                            |
| -------- | -------------------------- |
| OS       | Ubuntu 22.04, m6i.16xlarge |
| rustc    | v1.83.0                    |
| boa      | v0.19.1                    |
| mlua     | v0.10.2                    |
| rhai     | v1.20.0                    |
| rquickjs | v0.8.1                     |
| rune     | v0.13.4                    |
| wasmi    | v0.40.0                    |
| wasmtime | v27.0.0                    |

## Results

![Sort Rust objects](Sort%20Rust%20objects.png)

[//]: # 1733534949
