# Minesweeper + Rust + Machine learning ?

This is a rather simple machinelearning project, made a lot harder by writing everything in rust. Yep, even the machine learning part.

## Project Background

I can't lie, im a rust fanboy. So ever since i stumbled upon [burn-rs](https://burn.dev) i've wanted to try it out.
Now, minesweeper is a rather weird choice, one could of course just write some simple rules that would solve most minesweeper board, but this is more about exploring burn than solving a hard problem.

## Limitations

- Burn features hotswappable backends, but currently this project is only using Cuda.
- Its not very pretty, doesnt work very good either. (Oh well, it does predict somewhat good)

## How to run

1. Clone the repo

```bash
git clone git@github.com:thomsen85/rusty-minesweeper.git
```

2. Train your model

```bash
cargo run --bin cuda-train --release
```

3. Start the app

```bash
cargo run --bin minesweeper
```
