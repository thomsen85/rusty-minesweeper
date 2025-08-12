{
  description = "Cuda development environment";

  inputs = {
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = {
    self,
    nixpkgs,
    rust-overlay,
  }: let
    system = "x86_64-linux";
    overlays = [(import rust-overlay)];
    pkgs = import nixpkgs {
      inherit system overlays;
      config.allowUnfree = true;
    };
  in {
    devShells.${system}.default = pkgs.mkShell {
      name = "cuda-env-shell";
      buildInputs = with pkgs; [
        # Cuda Stuff
        gcc
        stdenv.cc.cc
        stdenv.cc.cc.lib
        ffmpeg
        gnuplot
        clang-tools
        gdb
        bc
        fontconfig

        cudatoolkit
        cudaPackages.cuda_cudart

        # Rust Stuff
        openssl
        pkg-config
        (rust-bin.stable.latest.default.override
          {
            extensions = [
              "rust-src"
              "clippy"
              "rust-analyzer"
            ];
          })
        cargo-generate
      ];
      LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
        pkgs.cudatoolkit
        pkgs.cudaPackages.cuda_cudart
        pkgs.clang-tools
      ];
    };
  };
}
