{
  description = "Mote development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      devShells.${system} = {
        default = pkgs.mkShell {
          packages = with pkgs; [
            rustup
            shader-slang
            vulkan-loader
            vulkan-tools
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.vulkan-loader ];
        };

        rocm = pkgs.mkShell {
          packages = with pkgs; [
            cmake
            clang
            ninja
            rustup
            shader-slang
            vulkan-loader
            vulkan-tools
            rocmPackages.clr
            rocmPackages.hipblas-common
            rocmPackages.hipblaslt
            rocmPackages.rocblas
            rocmPackages.rocm-comgr
            rocmPackages.rocm-device-libs
            rocmPackages.rocm-runtime
          ];

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.vulkan-loader
            pkgs.rocmPackages.clr
            pkgs.rocmPackages.hipblaslt
            pkgs.rocmPackages.rocblas
          ];
        };
      };
    };
}
