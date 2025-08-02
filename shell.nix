{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    python312Packages.opencv4
    ];
  shellHook = "echo 'Nix Shell Started'";
}
