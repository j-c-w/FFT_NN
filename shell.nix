{ pkgs ? import<nixpkgs> {} }:

with pkgs;

mkShell {
	buildInputs = [ python310 python310Packages.torch python310Packages.torchvision ];
}
