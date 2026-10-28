# Security

Report vulnerabilities privately to the maintainers (open a GitHub security
advisory on the project repository). Index files are untrusted input:
GraphVec validates all offsets/lengths and never uses pickle or eval on
file contents. Do not open indexes from untrusted sources with checksum
verification disabled in security-sensitive contexts.
