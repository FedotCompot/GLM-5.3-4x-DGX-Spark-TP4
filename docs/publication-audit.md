# Public-export audit

## October 10 full-model candidate

Scope: one local follow-up commit on fetched public main `ef85a6e784f51a0356bb62b5abe6438dc7125f61`.
The previously prepared non-fast-forward snapshot is excluded from this outgoing range. Only the requested
export-audit, display-formatting and same-boot needle supplement deltas are carried from the dev tree;
dev ancestry is not imported, merged or pushed. Public main's restored full RigMark section and
sanitized receipt logs are retained. Preparation is local, apart from fetching public main, and performs no push.

The extended detector rejects macOS temporary roots, temporary session directories, both home-directory
roots, tailnet DNS names and every IPv4 address starting with 100, alongside existing endpoint, key and
attribution patterns. Its own source and tracked filenames are scanned; escaped pattern spellings avoid
a blanket file exemption. Regression fixtures independently reject each private path/address form and
check that findings never print the matched bytes. Outgoing-history tests also catch a leak deleted in a
later commit and attribution in commit metadata. Two remaining temporary sidecar references in historical
notes use portable `temporary/` substitutions. Historical exported receipts are sanitized copies, not
exact original identity evidence; original private receipts retain their bytes and numeric precision.

Audit the tracked worktree and the exact outgoing commit range, including metadata and every full tree:

```bash
python3 scripts/public_export_audit.py --history origin/main..HEAD --out "$RECEIPTS/public-outgoing-audit.json"
```

Run this in the public clone; require exactly one outgoing commit whose parent equals fetched public
main. Review the exact outgoing diff separately. These checks make no claim about private dev ancestors.

Broad matches for generic experiment labels, coordination and ownership were reviewed: historical
provenance and an external test CLI flag; benchmark scheduling roles and vLLM coordinator namespaces;
licence terminology, lock/rank/tensor ownership and synthetic test fields. They contain no private
endpoints, credentials or tool attribution. Required dependency and credited organization names remain.
The detector's tool-name alternatives use equivalent escaped spellings and were checked against positive
fixtures; their detection is unchanged.

Offline evidence: [release-1010-offline.json](results/release-1010-offline.json), **670 cases: 644 passed,
26 skipped**, no unresolved test failure. The stock runner reached a native Gloo bind denied by the
sandbox. K-stop's 29 cases were checked with mocked one-rank collectives; five native RoCE Gloo cases
and one TCPStore case remain skipped. Other explicit skips require unavailable image/GPU/platform support.
Native frozen-config import ordering and full-size FP8 preparation also passed with explicit CUDA
library/repack substitutes. None of these CPU checks is GPU or live release qualification.
The serving-source hashes are in [release-1010-source.sha256](results/release-1010-source.sha256).

## Historical release 1006 public-export audit

Scope: tracked release-tip snapshot, not private ancestor commits. Original unsanitized receipts remain in the private local campaign archive. Historical exported receipts have portable substitutions and are not exact original identity evidence.

Initial tracked-tree scan found 188 home-path occurrences, 308 deployed-hostname occurrences, 98 IPv4 spellings, one tailnet name and one non-owner email (the coordinator destination). Seven broad name hits were software dependency/API namespaces, not authorship. No key material or session links were found.

Removed site home paths, deployed host/tailnet names and coordinator email; blanked site configuration in `.env.example`; replaced address fixtures with RFC 5737 TEST-NET values and made the stress sampler consume configured management hosts. Service templates require an explicitly rendered service user. `knapcio@gmail.com` is the only permitted email. Standard loopback and synthetic TEST-NET/invalid-address validation fixtures remain.

Reproduce after staging:

```bash
python3 scripts/public_export_audit.py --out /tmp/release-export-audit.json
git archive HEAD > /tmp/release-export.tar
```

## October 7 finalization audit

The 371-file would-be public snapshot was scanned again after the completed gate was added.
Initial findings: **two deployed-hostname references**, in `GATE-RESULT.md` lines 59 and 184.
Both were replaced with rank labels; no runtime source was changed by this finalization.
There were no site IPs, tailnet names, home paths, key material, non-owner emails or session links.

Final audit: **0 unresolved findings** across 371 text files. Allowed fixtures: 110 loopback/TEST-NET
spellings, two deliberately invalid IPv4 values, and three occurrences of `knapcio@gmail.com`.
A supplemental IPv6, broader hostname/home-path, key and session scan found eight regex artifacts:
four `threading.local()` namespace uses and four Python strided slices. Inspection confirmed that none
is an address or hostname. Software package names and credited organizations remain where needed for
working code, attribution and licence terms.

The prepared public snapshot is a separate local repository with one root commit and no remote.
It contains the sanitized tracked tree, with no private ancestor commits or site configuration.
The public README clone was read only for the comparison. Public publication remains owner-controlled;
no public push is part of this finalization.

## October 10 reduced-gate result finalization

The 24 sparkDash cells and ten quality rows are reproducible offline by `scripts/release_table.py`,
including `--needle-supplement` for same-boot S-class 16K x2 and 250K x2 receipts collected after the
reduced gate. README and GATE-RESULT use the same generated quality table. Fresh gate phases,
supplemental measurements and carried diagnostics are distinguished; fresh qeval x3 remains absent.
Repeat-2 supplemental TTFT is a prefix-cache-hit replay, not an independent cold prefill. The restored
full RigMark tables retain main's published precision; formatter output uses one decimal for TTFT and
observed durations, two for GiB and decode rates, and whole tokens/s for prefill rates. Raw receipts and
the machine-readable release summary preserve full precision.
Receipt paths and SHA256s are in [the gate notes](release-1010-gate.md#recorded-reduced-gate-and-cell-provenance).
The source manifest changes only the offline formatter hash; serving source and launch selectors are unchanged.

The supplement formatter verifies same-boot identity, paired repeats, individual run bindings, registry
correctness and completion before rendering. The matching-count survey remains 0/67, including the
four supplemental runs; its receipt and interpretation are retained in the gate notes and release summary.
Follow-up verification: release/recipe suites **123 passed**, export regressions **3 passed**;
`tests/run_cpu_tests.sh` **521 cases: 501 passed, 20 skipped, exit 0**, no failures. All **69 receipt
hashes** and **111 source-manifest entries** verified. Generated tables and provenance match the
published documents; wrong-boot, missing-repeat, altered-run and registry-failure supplements are rejected.
The tracked tree (**434 text files**) and exact outgoing additions have **zero audit findings**.
The requested literal-pattern grep has **zero matches**, including synthetic test strings: those
fixtures assemble private spellings only at runtime. All nine receipt files cleaned by public main's
latest commit remain byte-for-byte unchanged. The one outgoing commit's metadata and full tree must
also pass `--history origin/main..HEAD`; this follow-up remains local and unpushed.
