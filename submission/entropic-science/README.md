# entropic.science submission package

Drop-in Markdown entries for submitting the QRNG Analysis Toolkit to
[entropic.science](https://entropic.science). Each file mirrors the path it
should land at inside `artifacts/entropic-science/src/content/` in the
entropic-science monorepo.

## Files

| Local path | Target path in entropic-science repo |
|---|---|
| `contributors/alexander-bone.md` | `artifacts/entropic-science/src/content/contributors/alexander-bone.md` |
| `projects/qrng-analysis-toolkit/index.md` | `artifacts/entropic-science/src/content/projects/qrng-analysis-toolkit/index.md` |
| `blog/qrng-analysis-toolkit-launch.md` | `artifacts/entropic-science/src/content/blog/qrng-analysis-toolkit-launch.md` |
| `research/nist-sp-800-22.md` | `artifacts/entropic-science/src/content/research/nist-sp-800-22.md` |

## Submission steps

1. Fork the entropic-science repo.
2. Copy these files into the matching paths above.
3. From the workspace root, run:
   ```bash
   pnpm install
   pnpm run typecheck      # schema validation
   pnpm run check:voice    # banned-phrase lint
   pnpm --filter @workspace/entropic-science run dev
   ```
4. Walk to `/labs/projects/qrng-analysis-toolkit`, `/blog/qrng-analysis-toolkit-launch`,
   `/learn/research`, and `/contributors/alexander-bone` and confirm each page renders.
5. Open a PR. The contributor profile must merge before/with the project so the
   `owners: [alexander-bone]` reference resolves.

## Notes

- `status` starts as `draft` on the project and blog post — flip to `live` after
  review.
- `publishedAt` uses today's date (2026-05-17). Adjust if you submit later.
- All entries are `scope: community`.
