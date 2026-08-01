# PHASE19 — Layout Polish

Closes the "flat layout" gap named since Phase 8: lightweight section headers
(`#### Sampling` / `#### Size` / `#### Batch` / `#### Options` / `#### Sampler &
Seed`) group the existing controls on both tabs, loosely matching the shipped UI's
category sectioning.

## Deliberately minimal, on purpose

This is a purely cosmetic pass: `gr.Markdown()` headers inserted between existing
`gr.Row()` blocks, in the exact same order those blocks already appeared in. No
component was moved, renamed, or rewired — every input to `run_txt2img()`/
`run_img2img()`/`run_batch_img2img()` and every `.click()`'s `inputs=[...]` list is
untouched, byte-for-byte the same list in the same order as before this phase. The
only diff is markdown strings inserted between existing blocks.

This was a deliberate choice over the shipped UI's actual layout (nested
`FormRow`/`FormGroup`/conditional-visibility panels) given this is explicitly the
lowest-priority, purely-cosmetic item on the list — reorganizing into matching nested
accordions-per-category would cost meaningfully more effort (and more risk of
component-scope mistakes, since deeper nesting increases the chance of referencing a
component before its enclosing `with` block has run) for a visual-only payoff that
doesn't change what the frontend can do.

## Verification

**Build check**: `create_ui()` builds cleanly on both tabs with the new headers.

**Regression check**: `pytest test/test_txt2img.py test/test_img2img.py --no-server`
— identical `2 failed, 13 passed, 1 skipped` baseline as every prior phase (expected,
since no backend code or request-payload logic changed in this phase at all).

**Live UI**: this is the last of the Phase 16-19 batch — bundled into one
consolidated live review with the user rather than checked in isolation, per their
explicit instruction to implement all of this batch together and check once at the
end.

## What this phase did NOT do

- **Not the shipped UI's actual nested-accordion/conditional-visibility layout** — see
  above; this is section headers over the existing flat row structure, not a true
  structural rebuild.
- **No responsive/mobile layout work** — column widths (`scale=4`/`scale=5`) are
  unchanged from every prior phase.
