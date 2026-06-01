# ARC Math Typesetting Reference

Use this reference for user-facing ARC Markdown that may be rendered to PDF,
including work notes, planning handoffs, domain summaries, ranked ideas, and
note-check reports.

## Markdown Math

- Write mathematical expressions in standard LaTeX Markdown form.
- Use `$...$` for inline math and `$$...$$` for display equations.
- Use LaTeX commands and grouped indices inside math delimiters, for example
  `$\rho$`, `$T_{ab}$`, and `$G^a_{b}$`; do not leave ASCII placeholders such
  as `rho`, `T_ab`, or `G^a_b` in rendered equations.
- Preserve valid UTF-8 math symbols such as `ρ`, `η`, `Δ`, `∫`, and `⟨...⟩`
  when the report will be rendered through XeLaTeX or LuaLaTeX; do not
  transliterate them just for Markdown/PDF export.
- Do not wrap equations in `..`, bare multiline blocks, or fenced code blocks.

## Code Spans

- Do not use Markdown code spans for TeX or math snippets that should render as
  mathematics.
- Bad: `\partial_{x_0}^2`
- Good: $\partial_{x_0}^2$
- Bad: `\hat{\mathcal K}_+ - \hat{\mathcal K}_-`
- Good: $\hat{\mathcal K}_+ - \hat{\mathcal K}_-$
- Use code spans and fenced code blocks only for literal code, JSON, logs,
  shell commands, file paths, stable IDs such as `eq_00009`, statuses such as
  `blocked_for_user`, and verbatim source text that is intentionally not being
  rendered as math.

