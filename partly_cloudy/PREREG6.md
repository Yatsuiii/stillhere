# How reliable is a per-child "functional maturity" score? (frozen 2026-11-24, before running)

Richardson et al. 2018 scored each child by correlating their ToM (and pain) time course with the adult average, then related that score to other measures. For example, ToM maturity did not differ by false-belief group, p = .73. A score can only correlate with behaviour as far as its own reliability allows. This analysis measures that reliability.

- Score: Pearson r between a child's network series (same cleaning as before, TRs 11-168) and the mean of all 33 adults.
- Split half: the same score computed on TRs 11-89 (half 1) and on TRs 90-168 (half 2).
- Reliability per age group (3, 4, 5, 7, 8-12; adults with leave-self-out) = Pearson r between the half-1 and half-2 scores across people. Spearman-Brown corrected to full length: 2r/(1+r).
- 95% CI by bootstrapping people (5000 resamples). Run for the ToM and pain networks.
- Also reported, so reliability isn't confused with restricted range: the between-person SD of the full-length score per group.
- Fixed in advance: halves differ in content (different scenes), so this is a lower bound on test-retest reliability.
