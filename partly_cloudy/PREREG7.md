# Does one viewing's ToM maturity score track false-belief performance, and how much can it? (frozen 2026-11-26, before running)

Richardson et al. (2018) related each child's "functional maturity" (their ToM time course correlated with the adult average) to behaviour, and found no difference by false-belief group (p = .73). PREREG6 measured the score's split-half reliability. This file fixes, before any brain score is correlated with any behavioural score, the brain-behaviour test and the reliability ceiling on it.

Only the count of non-missing behavioural values per age group was looked at before freezing: all 122 children have FB_Composite and ToM Booklet scores; adults have neither.

**Brain score.** As PREREG6: Pearson r between the child's ToM network series (same cleaning, TRs 11-168) and the mean of all 33 adults, Fisher-z. Pain network score computed the same way as a control.

**Primary (P1).** Children aged 3-5 (AgeGroup 3yo, 4yo, 5yo; n = 65), where false-belief performance still varies. Spearman partial correlation between ToM maturity and FB_Composite, controlling for age (years, continuous). Two-sided permutation p (10,000 shuffles of the age-residualised ranks); 95% CI by bootstrapping children (10,000). Bar: p < .05.

**Secondary.**
- S1: all 122 children, ToM maturity vs ToM Booklet-Matched-NOFB (booklet without false-belief items), partial for age. Same statistics.
- S2: control network. P1 and S1 repeated with pain maturity in place of ToM maturity. Specificity is claimed only if ToM's partial rho is above pain's (bootstrap CI of the difference excludes 0).
- S3: motion sensitivity. P1 and S1 also partialling the child's motion-spike count.
- S4: reliability ceiling. Brain reliability for the P1 sample = PREREG6's method (half 1 TRs 11-89 vs half 2 TRs 90-168, scores residualised on age group, pooled), Spearman-Brown corrected. Behaviour reliability = .71 (FB composite alpha, as reported by Richardson et al. 2018). Ceiling on observable r = sqrt(rel_brain x .71). Disattenuated r = observed / ceiling, with brain reliability and observed r re-estimated inside each bootstrap resample. If the brain reliability CI includes 0, the disattenuated value is reported as "not estimable" and only the ceiling is given.
- S5: sample size. Children needed for 80% power (two-sided .05) to detect the observable r implied by a true r of .2, .3 and .5 under the measured reliabilities.

**Pre-written lines.**
- P1 fails: "In 65 children aged 3-5, one viewing's ToM maturity score did not track false-belief performance (partial rho X [CI], p = Y). Given the score's reliability, the largest correlation this design could show is about Z, so this null can't separate 'no relation' from 'too noisy to see one'."
- P1 passes: "In 65 children aged 3-5, ToM maturity tracked false-belief performance after age (partial rho X [CI], p = Y)", plus S2's specificity result whatever it is.

Never "understanding", "theory of mind ability" or any clinical wording for the brain score. Group-level claims of Richardson et al. are not tested and are not questioned.
