# valve shims

A quick n' dirty project to figure out the swaps/purchases for valves.

- Mostly ignores the lower spec, as clearances tighten anyway
- Hungarian algorithm to solve assignment problem
- Threshold configures cutoff for 'swapping' cost - if we have to buy a shim it doesn't matter how out of spec it is
- Quantisation configures rounding for purchaseable shim sizes, e.g. `2.65`, `2.70`

Rainy day ideas:
- [ ] make it a bit more presentable
- [ ] separate valve into shim + bucket, allowing rectangular matrix for swap algorithm (i.e. more shims than buckets)
- [ ] webify
