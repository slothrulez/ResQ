# Deterministic demo scenario

The demo uses ten synthetic, Kochi-flavoured intersection names and fixed
layout coordinates. The coordinates only support a future plot; they are not
real geographic coordinates. The road graph is a cycle with integer edge
minutes. The edge between Maradu and Fort Kochi takes 12 minutes; the remaining
cycle edges take 3 minutes, except the Thevara–Vyttila connector, which takes
1 minute.

Three ambulances start at Vyttila Jn, Kakkanad, and Maradu. The default demo
has a critical incident at Kaloor (severity 3) and a serious incident at Fort
Kochi (severity 2). The optional third, minor incident is at Edappally
(severity 1).

`DEMO_CLOSURE` removes the Thevara–Vyttila connector. The remaining graph is a
connected chain, so every ambulance-to-incident pair remains reachable.

The 3-by-2 matching is intentionally sensitive to this closure:

| Assignment | Before closure | After closure |
| --- | ---: | ---: |
| Vyttila → critical Kaloor | 18 | 18 |
| Vyttila → serious Fort Kochi | 8 | 66 |
| Kakkanad → critical Kaloor | 27 | 27 |
| Kakkanad → serious Fort Kochi | 36 | 36 |
| Maradu → critical Kaloor | 45 | 45 |
| Maradu → serious Fort Kochi | 24 | 24 |

Costs are severity-weighted travel minutes. Before the closure the unique
optimum sends Vyttila to Fort Kochi and Kakkanad to Kaloor, totaling 35. After
the closure the unique optimum sends Vyttila to Kaloor and Maradu to Fort
Kochi, totaling 42. The two assignments differ, while every pair remains
reachable in each graph. This deliberate, transparent change makes the
closure behavior testable without relying on real traffic or coordinates.
