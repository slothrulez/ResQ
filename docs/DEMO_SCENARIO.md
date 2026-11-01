# Deterministic demo scenario

The map is a 12-intersection, synthetic Kochi-flavoured mesh with schematic
coordinates. It is not a geographic map. Six nodes are on each side of a
schematic backwater divide, crossed by exactly three named bridges:

- Venduruthy Bridge: Vyttila Jn–Maradu (2 minutes; `DEMO_CLOSURE`)
- Thevara Bridge: Thevara–Willingdon Island (9 minutes)
- Kaloor-Kakkanad Bridge: Kaloor–Kakkanad (13 minutes)

All other roads remain on their respective bank. The mesh has degree 2–4 and
integer edge travel times. Closing Venduruthy Bridge leaves the graph
connected, so all ambulance-to-incident pairs remain reachable. For this
scenario the bridge closure makes the optimal route to Aluva switch from
Venduruthy Bridge to Thevara Bridge.

Bridge names are stored as `name` attributes on the corresponding NetworkX
graph edges; there is no separate edge-name mapping in the scenario API.

Ambulances start at Kaloor, Vyttila Jn, and Edappally. The default incidents
are a critical incident at MG Road (severity 3) and a serious incident at
Aluva (severity 2). The optional third incident is a minor incident at Maradu
(severity 1). Costs below are severity-weighted travel minutes:

| Ambulance | Critical MG Road, open | Serious Aluva, open | Critical MG Road, closed | Serious Aluva, closed |
| --- | ---: | ---: | ---: | ---: |
| Kaloor Unit | 18 | 52 | 18 | 56 |
| Vyttila Unit | 54 | 16 | 54 | 60 |
| Edappally Unit | 36 | 42 | 36 | 50 |

Before closure, the unique 3-by-2 optimum sends Kaloor to MG Road and Vyttila
to Aluva, total 34. After closure, it sends Kaloor to MG Road and Edappally to
Aluva, total 68. The 3-by-3 optimum is also unique in both cases: totals are
49 before and 97 after closure. These fixed integer weights make closure,
reachability, and uniqueness guarantees deterministic and testable.
