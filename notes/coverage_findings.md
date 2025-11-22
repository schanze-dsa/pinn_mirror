# Mirror surface coverage findings

## Observed logs
- Blank-region diagnostics report coverage ratios around 37% for `"MIRROR up"` (triangles vs boundary envelope/convex hull) and mark primary cause as low coverage.
- Surface triangulation logs show only C3D4 elements were converted and no unsupported element types were skipped.

## Implications
- Because the coverage ratio is computed from the area of the triangulated faces relative to the projected surface boundary, a ~37% ratio means the surface definition delivered to the visualizer contains significantly fewer faces than the full annulus geometry.
- Since no element types were skipped, the shortfall likely comes from the surface definition itself (e.g., the ABAQUS surface set contains only a subset of the intended faces, or only some instances), rather than from missing element-type support in the triangulation code.

## What to check next (no code changes)
- Verify the ABAQUS surface definition for `MIRROR up` actually includes all top faces of the annulus and is scoped at the assembly level if multiple instances are present.
- Confirm the extracted surface set is element-based (not node-based) and covers both inner and outer rings; partial selection would directly lower the coverage ratio the visualizer reports.
