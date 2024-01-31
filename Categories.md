Category  |  Count  | Code
---------------------------
roofing   |        18447  |  1

tools-equipment  |       11160 | 4

building-materials  |     5081 | 6

siding  |                 4150 | 5

gutters  |                2433 | 0

insulation  |             1206 | 2

waterproofing  |          1049 | 3

P.s. This is before cleaning

**After Augmenting (roofing and tools-equipment not augmented)**

Category | # | subcategories | subcategories-names
------

building-materials |   17044 |  5 | decking-railing, hvac, lumber-composites, plywood-osb, skylights-windows

gutters       |         8265 | 2 | gutter-accessories, gutter-styles

roofing        |        7326 | 1 | commercial-insulation

insulation      |       5184 | 8 | batt-insulation, blown-in-insulation-equipment, fiberglass-insulation, foam-board-insulation, mineral-wool-insulation, radiant-barriers, roll-insulation, spray-foam-insulation

waterproofing   |       4414 | 6 | above-grade-membranes-coatings, air-vapor-barriers, below-grade-membranes-coatings, damp-proofing-coatings, deck-floor-coatings, plaza-deck-waterproofing

tools-equipment  |      7716 | 13 | air-tools-compressors, cleaning-supplies, drill-bits, generators, hand-tools, job-site-supplies, ladders-scaffolding, nails-screws-fasteners, power-tools, saw-blades, tool-bags-belts, welding-soldering, work-wear-safety-gear

siding         |       3358 | 10 | aluminum-siding, composite-siding, engineered-wood-siding, fiber-cement-siding, steel-siding, stone-venner, trim, vapor-barriers-caulk, vinyl-siding, wood-siding


# Hyperparameters dictionary
```python
hp = {}
hp["image_size"] = 256
hp["num_channels"] = 3 
hp["patch_size"] = 32
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 32
hp["learning_rate"] = 1e-4
hp["num_epochs"] = 500
hp["num_classes"] = 7
hp["class_names"] = ["gutters","roofing","insulation","waterproofing","tools-equipment","siding","building-materials"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1
```
