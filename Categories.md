Category  |  Count  | Code
---------------------------
roofing   |           18447  |  1

tools-equipment  |       11160 | 4

building-materials  |     5081 | 6

siding  |                 4150 | 5

gutters  |                2433 | 0

insulation  |             1206 | 2

waterproofing  |          1049 | 3

P.s. This is before cleaning

**Augmented bunch**

Category

building-materials    17044

gutters                8265

roofing                7326

insulation             5184

waterproofing          4414


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
